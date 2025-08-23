import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import hashlib


class VideoFrameDataset(Dataset):
    """
    PyTorch dataset for video frame sequences
    Extracts sequences of frames from 3/4 into video, resizes to target size with no antialiasing
    """
    
    def __init__(self, video_path, num_frames=None, target_size=(320, 180), sequence_length=32, target_fps=12):
        """
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (None = all frames)
            target_size: Target resolution (width, height)
            sequence_length: Length of each sequence returned (default: 32)
            target_fps: Target FPS for subsampling (default: 12)
        """
        self.video_path = video_path
        self.num_frames = num_frames
        self.target_size = target_size
        self.sequence_length = sequence_length
        self.target_fps = target_fps
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video info and create frame mapping
        self._analyze_video()
        
        # Preload all frames into memory
        print("Preloading all frames into memory...")
        self._preload_frames()
    
    def _analyze_video(self):
        """Analyze video and determine frame range to extract"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / self.original_fps
        
        # Calculate time interval for target FPS
        self.target_frame_interval = 1.0 / self.target_fps if self.target_fps else 1.0 / self.original_fps
        self.original_frame_interval = 1.0 / self.original_fps
        
        print(f"Video info:")
        print(f"  Path: {self.video_path}")
        print(f"  Total frames: {total_frames:,}")
        print(f"  Original FPS: {self.original_fps:.2f}")
        print(f"  Target FPS: {self.target_fps}")
        print(f"  Target frame interval: {self.target_frame_interval:.4f}s")
        print(f"  Duration: {duration:.2f} seconds")
        
        if self.num_frames is None:
            # Use all frames from the entire video, but skip first 30 seconds
            skip_seconds = 30.0
            skip_frames = int(skip_seconds * self.original_fps)
            print(f"Using ALL frames from entire video (skipping first {skip_seconds}s / {skip_frames:,} frames)")
            self.start_frame = min(skip_frames, total_frames - 1)
            self.end_frame = total_frames
            # Estimate how many frames we'll get after subsampling
            usable_duration = duration - skip_seconds
            estimated_subsampled_frames = int(max(0, usable_duration) / self.target_frame_interval)
            print(f"Estimated frames after subsampling: ~{estimated_subsampled_frames:,}")
        else:
            # Load from beginning (after 30s skip) when num_frames is specified
            skip_seconds = 30.0
            skip_frames = int(skip_seconds * self.original_fps)
            print(f"Loading {self.num_frames} frames from beginning (skipping first {skip_seconds}s / {skip_frames:,} frames)")
            
            self.start_frame = min(skip_frames, total_frames - 1)
            # Calculate how many original frames we need to scan to get num_frames subsampled frames
            estimated_frames_needed = int(self.num_frames * self.target_frame_interval * self.original_fps) + 100
            self.end_frame = min(total_frames, self.start_frame + estimated_frames_needed)
            
            # Check if we have enough frames
            usable_duration = (self.end_frame - self.start_frame) / self.original_fps
            estimated_subsampled_frames = int(usable_duration / self.target_frame_interval)
            
            if estimated_subsampled_frames < self.num_frames:
                print(f"Warning: Available section has only ~{estimated_subsampled_frames} subsampled frames, using all available")
                self.num_frames = estimated_subsampled_frames
            
            print(f"Using frames {self.start_frame:,} to {self.end_frame:,} from beginning of video")
        
        print(f"Target size: {self.target_size[0]}x{self.target_size[1]}")
        
        cap.release()
    
    def _get_cache_path(self):
        """Generate cache file path based on video settings"""
        # Create unique hash based on video path and settings
        settings_str = f"{self.video_path}_{self.target_size[0]}x{self.target_size[1]}_{self.target_fps}fps_{self.num_frames}"
        cache_hash = hashlib.md5(settings_str.encode()).hexdigest()[:16]
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(self.video_path), '.video_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, f"frames_{cache_hash}.h5")
    
    def _load_from_h5_cache(self, cache_path):
        """Load frames from H5 cache if valid"""
        try:
            with h5py.File(cache_path, 'r') as f:
                # Validate cache metadata
                if (f.attrs['target_width'] == self.target_size[0] and
                    f.attrs['target_height'] == self.target_size[1] and
                    f.attrs['target_fps'] == self.target_fps and
                    f.attrs['num_frames'] == self.num_frames and
                    f.attrs['video_path'] == self.video_path):
                    
                    # Load frames dataset
                    frames_data = f['frames'][:]
                    self.frames_tensor = torch.from_numpy(frames_data).float()
                    print(f"Loaded {len(self.frames_tensor)} frames from H5 cache!")
                    return True
                else:
                    print("Cache metadata mismatch, regenerating...")
                    return False
        except (OSError, KeyError, ValueError) as e:
            print(f"Cache loading failed: {e}, regenerating...")
            return False
    
    def _save_to_h5_cache(self, cache_path):
        """Save frames to H5 cache with metadata"""
        try:
            with h5py.File(cache_path, 'w') as f:
                # Save frames data with compression
                f.create_dataset('frames', data=self.frames_tensor.numpy(), 
                               compression='gzip', compression_opts=1, shuffle=True)
                
                # Save metadata as attributes
                f.attrs['target_width'] = self.target_size[0]
                f.attrs['target_height'] = self.target_size[1]
                f.attrs['target_fps'] = self.target_fps
                f.attrs['num_frames'] = self.num_frames
                f.attrs['video_path'] = self.video_path
                f.attrs['total_frames'] = len(self.frames_tensor)
                
            print(f"Cached {len(self.frames_tensor)} frames to H5: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save H5 cache: {e}")
            # Remove partial cache file
            if os.path.exists(cache_path):
                os.remove(cache_path)
    
    def _preload_frames(self):
        """Preload frames with H5 caching and optimized batch processing"""
        # Check for H5 cache first
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            if self._load_from_h5_cache(cache_path):
                return  # Successfully loaded from cache
        
        print("Processing frames (will cache to H5)...")
        # Open video capture for preloading
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Disable buffering for more predictable frame access
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # First pass: determine which frame indices we need
        target_frame_indices = self._compute_target_frame_indices()
        
        if not target_frame_indices:
            self.frames_tensor = torch.empty(0, 3, *self.target_size[::-1])
            print("No frames to load")
            return
        
        # Preallocate tensor for better memory efficiency
        num_frames = len(target_frame_indices)
        h, w = self.target_size[1], self.target_size[0]  # target_size is (width, height)
        self.frames_tensor = torch.empty(num_frames, 3, h, w, dtype=torch.float32)
        
        print(f"Preloading {num_frames:,} frames from {target_frame_indices[0]} to {target_frame_indices[-1]}")
        print(f"Preallocated tensor shape: {self.frames_tensor.shape}")
        
        # Optimized parallel processing 
        num_workers = min(mp.cpu_count(), 16)  # Reduced workers to avoid overhead
        batch_size = max(50, min(200, num_frames // num_workers))  # Larger batches for efficiency
        
        print(f"Using {num_workers} workers with batch size {batch_size}")
        
        # Split frame indices into batches
        frame_batches = []
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_indices = target_frame_indices[batch_start:batch_end]
            frame_batches.append((self.video_path, batch_indices, self.target_size))
        
        cap.release()  # Close main capture before spawning workers
        
        # Process batches in parallel 
        frames_loaded = 0
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                with tqdm(total=num_frames, desc="Loading frames", unit="frames") as pbar:
                    # Submit all tasks
                    future_to_batch = {executor.submit(VideoFrameDataset._load_and_process_frame_batch, batch): i 
                                      for i, batch in enumerate(frame_batches)}
                    
                    # Process results as they complete
                    for future in as_completed(future_to_batch):
                        try:
                            batch_frames = future.result()
                            if batch_frames:
                                # Convert numpy arrays to tensors efficiently
                                batch_size_actual = len(batch_frames)
                                for i, frame_array in enumerate(batch_frames):
                                    frame_tensor = torch.from_numpy(frame_array)
                                    self.frames_tensor[frames_loaded + i] = frame_tensor
                                frames_loaded += batch_size_actual
                                pbar.update(batch_size_actual)
                                
                                # Update progress info more frequently
                                if frames_loaded % (batch_size * 2) == 0:  # Every 2 batches
                                    elapsed = pbar.format_dict.get('elapsed', 1)
                                    rate = frames_loaded / elapsed if elapsed > 0 else 0
                                    pbar.set_postfix({"loaded": frames_loaded, "rate": f"{rate:.1f} fps"})
                        except Exception as e:
                            print(f"Batch processing error: {e}")
        except Exception as e:
            print(f"Parallel processing failed ({e}), falling back to sequential processing...")
            # Fallback to sequential processing
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            with tqdm(total=num_frames, desc="Preloading frames (sequential)") as pbar:
                for frame_idx in target_frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        self._process_frame_into_tensor(frame, frames_loaded)
                        frames_loaded += 1
                    pbar.update(1)
            
            cap.release()
        
        # Trim tensor if some frames failed to load
        if frames_loaded < num_frames:
            self.frames_tensor = self.frames_tensor[:frames_loaded]
        
        # Update actual number of frames
        if self.num_frames is None:
            self.num_frames = frames_loaded
        
        print(f"Successfully preloaded {len(self.frames_tensor):,} frames")
        print(f"Tensor shape: {self.frames_tensor.shape}")
        print(f"Effective FPS: {1/self.target_frame_interval:.2f} (target: {self.target_fps})")
        print(f"Memory usage: ~{self.frames_tensor.numel() * 4 / 1024**2:.1f} MB")
        
        # Save to H5 cache for future runs
        self._save_to_h5_cache(cache_path)
    
    def _compute_target_frame_indices(self):
        """Compute which frame indices we need based on time sampling"""
        target_frame_indices = []
        
        # Time-based sampling variables
        current_time = 0.0
        next_target_time = 0.0
        frames_collected = 0
        frame_index = self.start_frame
        
        target_frames = self.num_frames if self.num_frames is not None else float('inf')
        
        while frames_collected < target_frames and frame_index < self.end_frame:
            # Check if current time matches or exceeds next target time
            if current_time >= next_target_time:
                target_frame_indices.append(frame_index)
                frames_collected += 1
                next_target_time += self.target_frame_interval
            
            # Advance time by one original frame interval
            current_time += self.original_frame_interval
            frame_index += 1
        
        return target_frame_indices
    
    @staticmethod
    def _load_and_process_frame_batch(args):
        """Optimized batch frame loading with reduced OpenCV overhead"""
        video_path, frame_indices, target_size = args
        
        # Each worker opens its own video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Optimize OpenCV settings for speed
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)  # Reset position efficiently
        
        # Pre-allocate arrays for better memory performance
        batch_frames = []
        w, h = target_size
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Optimized processing pipeline
                # 1. Resize with fastest interpolation
                resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # 2. Convert BGR to RGB in-place when possible
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # 3. Efficient transpose and normalize in one step
                frame_array = rgb_frame.transpose(2, 0, 1).astype(np.float32)
                frame_array *= (2.0 / 255.0)  # Scale to [0, 2]
                frame_array -= 1.0  # Shift to [-1, 1]
                
                batch_frames.append(frame_array)
        
        cap.release()
        return batch_frames
    
    def _process_frame_into_tensor(self, frame, tensor_idx):
        """Process frame directly into preallocated tensor (more efficient)"""
        # Resize with no antialiasing (INTER_NEAREST)
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert BGR to RGB and normalize in one step
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize directly into preallocated tensor
        # More efficient than creating intermediate tensors
        frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()
        frame_tensor = frame_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
        
        # Copy into preallocated tensor
        self.frames_tensor[tensor_idx] = frame_tensor
    
    def _process_frame(self, frame):
        """Process a single frame: resize and convert to tensor"""
        # Resize with no antialiasing (INTER_NEAREST)
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and move channels first (H,W,C) -> (C,H,W)
        tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()
        
        # Always normalize to [-1, 1]
        tensor = tensor / 127.5 - 1.0
        
        return tensor
    
    def _get_frame_by_index(self, idx):
        """Get a preloaded frame by index from stacked tensor"""
        if idx >= len(self.frames_tensor):
            raise IndexError(f"Frame index {idx} out of range for {len(self.frames_tensor)} preloaded frames")
        return self.frames_tensor[idx]
    
    def __len__(self):
        # Number of possible sequences (accounting for sequence length)
        return max(0, len(self.frames_tensor) - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for {len(self)} sequences")
        
        # Get sequence using tensor slicing (much faster than individual access)
        return self.frames_tensor[idx:idx + self.sequence_length]
    
    def __del__(self):
        """Cleanup: clear preloaded frames tensor"""
        if hasattr(self, 'frames_tensor'):
            del self.frames_tensor
        if hasattr(self, 'frames') and self.frames is not None:
            self.frames.clear()


def find_video_file(video_dir='./videos'):
    """Find the first video file in the videos directory"""
    if not os.path.exists(video_dir):
        return None
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    for filename in os.listdir(video_dir):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            return os.path.join(video_dir, filename)
    
    return None


def create_video_dataset(video_path=None, **kwargs):
    """
    Convenience function to create video dataset
    
    Args:
        video_path: Path to video file (if None, searches ./videos/)
        **kwargs: Additional arguments for VideoFrameDataset (including target_fps)
    
    Returns:
        VideoFrameDataset instance
    """
    if video_path is None:
        video_path = find_video_file()
        if video_path is None:
            raise FileNotFoundError("No video file found in ./videos/ directory")
    
    return VideoFrameDataset(video_path, **kwargs)


def preview_batch(dataset, batch_size=4):
    """
    Create and display a preview of a sample batch of sequences
    
    Args:
        dataset: VideoFrameDataset instance
        batch_size: Number of sequences to show in preview
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))
    
    print(f"Batch shape: {batch.shape}")  # (batch_size, sequence_length, channels, height, width)
    print(f"Batch dtype: {batch.dtype}")
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    # Show first 8 frames from first sequence in batch
    sequence = batch[0]  # Shape: (sequence_length, channels, height, width)
    frames_to_show = min(8, sequence.shape[0])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(frames_to_show):
        # Convert from (C,H,W) to (H,W,C) and denormalize
        frame = sequence[i].permute(1, 2, 0)
        frame = (frame + 1) / 2  # [-1,1] -> [0,1]
        frame = torch.clamp(frame, 0, 1)
        
        axes[i].imshow(frame.numpy())
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i}', fontsize=10)
    
    plt.suptitle(f'Video Sequence Preview (First {frames_to_show} frames from sequence)', fontsize=14)
    plt.tight_layout()
    plt.show()


def save_sample_video(dataset, output_dir='./samples', video_length=64, fps=12):
    """
    Save a sample video from the dataset
    
    Args:
        dataset: VideoFrameDataset instance
        output_dir: Directory to save the video
        video_length: Number of frames to include in the video
        fps: Frames per second for the output video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate how many sequences we need for the target length
    seq_len = dataset.sequence_length
    num_sequences_needed = (video_length + seq_len - 1) // seq_len  # Ceiling division
    
    print(f"Creating sample video of {video_length} frames...")
    print(f"Using {num_sequences_needed} sequences of length {seq_len}")
    
    # Collect frames
    all_frames = []
    for i in range(num_sequences_needed):
        if i >= len(dataset):
            break
        sequence = dataset[i]  # [seq_len, 3, H, W] in range [-1, 1]
        all_frames.append(sequence)
    
    # Concatenate sequences and trim to desired length
    if all_frames:
        video_tensor = torch.cat(all_frames, dim=0)[:video_length]  # [video_length, 3, H, W]
        
        # Get video dimensions
        _, channels, height, width = video_tensor.shape
        
        # Setup video writer
        video_path = os.path.join(output_dir, f'sample_video_{video_length}frames.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height), True)
        
        if not video_writer.isOpened():
            # Try alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = video_path.replace('.mp4', '.avi')
            video_writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height), True)
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to initialize video writer for {video_path}")
        
        print(f"Saving video to: {video_path}")
        print(f"Video specs: {len(video_tensor)} frames, {height}x{width}, {fps} FPS")
        
        # Write frames
        for frame_idx in tqdm(range(len(video_tensor)), desc="Writing frames"):
            frame = video_tensor[frame_idx]  # [3, H, W] in range [-1, 1]
            
            # Convert to numpy and scale to [0, 255]
            frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, 3]
            frame_np = (frame_np + 1) / 2  # Convert [-1, 1] to [0, 1]
            frame_np = np.clip(frame_np, 0, 1)
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Write frame
            success = video_writer.write(frame_bgr)
            if not success:
                print(f"Warning: Failed to write frame {frame_idx}")
        
        video_writer.release()
        
        # Verify file was created
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            print(f"✅ Successfully saved: {video_path} ({file_size:,} bytes)")
            return video_path
        else:
            print(f"❌ Failed to create video file: {video_path}")
            return None
    else:
        print("❌ No frames available in dataset")
        return None


if __name__ == "__main__":
    # Example usage with all frames (12 FPS subsampling)
    print("Loading ALL frames from video...")
    dataset = create_video_dataset(num_frames=None, target_fps=12)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Frame shape: {dataset[0].shape}")
    print(f"Frame dtype: {dataset[0].dtype}")
    print(f"Frame range: [{dataset[0].min():.3f}, {dataset[0].max():.3f}]")
    
    # Show preview
    preview_batch(dataset, batch_size=4)
    
    # Save sample video
    print("\n" + "="*50)
    save_sample_video(dataset, output_dir='./samples', video_length=64, fps=12)
