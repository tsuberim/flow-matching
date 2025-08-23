import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


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
            # Original behavior: use subset from 3/4 into video
            section_duration = duration * 0.5  # We'll use middle 50% of video
            estimated_subsampled_frames = int(section_duration / self.target_frame_interval)
            
            if estimated_subsampled_frames < self.num_frames:
                print(f"Warning: Video section has only ~{estimated_subsampled_frames} subsampled frames, using all available")
                self.num_frames = estimated_subsampled_frames
            
            # Calculate 3/4 section timing
            three_quarter_time = duration * 0.75
            section_start_time = max(0, three_quarter_time - (self.num_frames * self.target_frame_interval) / 2)
            
            # Convert times back to frame indices for the scan range
            self.start_frame = max(0, int(section_start_time * self.original_fps))
            # Add buffer for time-based sampling
            estimated_frames_needed = int(self.num_frames * self.target_frame_interval * self.original_fps) + 100
            self.end_frame = min(total_frames, self.start_frame + estimated_frames_needed)
            
            print(f"Using frames {self.start_frame:,} to {self.end_frame:,} from 3/4 into video")
        
        print(f"Target size: {self.target_size[0]}x{self.target_size[1]}")
        
        cap.release()
    
    def _preload_frames(self):
        """Preload all frames into memory with optimized batch processing"""
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
        
        # Parallel processing setup
        num_workers = min(mp.cpu_count(), 8)  # Limit workers to avoid too many file handles
        batch_size = max(50, num_frames // (num_workers * 4))  # Adaptive batch size
        
        print(f"Using {num_workers} workers with batch size {batch_size}")
        
        # Split frame indices into batches for parallel processing
        frame_batches = []
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_indices = target_frame_indices[batch_start:batch_end]
            frame_batches.append((self.video_path, batch_indices, self.target_size))
        
        cap.release()  # Close main capture before spawning workers
        
        # Process batches in parallel with fallback
        frames_loaded = 0
        try:
            with mp.Pool(num_workers) as pool:
                with tqdm(total=num_frames, desc="Preloading frames (parallel)") as pbar:
                    # Use imap for better progress tracking
                    for batch_frames in pool.imap(VideoFrameDataset._load_and_process_frame_batch, frame_batches):
                        if batch_frames:
                            # Copy batch results into preallocated tensor
                            batch_size_actual = len(batch_frames)
                            for i, frame_tensor in enumerate(batch_frames):
                                self.frames_tensor[frames_loaded + i] = frame_tensor
                            frames_loaded += batch_size_actual
                            pbar.update(batch_size_actual)
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
        """Static method for parallel frame loading and processing"""
        video_path, frame_indices, target_size = args
        
        # Each worker opens its own video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        batch_frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Process frame
                resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor and normalize
                frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()
                frame_tensor = frame_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
                batch_frames.append(frame_tensor)
            else:
                print(f"Warning: Could not read frame {frame_idx}")
        
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
