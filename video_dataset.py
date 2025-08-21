import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class VideoFrameDataset(Dataset):
    """
    PyTorch dataset for video frames
    Extracts frames from 3/4 into video, resizes to target size with no antialiasing
    """
    
    def __init__(self, video_path, num_frames=10000, target_size=(320, 180)):
        """
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract from 3/4 into video
            target_size: Target resolution (width, height)
        """
        self.video_path = video_path
        self.num_frames = num_frames
        self.target_size = target_size
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video info and frame indices
        self._analyze_video()
        
        # Always preload frames
        print("Preloading frames into memory...")
        self._preload_frames()
    
    def _analyze_video(self):
        """Analyze video and determine frame range to extract"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"Video info:")
        print(f"  Path: {self.video_path}")
        print(f"  Total frames: {total_frames:,}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {duration:.2f} seconds")
        
        if total_frames < self.num_frames:
            print(f"Warning: Video has only {total_frames} frames, using all available")
            self.num_frames = total_frames
        
        # Calculate 3/4 section
        three_quarter_point = int(total_frames * 0.75)
        self.start_frame = max(0, three_quarter_point - self.num_frames // 2)
        self.end_frame = min(total_frames, self.start_frame + self.num_frames)
        
        # Adjust if we go past the end
        if self.end_frame > total_frames:
            self.end_frame = total_frames
            self.start_frame = max(0, total_frames - self.num_frames)
        
        self.num_frames = self.end_frame - self.start_frame
        
        print(f"Using frames {self.start_frame:,} to {self.end_frame:,} from 3/4 into video")
        print(f"Target size: {self.target_size[0]}x{self.target_size[1]}")
        
        cap.release()
    
    def _preload_frames(self):
        """Preload all frames into memory"""
        self.frames = []
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        with tqdm(total=self.num_frames, desc="Loading frames") as pbar:
            for i in range(self.num_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {self.start_frame + i}")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                self.frames.append(processed_frame)
                pbar.update(1)
        
        cap.release()
        print(f"Loaded {len(self.frames):,} frames into memory")
    
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
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        if idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range for {self.num_frames} frames")
        
        # Always use preloaded frames
        return self.frames[idx]


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
        **kwargs: Additional arguments for VideoFrameDataset
    
    Returns:
        VideoFrameDataset instance
    """
    if video_path is None:
        video_path = find_video_file()
        if video_path is None:
            raise FileNotFoundError("No video file found in ./videos/ directory")
    
    return VideoFrameDataset(video_path, **kwargs)


def preview_batch(dataset, batch_size=16):
    """
    Create and display a preview of a sample batch
    
    Args:
        dataset: VideoFrameDataset instance
        batch_size: Number of frames to show in preview
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))
    
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    # Create visualization grid
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if batch_size > 1 else [axes]
    
    for i in range(batch_size):
        # Convert from (C,H,W) to (H,W,C) and denormalize
        frame = batch[i].permute(1, 2, 0)
        frame = (frame + 1) / 2  # [-1,1] -> [0,1]
        frame = torch.clamp(frame, 0, 1)
        
        axes[i].imshow(frame.numpy())
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i}', fontsize=8)
    
    # Hide unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Video Frame Batch Preview ({batch_size} frames)', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    dataset = create_video_dataset(num_frames=1000)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Frame shape: {dataset[0].shape}")
    print(f"Frame dtype: {dataset[0].dtype}")
    print(f"Frame range: [{dataset[0].min():.3f}, {dataset[0].max():.3f}]")
    
    # Show preview
    preview_batch(dataset, batch_size=16)
