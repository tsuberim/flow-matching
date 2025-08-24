"""
Simplified Video Dataset - loads preprocessed H5 files into memory.

This is a much cleaner implementation that assumes video has already been 
preprocessed using preprocess_video.py. Always preloads frames for fast training.
"""

import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm


class VideoFrameDataset(Dataset):
    """
    Simple PyTorch dataset for preprocessed video frames from H5 files
    """
    
    def __init__(self, h5_path, sequence_length=32, num_frames=None):
        """
        Args:
            h5_path: Path to preprocessed H5 file
            sequence_length: Length of each sequence returned
            num_frames: Number of frames to load from H5 (None = load all frames)
        """
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        self.num_frames = num_frames
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        
        # Load metadata
        self._load_metadata()
        
        # Always preload all frames into memory for fast training
        self._preload_frames()
    
    def _load_metadata(self):
        """Load metadata from H5 file"""
        with h5py.File(self.h5_path, 'r') as f:
            # Get frame count from H5 file
            h5_total_frames = f['frames'].shape[0]
            
            # Apply num_frames limit if specified
            if self.num_frames is not None:
                self.total_frames = min(self.num_frames, h5_total_frames)
                if self.num_frames > h5_total_frames:
                    print(f"Warning: Requested {self.num_frames} frames but H5 only has {h5_total_frames}, using {h5_total_frames}")
            else:
                self.total_frames = h5_total_frames
            
            # Load metadata attributes
            self.video_path = f.attrs['video_path'].decode('utf-8') if isinstance(f.attrs['video_path'], bytes) else str(f.attrs['video_path'])
            self.target_width = int(f.attrs['target_width'])
            self.target_height = int(f.attrs['target_height'])
            self.target_fps = float(f.attrs['target_fps'])
            self.original_fps = float(f.attrs.get('original_fps', 30.0))
            self.skip_seconds = float(f.attrs.get('skip_seconds', 30.0))
            
            print(f"H5 Dataset Info:")
            print(f"  Source video: {os.path.basename(self.video_path)}")
            print(f"  H5 total frames: {h5_total_frames:,}")
            if self.num_frames is not None:
                print(f"  Using frames: {self.total_frames:,} (limited from {h5_total_frames:,})")
            else:
                print(f"  Using frames: {self.total_frames:,} (all)")
            print(f"  Frame size: {self.target_width}x{self.target_height}")
            print(f"  Target FPS: {self.target_fps}")
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Possible sequences: {self.__len__():,}")
    
    def _preload_frames(self):
        """Preload all frames into memory with progress bar"""
        print(f"Preloading {self.total_frames:,} frames into memory...")
        
        # Calculate memory usage
        memory_mb = self.total_frames * 3 * self.target_height * self.target_width * 4 / 1024**2
        print(f"Estimated memory usage: ~{memory_mb:.1f} MB")
        
        with h5py.File(self.h5_path, 'r') as f:
            frames_dataset = f['frames']
            
            # Load in chunks with progress bar
            chunk_size = min(1000, max(100, self.total_frames // 50))
            
            # Preallocate tensor
            self.frames_tensor = torch.empty(
                self.total_frames, 3, self.target_height, self.target_width, 
                dtype=torch.float32
            )
            
            with tqdm(total=self.total_frames, desc="Loading frames", unit="frames") as pbar:
                for start_idx in range(0, self.total_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, self.total_frames)
                    
                    # Load chunk from H5 (only up to total_frames limit)
                    chunk_data = frames_dataset[start_idx:end_idx]
                    
                    # Convert to tensor and store
                    self.frames_tensor[start_idx:end_idx] = torch.from_numpy(chunk_data)
                    
                    pbar.update(end_idx - start_idx)
        
        print(f"✅ Successfully loaded {self.total_frames:,} frames into memory")
        print(f"Tensor shape: {self.frames_tensor.shape}")
        print(f"Memory usage: ~{self.frames_tensor.numel() * 4 / 1024**2:.1f} MB")
    
    def __len__(self):
        """Number of possible sequences"""
        return max(0, self.total_frames - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        """Get a sequence of frames"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for {len(self)} sequences")
        
        # Fast access from preloaded tensor
        return self.frames_tensor[idx:idx + self.sequence_length]
    

    def get_frame(self, idx):
        """Get a single frame by index"""
        if idx >= self.total_frames:
            raise IndexError(f"Frame index {idx} out of range for {self.total_frames} frames")
        
        return self.frames_tensor[idx]
    
    def get_info(self):
        """Get dataset information"""
        return {
            'total_frames': self.total_frames,
            'width': self.target_width,
            'height': self.target_height,
            'fps': self.target_fps,
            'sequence_length': self.sequence_length,
            'num_sequences': len(self),
            'source_video': self.video_path,
            'preloaded': True
        }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'frames_tensor') and self.frames_tensor is not None:
            del self.frames_tensor


def create_dataset(h5_path, sequence_length=32, num_frames=None):
    """
    Convenience function to create dataset
    
    Args:
        h5_path: Path to preprocessed H5 file
        sequence_length: Sequence length for training
        num_frames: Number of frames to load from H5 (None = load all frames)
    
    Returns:
        VideoFrameDataset instance
    """
    return VideoFrameDataset(h5_path, sequence_length, num_frames)


def preview_dataset(dataset, num_samples=4):
    """
    Preview some sequences from the dataset
    
    Args:
        dataset: VideoFrameDataset instance
        num_samples: Number of sequences to preview
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(loader))
    
    print(f"Batch shape: {batch.shape}")  # (batch_size, sequence_length, channels, height, width)
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    # Show first 8 frames from first sequence
    sequence = batch[0]  # Shape: (sequence_length, channels, height, width)
    frames_to_show = min(8, sequence.shape[0])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(frames_to_show):
        # Convert from (C,H,W) to (H,W,C) and denormalize
        frame = sequence[i].permute(1, 2, 0)
        frame = (frame + 1) / 2  # [-1,1] -> [0,1]
        frame = torch.clamp(frame, 0, 1)
        
        axes[i].imshow(frame.numpy())
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i}', fontsize=10)
    
    plt.suptitle(f'Video Sequence Preview', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_dataset2.py <h5_file_path>")
        print("First preprocess your video using: python preprocess_video.py")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(h5_path, sequence_length=32, num_frames=200)
    
    # Print info
    info = dataset.get_info()
    print(f"\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test data loading
    print(f"\nTesting data loading...")
    sample_sequence = dataset[0]
    print(f"Sample sequence shape: {sample_sequence.shape}")
    print(f"Sample value range: [{sample_sequence.min():.3f}, {sample_sequence.max():.3f}]")
    
    # Show preview if matplotlib available
    try:
        preview_dataset(dataset, num_samples=2)
    except ImportError:
        print("Install matplotlib to see preview: pip install matplotlib")
