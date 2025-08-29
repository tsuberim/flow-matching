import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random
from typing import Optional, Tuple
from tqdm import tqdm


class EncodingDataset(Dataset):
    """
    Dataset for encoded VAE parameters (mu, logvar) that supports sequence sampling
    Similar to VideoDataset but works with latent representations
    """
    
    def __init__(self, h5_path: str, sequence_length: int = 8, num_sequences: Optional[int] = None, 
                 sample_latents: bool = True, device: str = 'cpu', max_frames: Optional[int] = None):
        """
        Initialize EncodingDataset
        
        Args:
            h5_path: Path to H5 file with encoded mu/logvar data
            sequence_length: Length of sequences to sample
            num_sequences: Number of sequences to use (None for all available)
            sample_latents: Whether to sample from distribution or just return mu
            device: Device to load data on
            max_frames: Maximum number of frames to load (None for all frames)
        """
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        self.sample_latents = sample_latents
        self.device = device
        
        # Load the H5 file and get metadata
        with h5py.File(h5_path, 'r') as f:
            # Get dataset info
            self.mu_shape = f['mu'].shape
            self.logvar_shape = f['logvar'].shape
            
            # Extract metadata
            self.latent_dim = f.attrs.get('latent_dim', None)
            self.model_size = f.attrs.get('model_size', None)
            self.original_sequence_length = f.attrs.get('sequence_length', 1)
            self.num_original_sequences = f.attrs.get('num_sequences', self.mu_shape[0])
            self.original_frame_size = f.attrs.get('original_frame_size', 'unknown')
            self.vae_checkpoint_path = f.attrs.get('vae_checkpoint_path', 'unknown')
            
            print(f"Loading encoding dataset from: {h5_path}")
            print(f"  Mu shape: {self.mu_shape}")
            print(f"  Logvar shape: {self.logvar_shape}")
            print(f"  Latent dim: {self.latent_dim}")
            print(f"  Model size: {self.model_size}")
            print(f"  Original sequence length: {self.original_sequence_length}")
            print(f"  Original frame size: {self.original_frame_size}")
            print(f"  VAE checkpoint: {self.vae_checkpoint_path}")
            
            # Load data into memory for fast access
            print("Loading mu and logvar into memory...")
            
            # Determine how many frames to load
            total_frames_in_file = self.mu_shape[0]
            
            # Apply max_frames limit first if specified
            if max_frames is not None:
                total_frames_in_file = min(total_frames_in_file, max_frames)
            
            # Then apply num_sequences limit if specified
            if num_sequences is not None and self.original_sequence_length == 1:
                frames_to_load = min(num_sequences, total_frames_in_file)
            else:
                frames_to_load = total_frames_in_file
            
            print(f"  Loading {frames_to_load:,} frames out of {total_frames_in_file:,} available...")
            
            # Load with progress bar
            with tqdm(total=frames_to_load, desc="Loading frames", unit="frames") as pbar:
                # Load in chunks for better progress tracking
                chunk_size = min(1000, max(100, frames_to_load // 50))
                mu_chunks = []
                logvar_chunks = []
                
                for start_idx in range(0, frames_to_load, chunk_size):
                    end_idx = min(start_idx + chunk_size, frames_to_load)
                    
                    mu_chunk = torch.from_numpy(f['mu'][start_idx:end_idx]).float()
                    logvar_chunk = torch.from_numpy(f['logvar'][start_idx:end_idx]).float()
                    
                    mu_chunks.append(mu_chunk)
                    logvar_chunks.append(logvar_chunk)
                    
                    pbar.update(end_idx - start_idx)
                
                # Concatenate chunks
                self.mu_data = torch.cat(mu_chunks, dim=0)
                self.logvar_data = torch.cat(logvar_chunks, dim=0)
        
        # Calculate available sequences based on requested sequence length
        if self.original_sequence_length == 1:
            # Original data is individual frames, we can create sequences
            self.total_frames = self.mu_shape[0]
            self.available_sequences = max(0, self.total_frames - sequence_length + 1)
        else:
            # Original data is already sequences, adjust accordingly
            self.total_frames = self.mu_shape[0] * self.mu_shape[1]  # total frames
            # We need to be careful about sequence boundaries
            if sequence_length > self.original_sequence_length:
                raise ValueError(f"Requested sequence length ({sequence_length}) > original sequence length ({self.original_sequence_length}). "
                               f"Cannot create longer sequences without padding.")
            self.available_sequences = self.mu_shape[0]  # Use original sequences
        
        # Limit number of sequences if requested
        if num_sequences is not None:
            self.available_sequences = min(self.available_sequences, num_sequences)
        
        print(f"  Available sequences: {self.available_sequences:,}")
        print(f"  Requested sequence length: {sequence_length}")
        print(f"  Sample latents: {sample_latents}")
        
        # Move data to device if specified
        if device != 'cpu':
            self.mu_data = self.mu_data.to(device)
            self.logvar_data = self.logvar_data.to(device)
            print(f"  Data moved to: {device}")
    
    def __len__(self) -> int:
        """Return number of available sequences"""
        return self.available_sequences
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sequence of latent codes
        
        Args:
            idx: Sequence index
            
        Returns:
            latent_sequence: [sequence_length, latent_dim, H, W] tensor
        """
        if self.original_sequence_length == 1:
            # Original data is individual frames with shape [N, 1, C, H, W]
            start_idx = idx
            end_idx = start_idx + self.sequence_length
            
            # Get mu and logvar for the sequence and flatten the middle dimension
            mu_seq = self.mu_data[start_idx:end_idx, 0]  # [seq_len, C, H, W]
            logvar_seq = self.logvar_data[start_idx:end_idx, 0]  # [seq_len, C, H, W]
        else:
            # Original data is sequences, extract subsequence
            mu_seq = self.mu_data[idx, :self.sequence_length]
            logvar_seq = self.logvar_data[idx, :self.sequence_length]
        
        if self.sample_latents:
            # Sample from the latent distribution using reparameterization trick
            latent_sequence = self.reparameterize(mu_seq, logvar_seq)
        else:
            # Just return the means
            latent_sequence = mu_seq
        
        return latent_sequence
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution
        
        Args:
            mu: Mean tensor [seq_len, latent_dim, H, W]
            logvar: Log variance tensor [seq_len, latent_dim, H, W]
            
        Returns:
            z: Sampled latent codes [seq_len, latent_dim, H, W]
        """
        # Clamp logvar to prevent extreme values
        logvar_clamped = torch.clamp(logvar, min=-20, max=20)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_mu_logvar(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mu and logvar directly without sampling
        
        Args:
            idx: Sequence index
            
        Returns:
            mu, logvar: Distribution parameters [sequence_length, latent_dim, H, W]
        """
        if self.original_sequence_length == 1:
            start_idx = idx
            end_idx = start_idx + self.sequence_length
            mu_seq = self.mu_data[start_idx:end_idx, 0]  # Remove the singleton dimension
            logvar_seq = self.logvar_data[start_idx:end_idx, 0]
        else:
            mu_seq = self.mu_data[idx, :self.sequence_length]
            logvar_seq = self.logvar_data[idx, :self.sequence_length]
        
        return mu_seq, logvar_seq
    
    def get_info(self) -> dict:
        """Get dataset information"""
        return {
            'total_sequences': self.available_sequences,
            'sequence_length': self.sequence_length,
            'mu_shape': self.mu_shape,
            'logvar_shape': self.logvar_shape,
            'latent_dim': self.latent_dim,
            'model_size': self.model_size,
            'original_sequence_length': self.original_sequence_length,
            'original_frame_size': self.original_frame_size,
            'vae_checkpoint_path': self.vae_checkpoint_path,
            'sample_latents': self.sample_latents,
            'device': self.device
        }
    
    def sample_random_sequence(self) -> torch.Tensor:
        """Sample a random sequence from the dataset"""
        idx = random.randint(0, len(self) - 1)
        return self[idx]
    
    def get_latent_statistics(self) -> dict:
        """Get statistics about the latent distributions"""
        mu_mean = torch.mean(self.mu_data).item()
        mu_std = torch.std(self.mu_data).item()
        logvar_mean = torch.mean(self.logvar_data).item()
        logvar_std = torch.std(self.logvar_data).item()
        var_mean = torch.mean(torch.exp(self.logvar_data)).item()
        var_std = torch.std(torch.exp(self.logvar_data)).item()
        
        return {
            'mu_mean': mu_mean,
            'mu_std': mu_std,
            'logvar_mean': logvar_mean,
            'logvar_std': logvar_std,
            'variance_mean': var_mean,
            'variance_std': var_std
        }


def create_encoding_dataset(h5_path: str, sequence_length: int = 8, num_sequences: Optional[int] = None,
                          sample_latents: bool = True, device: str = 'cpu', max_frames: Optional[int] = None) -> EncodingDataset:
    """
    Factory function to create an EncodingDataset
    
    Args:
        h5_path: Path to encoded dataset H5 file
        sequence_length: Length of sequences to sample
        num_sequences: Number of sequences to use (None for all)
        sample_latents: Whether to sample from distribution or return mu
        device: Device to load data on
        max_frames: Maximum number of frames to load (None for all frames)
        
    Returns:
        EncodingDataset instance
    """
    return EncodingDataset(
        h5_path=h5_path,
        sequence_length=sequence_length,
        num_sequences=num_sequences,
        sample_latents=sample_latents,
        device=device,
        max_frames=max_frames
    )


if __name__ == "__main__":
    # Test the dataset
    import argparse
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from safetensors.torch import load_file
    from vae import create_video_vae
    from utils import get_device
    import os
    
    parser = argparse.ArgumentParser(description="Test EncodingDataset")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to encoded H5 file")
    parser.add_argument("--sequence_length", type=int, default=32, help="Sequence length")
    parser.add_argument("--num_sequences", type=int, default=None, help="Number of sequences")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to load")
    parser.add_argument("--sample_latents", action="store_true", help="Sample latents vs return mu")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--vae_checkpoint", type=str, help="Path to VAE checkpoint for decoding")
    parser.add_argument("--save_images", action="store_true", help="Save decoded images")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = create_encoding_dataset(
        h5_path=args.h5_path,
        sequence_length=args.sequence_length,
        num_sequences=args.num_sequences,
        sample_latents=args.sample_latents,
        max_frames=args.max_frames
    )
    
    # Test dataset
    print(f"\nTesting dataset:")
    print(f"Dataset length: {len(dataset)}")
    
    # Get dataset info for VAE parameters
    info = dataset.get_info()
    latent_dim = info['latent_dim']
    model_size = info['model_size']
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    
    # Get mu/logvar directly
    mu, logvar = dataset.get_mu_logvar(0)
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Get statistics
    stats = dataset.get_latent_statistics()
    print(f"\nLatent statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(dataloader))
    print(f"\nBatch shape: {batch.shape}")
    
    # Test multiple samples if sampling is enabled
    if args.sample_latents:
        print(f"\nTesting multiple samples from same latent distribution...")
        mu, logvar = dataset.get_mu_logvar(0)
        
        # Sample 3 different latent codes from same distribution
        for i in range(3):
            z = dataset.reparameterize(mu, logvar)
            print(f"Sample {i+1} shape: {z.shape}")
            print(f"Sample {i+1} range: [{z.min().item():.3f}, {z.max().item():.3f}]")
    
    # Decode and display first 32 frames if VAE checkpoint provided
    if args.vae_checkpoint:
        print(f"\n🔍 Decoding first 32 frames...")
        
        # Load VAE for decoding
        print(f"Loading VAE from: {args.vae_checkpoint}")
        vae = create_video_vae(latent_dim=latent_dim, model_size=model_size)
        vae = vae.to(device)
        vae.eval()
        
        # Load checkpoint
        try:
            model_state = load_file(args.vae_checkpoint)
            
            # Handle DataParallel loading
            is_dataparallel_checkpoint = any(key.startswith('module.') for key in model_state.keys())
            if is_dataparallel_checkpoint:
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            
            vae.load_state_dict(model_state)
            print(f"✅ VAE checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Error loading VAE checkpoint: {e}")
            print("Skipping frame decoding...")
        else:
            # Get first 32 frames from dataset
            frames_to_decode = min(32, len(dataset))
            print(f"Decoding first {frames_to_decode} frames...")
            
            decoded_frames = []
            with torch.no_grad():
                for i in range(frames_to_decode):
                    # Get frame data
                    if args.sample_latents:
                        frame_data = dataset[i]  # Already sampled
                    else:
                        mu, logvar = dataset.get_mu_logvar(i)
                        frame_data = dataset.reparameterize(mu, logvar)
                    
                    # Decode through VAE
                    frame_data = frame_data.to(device)
                    if len(frame_data.shape) == 4:  # [seq_len, latent_dim, h, w]
                        # Flatten sequence dimension for VAE
                        frame_data = frame_data.view(-1, *frame_data.shape[1:])
                    
                    decoded = vae.decode(frame_data)
                    
                    # Take first frame if sequence
                    if len(decoded.shape) == 4:  # [seq_len, 3, h, w]
                        decoded = decoded[0]  # [3, h, w]
                    
                    # Convert to displayable format
                    frame = decoded.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                    frame = (frame + 1.0) / 2.0  # [-1,1] -> [0,1]
                    frame = np.clip(frame, 0, 1)
                    
                    decoded_frames.append(frame)
            
            print(f"✅ Decoded {len(decoded_frames)} frames")
            
            # Display frames in a grid
            num_frames = len(decoded_frames)
            cols = 8
            rows = (num_frames + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(rows * cols):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
                if i < num_frames:
                    ax.imshow(decoded_frames[i])
                    ax.set_title(f'Frame {i+1}', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=8)
                
                ax.axis('off')
            
            plt.suptitle(f'First {num_frames} Decoded Frames (Sequence Length: {args.sequence_length})', fontsize=14)
            plt.tight_layout()
            
            if args.save_images:
                output_dir = "test_outputs"
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(f"{output_dir}/first_32_frames.png", dpi=150, bbox_inches='tight')
                print(f"💾 Saved frame grid to: {output_dir}/first_32_frames.png")
            else:
                plt.show()
    
    print("🎉 EncodingDataset test completed successfully!")
