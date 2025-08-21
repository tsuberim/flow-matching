import torch
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from video_dataset import create_video_dataset
from vae import create_video_vae
from utils import get_device


class EmbeddingDataset(Dataset):
    """
    Dataset that pre-embeds video frames using a trained VAE
    All frames are encoded to latent space during initialization
    """
    
    def __init__(self, video_path=None, num_frames=10000, latent_dim=16, 
                 vae_checkpoint_path=None, batch_size=32, cache_path=None):
        """
        Args:
            video_path: Path to video file (if None, searches ./videos/)
            num_frames: Number of frames to use from video
            latent_dim: VAE latent dimension
            vae_checkpoint_path: Path to VAE checkpoint (if None, auto-detects)
            batch_size: Batch size for embedding process
            cache_path: Path to save/load cached embeddings
        """
        self.video_path = video_path
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # Auto-detect VAE checkpoint if not provided
        if vae_checkpoint_path is None:
            vae_checkpoint_path = f'vae_final_dim{latent_dim}.pth'
            if not os.path.exists(vae_checkpoint_path):
                vae_checkpoint_path = f'vae_checkpoint_dim{latent_dim}.pth'
        
        self.vae_checkpoint_path = vae_checkpoint_path
        
        # Auto-generate cache path if not provided
        if cache_path is None:
            video_name = os.path.basename(video_path) if video_path else 'auto'
            cache_path = f'embeddings_{video_name}_{num_frames}frames_dim{latent_dim}.pt'
        
        self.cache_path = cache_path
        
        # Load or create embeddings
        if os.path.exists(self.cache_path):
            print(f"Loading cached embeddings from {self.cache_path}")
            self.embeddings = torch.load(self.cache_path, map_location='cpu')
            print(f"Loaded {len(self.embeddings)} embeddings from cache")
        else:
            print("Creating embeddings from video frames...")
            self.embeddings = self._create_embeddings()
            # Save to cache
            torch.save(self.embeddings, self.cache_path)
            print(f"Saved {len(self.embeddings)} embeddings to {self.cache_path}")
    
    def _load_vae(self):
        """Load the trained VAE model"""
        device = get_device()
        vae = create_video_vae(latent_dim=self.latent_dim).to(device)
        
        try:
            # Try loading final model first
            if self.vae_checkpoint_path.endswith('_final_'):
                vae.load_state_dict(torch.load(self.vae_checkpoint_path, map_location=device))
                print(f"Loaded VAE from final model: {self.vae_checkpoint_path}")
            else:
                # Load from checkpoint format
                checkpoint = torch.load(self.vae_checkpoint_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    vae.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded VAE from checkpoint: {self.vae_checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
                else:
                    vae.load_state_dict(checkpoint)
                    print(f"Loaded VAE state dict from: {self.vae_checkpoint_path}")
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"VAE checkpoint not found: {self.vae_checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE: {e}")
        
        vae.eval()
        return vae, device
    
    def _create_embeddings(self):
        """Create embeddings from video frames using VAE"""
        # Load video dataset
        print("Loading video dataset...")
        video_dataset = create_video_dataset(
            video_path=self.video_path, 
            num_frames=self.num_frames
        )
        
        # Create dataloader
        dataloader = DataLoader(
            video_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,  # Important: keep order
            num_workers=0
        )
        
        # Load VAE
        vae, device = self._load_vae()
        
        # Embed all frames
        embeddings = []
        
        print(f"Embedding {len(video_dataset)} frames using VAE...")
        with torch.no_grad():
            for batch_frames in tqdm(dataloader, desc="Embedding frames"):
                batch_frames = batch_frames.to(device)
                
                # Encode to latent space (get mu, ignore logvar for deterministic encoding)
                mu, logvar = vae.encode(batch_frames)
                
                # Use mean (mu) for deterministic embeddings
                # Alternatively: could use reparameterized samples
                batch_embeddings = mu.cpu()
                
                embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings, dim=0)
        
        print(f"Created embeddings with shape: {all_embeddings.shape}")
        print(f"Embedding range: [{all_embeddings.min():.3f}, {all_embeddings.max():.3f}]")
        
        return all_embeddings
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        """Get embedding at index"""
        return self.embeddings[idx]
    
    def get_embedding_info(self):
        """Get information about the embeddings"""
        if len(self.embeddings) > 0:
            sample = self.embeddings[0]
            return {
                'num_embeddings': len(self.embeddings),
                'embedding_shape': tuple(sample.shape),
                'dtype': sample.dtype,
                'device': sample.device,
                'range': (self.embeddings.min().item(), self.embeddings.max().item()),
                'mean': self.embeddings.mean().item(),
                'std': self.embeddings.std().item()
            }
        return {}


def create_embedding_dataset(video_path=None, num_frames=10000, latent_dim=16, 
                           vae_checkpoint_path=None, **kwargs):
    """
    Convenience function to create embedding dataset
    
    Args:
        video_path: Path to video file (if None, searches ./videos/)
        num_frames: Number of frames to use
        latent_dim: VAE latent dimension  
        vae_checkpoint_path: Path to VAE checkpoint
        **kwargs: Additional arguments for EmbeddingDataset
    
    Returns:
        EmbeddingDataset instance
    """
    return EmbeddingDataset(
        video_path=video_path,
        num_frames=num_frames,
        latent_dim=latent_dim,
        vae_checkpoint_path=vae_checkpoint_path,
        **kwargs
    )


def test_embedding_dataset():
    """Test the embedding dataset"""
    print("Testing Embedding Dataset...")
    
    try:
        # Create dataset
        dataset = create_embedding_dataset(
            num_frames=1000,  # Small test
            latent_dim=16,
            batch_size=16
        )
        
        # Print info
        info = dataset.get_embedding_info()
        print("\nDataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"\nDataLoader test:")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Num batches: {len(dataloader)}")
        
        # Get first batch
        batch = next(iter(dataloader))
        print(f"  Batch shape: {batch.shape}")
        print(f"  Batch dtype: {batch.dtype}")
        print(f"  Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
        
        print("\n✅ Embedding dataset test passed!")
        return dataset
        
    except Exception as e:
        print(f"\n❌ Embedding dataset test failed: {e}")
        return None


if __name__ == "__main__":
    test_embedding_dataset()
