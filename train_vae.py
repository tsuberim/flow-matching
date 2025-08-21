import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from vae import create_video_vae, vae_loss
from video_dataset import create_video_dataset
from utils import get_device


def visualize_reconstruction(original, reconstruction, epoch, save_path=None):
    """
    Visualize original vs reconstructed frames
    
    Args:
        original: [B, 3, 180, 320] original frames
        reconstruction: [B, 3, 180, 320] reconstructed frames
        epoch: current epoch number
        save_path: path to save image (optional)
    """
    # Take first 8 samples
    num_samples = min(8, original.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
    
    for i in range(num_samples):
        # Original
        orig_img = original[i].permute(1, 2, 0).cpu()
        orig_img = torch.clamp((orig_img + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
        axes[0, i].imshow(orig_img.numpy())
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontweight='bold')
        
        # Reconstruction
        recon_img = reconstruction[i].permute(1, 2, 0).cpu()
        recon_img = torch.clamp((recon_img + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
        axes[1, i].imshow(recon_img.numpy())
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontweight='bold')
    
    plt.suptitle(f'VAE Reconstruction - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def train_vae(epochs=100, batch_size=32, lr=1e-3, beta=1.0, latent_dim=8, 
              num_frames=10000, visualize_every=10):
    """
    Train the VAE on video frames
    
    Args:
        epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        beta: beta parameter for beta-VAE (KL weight)
        latent_dim: latent space dimensionality
        num_frames: number of frames to use from video
        visualize_every: visualize reconstruction every N epochs
    """
    device = get_device()
    
    # Create dataset and dataloader
    print("Loading video dataset...")
    dataset = create_video_dataset(num_frames=num_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Dataset size: {len(dataset)} frames")
    print(f"Number of batches: {len(dataloader)}")
    
    # Create VAE model
    vae = create_video_vae(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Load checkpoint if exists
    checkpoint_path = f'vae_checkpoint_dim{latent_dim}.pth'
    start_epoch = 0
    best_loss = float('inf')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    except FileNotFoundError:
        print(f"No checkpoint found, starting from scratch")
    
    # Training loop
    print(f"Starting VAE training for {epochs} epochs (from epoch {start_epoch})...")
    
    for epoch in range(start_epoch, epochs):
        vae.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        # Training loop with progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, frames in enumerate(pbar):
            frames = frames.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mu, logvar = vae(frames)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(reconstruction, frames, mu, logvar, beta=beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Total Loss: {avg_loss:.4f}')
        print(f'  Recon Loss: {avg_recon_loss:.4f}')
        print(f'  KL Loss: {avg_kl_loss:.4f}')
        
        # Step scheduler
        scheduler.step(avg_loss)
        
        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Visualize reconstruction after every epoch
        vae.eval()
        with torch.no_grad():
            # Get a batch for visualization
            sample_batch = next(iter(dataloader)).to(device)
            sample_recon, _, _ = vae(sample_batch)
            
            visualize_reconstruction(
                sample_batch, sample_recon, epoch + 1
            )
        vae.train()
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'latent_dim': latent_dim,
            'beta': beta,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} (loss: {avg_loss:.4f}, best: {best_loss:.4f})")
    
    # Final model save
    final_model_path = f'vae_final_dim{latent_dim}.pth'
    torch.save(vae.state_dict(), final_model_path)
    print(f"Final model saved as '{final_model_path}'")
    
    return vae


def test_vae_sampling(latent_dim=8, num_samples=16):
    """
    Test VAE sampling from latent space
    
    Args:
        latent_dim: latent space dimensionality
        num_samples: number of samples to generate
    """
    device = get_device()
    
    # Load trained VAE
    vae = create_video_vae(latent_dim=latent_dim).to(device)
    
    try:
        vae.load_state_dict(torch.load(f'vae_final_dim{latent_dim}.pth', map_location=device))
        print(f"Loaded trained VAE model")
    except FileNotFoundError:
        print(f"No trained model found, using random weights")
    
    vae.eval()
    
    # Generate samples
    with torch.no_grad():
        samples = vae.sample(num_samples, device)
    
    # Visualize samples
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        # Convert from tensor to image
        sample_img = samples[i].permute(1, 2, 0).cpu()
        sample_img = torch.clamp((sample_img + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
        
        axes[i].imshow(sample_img.numpy())
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'VAE Generated Samples (latent_dim={latent_dim})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'vae_samples_dim{latent_dim}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Samples saved as 'vae_samples_dim{latent_dim}.png'")


if __name__ == "__main__":
    # Train VAE
    trained_vae = train_vae(
        epochs=50,
        batch_size=128 + 128//2,  # Adjust based on GPU memory
        lr=1e-3,
        beta=1e-5,  # Start with beta~=0 (no KL regularization)
        latent_dim=16,
        num_frames=10000,  # Use subset for faster training
        visualize_every=1  # Show reconstructions every 3 epochs
    )
    
    # Test sampling
    print("\nTesting VAE sampling...")
    test_vae_sampling(latent_dim=16, num_samples=16)
