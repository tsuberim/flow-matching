import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import wandb
from safetensors.torch import save_file, load_file
import os

from vae import create_video_vae, vae_loss
from video_dataset import create_video_dataset
from utils import get_device





def log_reconstruction_to_wandb(original, reconstruction, epoch):
    """
    Log original vs reconstructed frames to wandb
    
    Args:
        original: [B, 3, 180, 320] original frames
        reconstruction: [B, 3, 180, 320] reconstructed frames
        epoch: current epoch number
    """
    # Take first 4 samples for wandb logging
    num_samples = min(4, original.shape[0])
    
    images = []
    for i in range(num_samples):
        # Original
        orig_img = original[i].permute(1, 2, 0).cpu()
        orig_img = torch.clamp((orig_img + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
        
        # Reconstruction
        recon_img = reconstruction[i].permute(1, 2, 0).cpu()
        recon_img = torch.clamp((recon_img + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
        
        # Create side-by-side comparison
        comparison = torch.cat([orig_img, recon_img], dim=1)  # Concatenate horizontally
        images.append(wandb.Image(comparison.numpy(), caption=f"Sample {i+1}: Original (left) vs Reconstructed (right)"))
    
    wandb.log({
        "reconstructions": images,
        "epoch": epoch
    })


def train_vae(epochs=100, batch_size=32, lr=1e-3, beta=1.0, latent_dim=8, 
              num_frames=None, visualize_every=10, model_size=1, project_name="video-vae"):
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
        model_size: model size multiplier for channels
        project_name: wandb project name
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Load checkpoint metadata first to get wandb run ID
    checkpoint_path = f'vae_checkpoint_dim{latent_dim}_size{model_size}.safetensors'
    metadata_path = f'vae_checkpoint_dim{latent_dim}_size{model_size}_metadata.pth'
    wandb_run_id = None
    
    try:
        metadata = torch.load(metadata_path, map_location=device)
        wandb_run_id = metadata.get('wandb_run_id')
        print(f"Found existing wandb run ID: {wandb_run_id}")
    except (FileNotFoundError, KeyError):
        print("No existing checkpoint found, will create new wandb run")
    
    # Initialize wandb (resume if we have a run ID)
    wandb.init(
        project=project_name,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "beta": beta,
            "latent_dim": latent_dim,
            "num_frames": num_frames,
            "model_size": model_size,
            "device": str(device)
        }
    )
    
    # Create dataset and dataloader
    print("Loading video dataset...")
    dataset = create_video_dataset(num_frames=num_frames)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-process only
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Using single-process DataLoader (num_workers=0)")
    print(f"Dataset size: {len(dataset)} frames")
    print(f"Number of batches: {len(dataloader)}")
    
    # Create VAE model and move to device
    vae = create_video_vae(latent_dim=latent_dim, model_size=model_size)
    vae = vae.to(device)
    
    # Set up optimizer and scheduler
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Load checkpoint if exists (metadata already loaded above for wandb)
    start_epoch = 0
    best_loss = float('inf')
    
    if wandb_run_id is not None:
        try:
            # Load model weights from safetensors
            model_state = load_file(checkpoint_path)
            
            # Remove 'module.' prefix if loading DataParallel weights to single GPU model
            if any(key.startswith('module.') for key in model_state.keys()):
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            
            vae.load_state_dict(model_state)
            
            # Use already loaded metadata
            optimizer.load_state_dict(metadata['optimizer_state_dict'])
            if 'scheduler_state_dict' in metadata:
                scheduler.load_state_dict(metadata['scheduler_state_dict'])
            start_epoch = metadata['epoch'] + 1
            best_loss = metadata.get('best_loss', float('inf'))
            print(f"Loaded checkpoint from epoch {metadata['epoch']}")
            print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
        except (FileNotFoundError, KeyError) as e:
            print(f"Checkpoint metadata found but model loading failed: {e}")
    else:
        print(f"No checkpoint found, starting from scratch")
    
    # Training loop
    print(f"Starting VAE training for {epochs} epochs (from epoch {start_epoch})...")
    
    for epoch in range(start_epoch, epochs):
        vae.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_sim_loss = 0
        total_diff_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, frames in enumerate(pbar):
            frames = frames.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss, recon_loss, kl_loss, sim_loss, diff_loss = vae_loss(vae, frames, beta=beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_sim_loss += sim_loss.item()
            total_diff_loss += diff_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
                'Sim': f'{sim_loss.item():.4f}',
                'Diff': f'{diff_loss.item():.4f}'
            })
            
            # Log metrics to wandb every batch
            wandb.log({
                "batch_loss": loss.item(),
                "batch_recon_loss": recon_loss.item(),
                "batch_kl_loss": kl_loss.item(),
                "batch_sim_loss": sim_loss.item(),
                "batch_diff_loss": diff_loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "step": epoch * len(dataloader) + batch_idx
            })
            
            # Checkpoint and log images every 100 batches
            if batch_idx % 100 == 0:
                # Save checkpoint
                model_state = vae.module.state_dict() if isinstance(vae, torch.nn.DataParallel) else vae.state_dict()
                save_file(model_state, checkpoint_path)
                
                # Save training metadata
                metadata = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'best_loss': best_loss,
                    'latent_dim': latent_dim,
                    'beta': beta,
                    'model_size': model_size,
                    'wandb_run_id': wandb.run.id,
                }
                torch.save(metadata, metadata_path)
                
                # Log reconstruction images
                vae.eval()
                with torch.no_grad():
                    # Use current batch for visualization
                    b, t, c, h, w = frames.shape
                    sample_flat = frames.view(b * t, c, h, w)
                    sample_recon, _, _ = vae(sample_flat)
                    sample_recon = sample_recon.view(b, t, c, h, w)
                    
                    log_reconstruction_to_wandb(
                        frames[:, 0], sample_recon[:, 0], f"E{epoch+1}_B{batch_idx}"
                    )
                vae.train()
                
                print(f"Checkpoint saved at epoch {epoch+1}, batch {batch_idx} (loss: {loss.item():.4f})")
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_sim_loss = total_sim_loss / len(dataloader)
        avg_diff_loss = total_diff_loss / len(dataloader)
            
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Total Loss: {avg_loss:.4f}')
        print(f'  Recon Loss: {avg_recon_loss:.4f}')
        print(f'  KL Loss: {avg_kl_loss:.4f}')
        print(f'  Sim Loss: {avg_sim_loss:.4f}')
        print(f'  Diff Loss: {avg_diff_loss:.4f}')
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_recon_loss": avg_recon_loss,
            "avg_kl_loss": avg_kl_loss,
            "avg_sim_loss": avg_sim_loss,
            "avg_diff_loss": avg_diff_loss
        })
        
        # Step scheduler
        scheduler.step(avg_loss)
        
        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        print(f"Epoch {epoch+1} completed (avg loss: {avg_loss:.4f}, best: {best_loss:.4f})")
    
    # Final model save using safetensors
    final_model_path = f'vae_final_dim{latent_dim}_size{model_size}.safetensors'
    final_model_state = vae.state_dict()
    save_file(final_model_state, final_model_path)
    print(f"Final model saved as '{final_model_path}'")
    
    # Finish wandb run
    wandb.finish()
    
    return vae


def test_vae_sampling(latent_dim=8, num_samples=16, model_size=1):
    """
    Test VAE sampling from latent space
    
    Args:
        latent_dim: latent space dimensionality
        num_samples: number of samples to generate
        model_size: model size multiplier
    """
    device = get_device()
    
    # Load trained VAE
    vae = create_video_vae(latent_dim=latent_dim, model_size=model_size)
    vae = vae.to(device)
    
    try:
        model_state = load_file(f'vae_final_dim{latent_dim}_size{model_size}.safetensors')
        
        # Remove 'module.' prefix if loading DataParallel weights to single GPU model
        if any(key.startswith('module.') for key in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        
        vae.load_state_dict(model_state)
        print(f"Loaded trained VAE model")
    except FileNotFoundError:
        print(f"No trained model found, using random weights")
    
    vae.eval()
    
    # Generate samples
    with torch.no_grad():
        samples = vae.sample(num_samples, device)
    
    # Log samples to wandb instead of matplotlib
    sample_images = []
    for i in range(min(num_samples, 16)):  # Limit to 16 for wandb
        sample_img = samples[i].permute(1, 2, 0).cpu()
        sample_img = torch.clamp((sample_img + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
        sample_images.append(wandb.Image(sample_img.numpy(), caption=f"Generated Sample {i+1}"))
    
    # Initialize wandb for sampling if not already initialized
    if not wandb.run:
        wandb.init(project="video-vae-sampling", config={"latent_dim": latent_dim, "model_size": model_size})
    
    wandb.log({"generated_samples": sample_images})
    print(f"Generated {len(sample_images)} samples and logged to wandb")


if __name__ == "__main__":
    # Train VAE
    trained_vae = train_vae(
        epochs=50,
        batch_size=6,
        lr=1e-3,
        beta=1e-5,  # Start with beta~=0 (no KL regularization)
        latent_dim=16,
        num_frames=10000,  # Use subset for faster training
        # visualize_every=1,  # Show reconstructions every epoch
        model_size=4,  # Model size multiplier
        project_name="video-vae"
    )
    
    # Test sampling
    print("\nTesting VAE sampling...")
    test_vae_sampling(latent_dim=16, num_samples=16, model_size=1)
