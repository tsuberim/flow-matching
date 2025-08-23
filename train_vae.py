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


def train_vae(epochs=100, batch_size=32, lr=1e-3, beta=0.0, latent_dim=8, 
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
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs with DataParallel")
        vae = torch.nn.DataParallel(vae)
        # Scale batch size by number of GPUs for better utilization
        effective_batch_size = batch_size * num_gpus
        # Scale learning rate by number of GPUs (linear scaling rule)
        scaled_lr = lr * num_gpus
        print(f"Effective batch size across GPUs: {effective_batch_size}")
        print(f"Scaled learning rate: {lr} -> {scaled_lr} (x{num_gpus})")
        # Update wandb config with scaled values
        wandb.config.update({
            "effective_batch_size": effective_batch_size,
            "scaled_learning_rate": scaled_lr,
            "num_gpus": num_gpus
        }, allow_val_change=True)
    else:
        print(f"Using single device: {device}")
        scaled_lr = lr
        # Update wandb config for single GPU
        wandb.config.update({
            "effective_batch_size": batch_size,
            "scaled_learning_rate": scaled_lr,
            "num_gpus": 1
        }, allow_val_change=True)
    
    optimizer = optim.Adam(vae.parameters(), lr=scaled_lr)
    
    # Use ReduceLROnPlateau but step it every batch with shorter patience
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=200, factor=0.8)
    
    # Load checkpoint if exists (metadata already loaded above for wandb)
    start_epoch = 0
    best_loss = float('inf')
    
    try:
        # Load model weights from safetensors
        model_state = load_file(checkpoint_path)
        
        # Handle DataParallel loading properly
        is_dataparallel_checkpoint = any(key.startswith('module.') for key in model_state.keys())
        is_current_dataparallel = isinstance(vae, torch.nn.DataParallel)
        
        if is_dataparallel_checkpoint and not is_current_dataparallel:
            # Loading DataParallel weights into single model
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        elif not is_dataparallel_checkpoint and is_current_dataparallel:
            # Loading single model weights into DataParallel
            model_state = {f'module.{k}': v for k, v in model_state.items()}
        
        print(f"Loaded checkpoint")
        vae.load_state_dict(model_state)
    except (FileNotFoundError, KeyError) as e:
        print(f"Checkpoint loading failed: {e}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting from scratch instead...")
    
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
            
            # Check for NaN/inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected at epoch {epoch+1}, batch {batch_idx}")
                print(f"  Loss: {loss.item()}")
                print(f"  Recon: {recon_loss.item()}")
                print(f"  KL: {kl_loss.item()}")
                print(f"  Sim: {sim_loss.item()}")
                print(f"  Diff: {diff_loss.item()}")
                print(f"  Frames min/max: {frames.min().item():.4f}/{frames.max().item():.4f}")
                continue  # Skip this batch
            
            # Backward pass
            loss.backward()

            # Calculate and clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

            optimizer.step()
            
            # Step scheduler after each batch with current loss
            scheduler.step(loss.item())
            
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
                "grad_norm": grad_norm.mean().item(),
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
        
        # Scheduler now steps per batch, not per epoch
        
        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        print(f"Epoch {epoch+1} completed (avg loss: {avg_loss:.4f}, best: {best_loss:.4f})")
    
    # Final model save using safetensors
    final_model_path = f'vae_final_dim{latent_dim}_size{model_size}.safetensors'
    # Handle DataParallel for final save
    final_model_state = vae.module.state_dict() if isinstance(vae, torch.nn.DataParallel) else vae.state_dict()
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
    
    # Use DataParallel if multiple GPUs are available (for consistency)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        vae = torch.nn.DataParallel(vae)
    
    try:
        model_state = load_file(f'vae_final_dim{latent_dim}_size{model_size}.safetensors')
        
        # Handle DataParallel loading properly
        is_dataparallel_checkpoint = any(key.startswith('module.') for key in model_state.keys())
        is_current_dataparallel = isinstance(vae, torch.nn.DataParallel)
        
        if is_dataparallel_checkpoint and not is_current_dataparallel:
            # Loading DataParallel weights into single model
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        elif not is_dataparallel_checkpoint and is_current_dataparallel:
            # Loading single model weights into DataParallel
            model_state = {f'module.{k}': v for k, v in model_state.items()}
        
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
        batch_size=20,
        lr=1e-5,
        beta=0.0,  # Start with beta~=0 (no KL regularization)
        latent_dim=16,
        num_frames=100_000,  # Use subset for faster training
        # visualize_every=1,  # Show reconstructions every epoch
        model_size=4,  # Model size multiplier
        project_name="video-vae"
    )
    
    # Test sampling
    print("\nTesting VAE sampling...")
    test_vae_sampling(latent_dim=16, num_samples=16, model_size=1)
