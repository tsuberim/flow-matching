import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import wandb
from safetensors.torch import save_file, load_file
import os

from vae import create_video_vae, vae_loss
from video_dataset import create_video_dataset
from utils import get_device


def setup_ddp(rank, world_size, backend='nccl'):
    """
    Setup Distributed Data Parallel
    
    Args:
        rank: process rank
        world_size: total number of processes
        backend: communication backend
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_model_ddp(model, rank):
    """
    Setup model for DDP
    
    Args:
        model: PyTorch model
        rank: process rank
    
    Returns:
        model: DDP wrapped model
        device: device for this rank
    """
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    return model, device


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


def train_vae_ddp(rank, world_size, epochs=100, batch_size=32, lr=1e-3, beta=1.0, latent_dim=8, 
                  num_frames=None, visualize_every=10, model_size=1, project_name="video-vae"):
    """
    Train the VAE using Distributed Data Parallel
    
    Args:
        rank: process rank
        world_size: total number of processes
        epochs: number of training epochs
        batch_size: batch size per GPU
        lr: learning rate
        beta: beta parameter for beta-VAE (KL weight)
        latent_dim: latent space dimensionality
        num_frames: number of frames to use from video
        visualize_every: visualize reconstruction every N epochs
        model_size: model size multiplier for channels
        project_name: wandb project name
    """
    try:
        # Setup DDP
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        
        # Only print from rank 0
        if rank == 0:
            print(f"Training with {world_size} GPUs using DDP")
            print(f"Batch size per GPU: {batch_size}")
            print(f"Effective batch size: {batch_size * world_size}")
        
        effective_batch_size = batch_size
    
        # Initialize wandb only on rank 0
        if rank == 0:
            wandb.init(
                project=project_name,
                config={
                    "epochs": epochs,
                    "batch_size_per_gpu": batch_size,
                    "world_size": world_size,
                    "effective_batch_size": batch_size * world_size,
                    "learning_rate": lr,
                    "beta": beta,
                    "latent_dim": latent_dim,
                    "num_frames": num_frames,
                    "model_size": model_size,
                    "device": str(device)
                }
            )
    
        # Create dataset and dataloader with DistributedSampler
        if rank == 0:
            print("Loading video dataset...")
        dataset = create_video_dataset(num_frames=num_frames)
        
        # Use DistributedSampler for proper data distribution
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        num_workers = min(2, os.cpu_count() // world_size)  # Divide workers among processes
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=effective_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        if rank == 0:
            print(f"Using {num_workers} DataLoader workers per process")
            print(f"Dataset size: {len(dataset)} frames")
            print(f"Number of batches per process: {len(dataloader)}")
    
        # Create VAE model and setup for DDP
        vae = create_video_vae(latent_dim=latent_dim, model_size=model_size)
        vae, device = setup_model_ddp(vae, rank)
        
        # Scale learning rate by world size
        scaled_lr = lr * world_size
        if rank == 0:
            print(f"Scaling learning rate: {lr} -> {scaled_lr} (x{world_size})")
        
        optimizer = optim.Adam(vae.parameters(), lr=scaled_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Load checkpoint if exists
        checkpoint_path = f'vae_checkpoint_dim{latent_dim}_size{model_size}.safetensors'
        metadata_path = f'vae_checkpoint_dim{latent_dim}_size{model_size}_metadata.pth'
        start_epoch = 0
        best_loss = float('inf')
        
        try:
            # Load model weights from safetensors
            model_state = load_file(checkpoint_path)
            
            # Remove 'module.' prefix if loading DataParallel weights to single GPU model
            if any(key.startswith('module.') for key in model_state.keys()):
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            
            vae.load_state_dict(model_state)
            
            # Load training metadata from regular torch file
            metadata = torch.load(metadata_path, map_location=device)
            optimizer.load_state_dict(metadata['optimizer_state_dict'])
            if 'scheduler_state_dict' in metadata:
                scheduler.load_state_dict(metadata['scheduler_state_dict'])
            start_epoch = metadata['epoch'] + 1
            best_loss = metadata.get('best_loss', float('inf'))
            if rank == 0:
                print(f"Loaded checkpoint from epoch {metadata['epoch']}")
                print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
        except (FileNotFoundError, KeyError) as e:
            if rank == 0:
                print(f"No checkpoint found, starting from scratch")
        
        # Training loop
        if rank == 0:
            print(f"Starting VAE training for {epochs} epochs (from epoch {start_epoch})...")
        
        for epoch in range(start_epoch, epochs):
            # Set epoch for distributed sampler
            sampler.set_epoch(epoch)
            
            vae.train()
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_sim_loss = 0
            total_diff_loss = 0
            
            # Only show progress bar on rank 0
            if rank == 0:
                pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
                iterator = pbar
            else:
                iterator = dataloader
            
            for batch_idx, frames in enumerate(iterator):
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
                
                # Update progress bar (only on rank 0)
                if rank == 0:
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Recon': f'{recon_loss.item():.4f}',
                        'KL': f'{kl_loss.item():.4f}',
                        'Sim': f'{sim_loss.item():.4f}',
                        'Diff': f'{diff_loss.item():.4f}'
                    })
            
            # Log metrics to wandb every few batches
            if batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_recon_loss": recon_loss.item(),
                    "batch_kl_loss": kl_loss.item(),
                    "batch_sim_loss": sim_loss.item(),
                    "batch_diff_loss": diff_loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "step": epoch * len(dataloader) + batch_idx
                })
        
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
        
        # Visualize reconstruction after every epoch
        vae.eval()
        with torch.no_grad():
            # Get a batch for visualization
            sample_batch = next(iter(dataloader)).to(device)
            # Reshape for VAE: [B, T, C, H, W] -> [B*T, C, H, W]
            b, t, c, h, w = sample_batch.shape
            sample_flat = sample_batch.view(b * t, c, h, w)
            sample_recon, _, _ = vae(sample_flat)
            # Reshape back: [B*T, C, H, W] -> [B, T, C, H, W]
            sample_recon = sample_recon.view(b, t, c, h, w)
            
            # Log reconstruction images to wandb
            log_reconstruction_to_wandb(
                sample_batch[:, 0], sample_recon[:, 0], epoch + 1
            )
        vae.train()
        
        # Save checkpoint every epoch using safetensors
        # Save model weights (handle DataParallel)
        model_state = vae.module.state_dict() if isinstance(vae, torch.nn.DataParallel) else vae.state_dict()
        save_file(model_state, checkpoint_path)
        
        # Save training metadata
        metadata = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'latent_dim': latent_dim,
            'beta': beta,
            'model_size': model_size,
        }
        torch.save(metadata, metadata_path)
        print(f"Checkpoint saved at epoch {epoch+1} (loss: {avg_loss:.4f}, best: {best_loss:.4f})")
    
        # Final model save using safetensors (only on rank 0)
        if rank == 0:
            final_model_path = f'vae_final_dim{latent_dim}_size{model_size}.safetensors'
            final_model_state = vae.module.state_dict() if hasattr(vae, 'module') else vae.state_dict()
            save_file(final_model_state, final_model_path)
            print(f"Final model saved as '{final_model_path}'")
            
            # Finish wandb run
            wandb.finish()
    
        return vae
    
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise e
    finally:
        cleanup_ddp()


def train_vae_wrapper(epochs=50, batch_size=2, lr=1e-3, beta=1e-5, latent_dim=16, 
                     num_frames=None, visualize_every=10, model_size=2, project_name="video-vae"):
    """
    Wrapper function to launch DDP training
    """
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Warning: Only 1 GPU available, falling back to single GPU training")
        # For single GPU, we can still use the DDP function with world_size=1
        world_size = 1
    
    print(f"Launching DDP training with {world_size} processes")
    
    mp.spawn(
        train_vae_ddp,
        args=(world_size, epochs, batch_size, lr, beta, latent_dim, num_frames, visualize_every, model_size, project_name),
        nprocs=world_size,
        join=True
    )


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
    vae, _ = setup_gpu(vae, device)
    
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
    # Train VAE using DDP
    train_vae_wrapper(
        epochs=50,
        batch_size=2,  # Batch size per GPU
        lr=1e-3,
        beta=1e-5,  # Start with beta~=0 (no KL regularization)
        latent_dim=16,
        # num_frames=1000,  # Use subset for faster training
        # visualize_every=1,  # Show reconstructions every epoch
        model_size=2,  # Model size multiplier
        project_name="video-vae"
    )
    
    # Test sampling
    print("\nTesting VAE sampling...")
    test_vae_sampling(latent_dim=16, num_samples=16, model_size=1)
