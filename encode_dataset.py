import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import os
from safetensors.torch import load_file

from vae import create_video_vae
from video_dataset2 import create_dataset
from utils import get_device


def encode_dataset(vae_checkpoint_path, h5_path, output_dir="encodings", batch_size=32, latent_dim=16, 
                  model_size=2, num_frames=None, sequence_length=1):
    """
    Encode video dataset using trained VAE
    
    Args:
        vae_checkpoint_path: path to trained VAE checkpoint
        h5_path: path to video dataset H5 file
        output_dir: directory to save encoded dataset
        batch_size: batch size for encoding
        latent_dim: VAE latent dimension
        model_size: VAE model size
        num_frames: number of frames to encode (None for all)
        sequence_length: sequence length for dataset loading
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Extract video hash from input filename and construct output path
    h5_basename = os.path.basename(h5_path)
    video_hash = os.path.splitext(h5_basename)[0]  # Remove .h5 extension
    output_filename = f"{video_hash}_encoded_dim{latent_dim}_size{model_size}.h5"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Video hash: {video_hash}")
    print(f"Output will be saved to: {output_path}")
    
    # Load VAE model
    print(f"Loading VAE from: {vae_checkpoint_path}")
    vae = create_video_vae(latent_dim=latent_dim, model_size=model_size)
    vae = vae.to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        vae = torch.nn.DataParallel(vae)
    
    try:
        # Load checkpoint
        model_state = load_file(vae_checkpoint_path)
        
        # Handle DataParallel loading
        is_dataparallel_checkpoint = any(key.startswith('module.') for key in model_state.keys())
        is_current_dataparallel = isinstance(vae, torch.nn.DataParallel)
        
        if is_dataparallel_checkpoint and not is_current_dataparallel:
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        elif not is_dataparallel_checkpoint and is_current_dataparallel:
            model_state = {f'module.{k}': v for k, v in model_state.items()}
        
        vae.load_state_dict(model_state)
        print("VAE checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Set VAE to evaluation mode
    vae.eval()
    
    # Load dataset
    print(f"Loading dataset from: {h5_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Dataset not found: {h5_path}")
    
    dataset = create_dataset(h5_path=h5_path, sequence_length=sequence_length, num_frames=num_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get dataset info
    info = dataset.get_info()
    print(f"Dataset info:")
    print(f"  Total frames: {info['total_frames']:,}")
    print(f"  Frame size: {info['width']}x{info['height']}")
    print(f"  Sequence length: {info['sequence_length']}")
    print(f"  Dataset sequences: {len(dataset):,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(dataloader):,}")
    
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode dataset
    print(f"Encoding dataset to: {output_path}")
    
    all_mu = []
    all_logvar = []
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(tqdm(dataloader, desc="Encoding frames")):
            frames = frames.to(device)
            
            # Flatten batch and time dimensions for VAE
            b, t, c, h, w = frames.shape
            frames_flat = frames.view(b * t, c, h, w)
            
            # Encode frames to get distribution parameters
            # For DataParallel, we need to extract mu, logvar from forward pass
            if isinstance(vae, torch.nn.DataParallel):
                # Use forward pass which is DataParallel compatible
                _, mu, logvar = vae(frames_flat)
            else:
                # Single GPU: use encode method directly
                mu, logvar = vae.encode(frames_flat)
            
            # Reshape back to batch format
            mu = mu.view(b, t, *mu.shape[1:])
            logvar = logvar.view(b, t, *logvar.shape[1:])
            
            # Move to CPU and store
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches "
                      f"({100 * (batch_idx + 1) / len(dataloader):.1f}%)")
    
    # Concatenate all results
    print("Concatenating results...")
    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)
    
    print(f"Final mu shape: {all_mu.shape}")
    print(f"Final logvar shape: {all_logvar.shape}")
    
    # Save to H5 file
    print("Saving encoded dataset...")
    with h5py.File(output_path, 'w') as f:
        # Save distribution parameters only
        f.create_dataset('mu', data=all_mu, compression='gzip', compression_opts=6)
        f.create_dataset('logvar', data=all_logvar, compression='gzip', compression_opts=6)
        
        # Save metadata
        f.attrs['original_h5_path'] = h5_path
        f.attrs['vae_checkpoint_path'] = vae_checkpoint_path
        f.attrs['latent_dim'] = latent_dim
        f.attrs['model_size'] = model_size
        f.attrs['sequence_length'] = sequence_length
        f.attrs['num_sequences'] = len(all_mu)
        f.attrs['mu_shape'] = all_mu.shape
        f.attrs['logvar_shape'] = all_logvar.shape
        f.attrs['original_frame_size'] = f"{info['width']}x{info['height']}"
        
        # Copy original dataset metadata if available
        try:
            with h5py.File(h5_path, 'r') as orig_f:
                if 'metadata' in orig_f.attrs:
                    f.attrs['original_metadata'] = orig_f.attrs['metadata']
        except Exception as e:
            print(f"Warning: Could not copy original metadata: {e}")
    
    print(f"Successfully saved encoded dataset to: {output_path}")
    print(f"File size: {(all_mu.nbytes + all_logvar.nbytes) / (1024**3):.2f} GB")
    
    # Calculate and print statistics
    print("\nDistribution statistics:")
    print(f"  Mean mu: {np.mean(all_mu):.4f}")
    print(f"  Std mu: {np.std(all_mu):.4f}")
    print(f"  Mean logvar: {np.mean(all_logvar):.4f}")
    print(f"  Std logvar: {np.std(all_logvar):.4f}")
    print(f"  Mean variance: {np.mean(np.exp(all_logvar)):.4f}")
    print(f"  Std variance: {np.std(np.exp(all_logvar)):.4f}")


def create_arg_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(description="Encode video dataset using trained VAE")
    
    # Required arguments
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                       help="Path to trained VAE checkpoint (.safetensors)")
    parser.add_argument("--input_h5", type=str, required=True,
                       help="Path to input video dataset (.h5)")
    parser.add_argument("--output_dir", type=str, default="encodings",
                       help="Output directory for encoded dataset (default: encodings)")
    
    # VAE parameters
    parser.add_argument("--latent_dim", type=int, default=16,
                       help="VAE latent dimension")
    parser.add_argument("--model_size", type=int, default=2,
                       help="VAE model size multiplier")
    
    # Dataset parameters
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding")
    parser.add_argument("--num_frames", type=int, default=None,
                       help="Number of frames to encode (None for all)")
    parser.add_argument("--sequence_length", type=int, default=1,
                       help="Sequence length for dataset loading")
    
    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    print("Encoding video dataset with parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    encode_dataset(
        vae_checkpoint_path=args.vae_checkpoint,
        h5_path=args.input_h5,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        model_size=args.model_size,
        num_frames=args.num_frames,
        sequence_length=args.sequence_length
    )
