#!/usr/bin/env python3
"""
Throwaway script to convert encoded dataset back to MP4 video
"""

import torch
import h5py
import numpy as np
import cv2
import argparse
from safetensors.torch import load_file
from tqdm import tqdm
import os

# Try to import imageio as fallback
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available. Install with: pip install imageio[ffmpeg]")

from vae import create_video_vae
from utils import get_device


def encoded_to_video(encoded_h5_path, vae_checkpoint_path, output_video_path, 
                    fps=30, max_frames=None, sample_latents=True):
    """
    Convert encoded dataset back to MP4 video
    
    Args:
        encoded_h5_path: Path to encoded H5 file
        vae_checkpoint_path: Path to VAE checkpoint
        output_video_path: Output MP4 file path
        fps: Frames per second for output video
        max_frames: Maximum number of frames to convert (None for all)
        sample_latents: Whether to sample from distribution or use mu
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Load encoded data
    print(f"Loading encoded data from: {encoded_h5_path}")
    with h5py.File(encoded_h5_path, 'r') as f:
        mu_data = torch.from_numpy(f['mu'][...]).float()
        logvar_data = torch.from_numpy(f['logvar'][...]).float()
        
        # Get metadata
        latent_dim = f.attrs.get('latent_dim', 16)
        model_size = f.attrs.get('model_size', 2)
        original_frame_size = f.attrs.get('original_frame_size', '320x180')
        
        print(f"Mu shape: {mu_data.shape}")
        print(f"Logvar shape: {logvar_data.shape}")
        print(f"Latent dim: {latent_dim}")
        print(f"Model size: {model_size}")
        print(f"Original frame size: {original_frame_size}")
    
    # Parse frame size
    width, height = map(int, original_frame_size.split('x'))
    
    # Load VAE
    print(f"Loading VAE from: {vae_checkpoint_path}")
    vae = create_video_vae(latent_dim=latent_dim, model_size=model_size)
    vae = vae.to(device)
    vae.eval()
    
    # Load checkpoint
    try:
        model_state = load_file(vae_checkpoint_path)
        
        # Handle DataParallel loading
        is_dataparallel_checkpoint = any(key.startswith('module.') for key in model_state.keys())
        if is_dataparallel_checkpoint:
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        
        vae.load_state_dict(model_state)
        print("VAE checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading VAE checkpoint: {e}")
        return
    
    # Determine total frames to process
    if len(mu_data.shape) == 5:
        # Shape: [sequences, seq_len, latent_dim, h, w]
        total_sequences, seq_len = mu_data.shape[:2]
        total_frames = total_sequences * seq_len
        print(f"Data shape: {total_sequences} sequences x {seq_len} frames = {total_frames} total frames")
        
        # Flatten to individual frames
        mu_flat = mu_data.view(-1, *mu_data.shape[2:])  # [total_frames, latent_dim, h, w]
        logvar_flat = logvar_data.view(-1, *logvar_data.shape[2:])
    else:
        # Shape: [frames, latent_dim, h, w]
        mu_flat = mu_data
        logvar_flat = logvar_data
        total_frames = mu_flat.shape[0]
        print(f"Data shape: {total_frames} individual frames")
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        mu_flat = mu_flat[:total_frames]
        logvar_flat = logvar_flat[:total_frames]
        print(f"Limiting to {total_frames} frames")
    
    # Move data to device
    mu_flat = mu_flat.to(device)
    logvar_flat = logvar_flat.to(device)
    
    # Set up video writer using the same reliable method as sample.py
    # Use H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (width, height), True)
    
    # Check if video writer was successfully initialized
    if not out.isOpened():
        print(f"  ❌ Failed to open video writer with H264 codec")
        print(f"  🔄 Trying alternative codec...")
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = output_video_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (width, height), True)
        
        if not out.isOpened():
            raise RuntimeError("Could not initialize video writer with any codec")
        else:
            print(f"  ✅ Using XVID codec, output: {output_video_path}")
    else:
        print(f"  ✅ Using H264 codec, output: {output_video_path}")
    
    print(f"Creating video: {output_video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Sample latents: {sample_latents}")
    
    # Process frames in batches
    batch_size = 32
    num_batches = (total_frames + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Decoding frames"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_frames)
            
            # Get batch
            mu_batch = mu_flat[start_idx:end_idx]
            logvar_batch = logvar_flat[start_idx:end_idx]
            
            if sample_latents:
                # Sample from distribution
                std = torch.exp(0.5 * torch.clamp(logvar_batch, min=-20, max=20))
                eps = torch.randn_like(std)
                z_batch = mu_batch + eps * std
            else:
                # Use means directly
                z_batch = mu_batch
            
            # Decode latent codes
            decoded_batch = vae.decode(z_batch)
            
            # Convert to numpy and process each frame
            decoded_np = decoded_batch.cpu().numpy()
            
            for i in range(decoded_np.shape[0]):
                # Get frame: [C, H, W] -> [H, W, C]
                frame = decoded_np[i].transpose(1, 2, 0)
                
                # Convert from [-1, 1] to [0, 255] (same as sample.py)
                frame = np.clip(frame, -1, 1)  # Ensure range
                frame = (frame + 1.0) / 2.0   # [-1,1] -> [0,1]
                frame = (frame * 255).astype(np.uint8)  # [0,1] -> [0,255]
                
                # Ensure frame has correct shape
                if frame.shape != (height, width, 3):
                    print(f"Warning: Frame shape {frame.shape} != expected {(height, width, 3)}")
                    # Resize if needed
                    frame = cv2.resize(frame, (width, height))
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Verify frame dimensions before writing (same as sample.py)
                if frame_bgr.shape[:2] != (height, width):
                    print(f"  ⚠️  Frame {start_idx + i} dimension mismatch: {frame_bgr.shape} vs expected ({height}, {width})")
                    continue
                
                # Write frame to video
                success = out.write(frame_bgr)
                if not success:
                    print(f"  ⚠️  Failed to write frame {start_idx + i}")
                    break
    
    # Release video writer
    out.release()
    
    # Verify the file was created and has reasonable size (same as sample.py)
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path)
        if file_size > 1000:  # At least 1KB
            print(f"  ✅ Video saved successfully: {output_video_path}")
            print(f"  📁 File size: {file_size:,} bytes")
        else:
            print(f"  ⚠️  Video file seems too small: {output_video_path} ({file_size} bytes)")
    else:
        print(f"  ❌ Video file not created: {output_video_path}")
    
    print(f"  🎬 Total frames written: {total_frames}")
    print(f"  ⏱️  Duration: {total_frames / fps:.2f} seconds")


def encoded_to_video_imageio(encoded_h5_path, vae_checkpoint_path, output_video_path, 
                           fps=30, max_frames=None, sample_latents=True):
    """
    Alternative implementation using imageio (often more reliable)
    """
    if not IMAGEIO_AVAILABLE:
        raise ImportError("imageio not available. Install with: pip install imageio[ffmpeg]")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load encoded data (same as opencv version)
    print(f"Loading encoded data from: {encoded_h5_path}")
    with h5py.File(encoded_h5_path, 'r') as f:
        mu_data = torch.from_numpy(f['mu'][...]).float()
        logvar_data = torch.from_numpy(f['logvar'][...]).float()
        
        latent_dim = f.attrs.get('latent_dim', 16)
        model_size = f.attrs.get('model_size', 2)
        original_frame_size = f.attrs.get('original_frame_size', '320x180')
    
    width, height = map(int, original_frame_size.split('x'))
    
    # Load VAE (same as opencv version)
    print(f"Loading VAE from: {vae_checkpoint_path}")
    vae = create_video_vae(latent_dim=latent_dim, model_size=model_size)
    vae = vae.to(device)
    vae.eval()
    
    model_state = load_file(vae_checkpoint_path)
    if any(key.startswith('module.') for key in model_state.keys()):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    vae.load_state_dict(model_state)
    
    # Process data shape
    if len(mu_data.shape) == 5:
        mu_flat = mu_data.view(-1, *mu_data.shape[2:])
        logvar_flat = logvar_data.view(-1, *logvar_data.shape[2:])
        total_frames = mu_flat.shape[0]
    else:
        mu_flat = mu_data
        logvar_flat = logvar_data
        total_frames = mu_flat.shape[0]
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        mu_flat = mu_flat[:total_frames]
        logvar_flat = logvar_flat[:total_frames]
    
    mu_flat = mu_flat.to(device)
    logvar_flat = logvar_flat.to(device)
    
    print(f"Creating video with imageio: {output_video_path}")
    print(f"Total frames: {total_frames}")
    
    # Collect all frames first
    frames = []
    batch_size = 32
    num_batches = (total_frames + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Decoding frames"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_frames)
            
            mu_batch = mu_flat[start_idx:end_idx]
            logvar_batch = logvar_flat[start_idx:end_idx]
            
            if sample_latents:
                std = torch.exp(0.5 * torch.clamp(logvar_batch, min=-20, max=20))
                eps = torch.randn_like(std)
                z_batch = mu_batch + eps * std
            else:
                z_batch = mu_batch
            
            decoded_batch = vae.decode(z_batch)
            decoded_np = decoded_batch.cpu().numpy()
            
            for i in range(decoded_np.shape[0]):
                frame = decoded_np[i].transpose(1, 2, 0)  # CHW -> HWC
                frame = (frame + 1.0) / 2.0  # [-1,1] -> [0,1]
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                
                # No BGR conversion needed for imageio
                frames.append(frame)
    
    # Write video with imageio
    print("Writing video...")
    imageio.mimsave(output_video_path, frames, fps=fps, quality=8)
    
    print(f"Video saved successfully: {output_video_path}")
    print(f"Total frames written: {len(frames)}")
    print(f"Duration: {len(frames) / fps:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Convert encoded dataset to MP4 video")
    
    parser.add_argument("--encoded_h5", type=str, required=True,
                       help="Path to encoded H5 file")
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                       help="Path to VAE checkpoint")
    parser.add_argument("--output_video", type=str, required=True,
                       help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to convert (default: all)")
    parser.add_argument("--use_mu", action="store_true",
                       help="Use mu instead of sampling from distribution")
    parser.add_argument("--use_imageio", action="store_true",
                       help="Use imageio instead of OpenCV (often more reliable)")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    
    print("Converting encoded dataset to video...")
    print(f"Input: {args.encoded_h5}")
    print(f"Output: {args.output_video}")
    print(f"FPS: {args.fps}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")
    print(f"Sampling mode: {'mu only' if args.use_mu else 'sample from distribution'}")
    print(f"Backend: {'imageio' if args.use_imageio else 'opencv'}")
    print()
    
    try:
        if args.use_imageio or not cv2:
            encoded_to_video_imageio(
                encoded_h5_path=args.encoded_h5,
                vae_checkpoint_path=args.vae_checkpoint,
                output_video_path=args.output_video,
                fps=args.fps,
                max_frames=args.max_frames,
                sample_latents=not args.use_mu
            )
        else:
            encoded_to_video(
                encoded_h5_path=args.encoded_h5,
                vae_checkpoint_path=args.vae_checkpoint,
                output_video_path=args.output_video,
                fps=args.fps,
                max_frames=args.max_frames,
                sample_latents=not args.use_mu
            )
    except Exception as e:
        print(f"Error with primary method: {e}")
        if not args.use_imageio and IMAGEIO_AVAILABLE:
            print("Trying imageio fallback...")
            encoded_to_video_imageio(
                encoded_h5_path=args.encoded_h5,
                vae_checkpoint_path=args.vae_checkpoint,
                output_video_path=args.output_video,
                fps=args.fps,
                max_frames=args.max_frames,
                sample_latents=not args.use_mu
            )
        else:
            raise


if __name__ == "__main__":
    main()
