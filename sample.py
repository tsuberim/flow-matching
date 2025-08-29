import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.integrate import solve_ivp
from dit import create_dit_flow_model
from vae import create_video_vae
from video_dataset2 import create_dataset
from utils import get_device
from torch.utils.data import DataLoader


def load_models(latent_dim=16, seq_len=32, dit_model_path=None, vae_checkpoint_path=None,
                d_model=256, n_layers=4, n_heads=8):
    """Load the trained DiT model and VAE"""
    device = get_device()
    
    # Load DiT model
    if dit_model_path is None:
        dit_model_path = f'dit_flow_model_dim{latent_dim}_seq{seq_len}.pth'
    
    dit_model = create_dit_flow_model(
        input_spatial_shape=(32, 18),
        latent_dim=latent_dim,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.1
    ).to(device)
    
    try:
        checkpoint = t.load(dit_model_path, map_location=device)
        dit_model.load_state_dict(checkpoint['model_state_dict'])
        dit_model.eval()
        print(f"Loaded DiT model from {dit_model_path}")
        print(f"Model config: {checkpoint.get('model_config', 'Not available')}")
    except FileNotFoundError:
        raise FileNotFoundError(f"DiT model checkpoint not found: {dit_model_path}")
    
    # Load VAE
    if vae_checkpoint_path is None:
        vae_checkpoint_path = 'vae_checkpoint_dim16_size2.safetensors'
    
    vae = create_video_vae(latent_dim=latent_dim, model_size=2).to(device)
    
    try:
        from safetensors.torch import load_file
        model_state = load_file(vae_checkpoint_path)
        vae.load_state_dict(model_state)
        vae.eval()
        print(f"Loaded VAE from {vae_checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading VAE from {vae_checkpoint_path}: {e}")
    
    return dit_model, vae, device


def sample_initial_frames(vae, device, batch_size=4, h5_path=None):
    """
    Sample random frames from video dataset and encode them with VAE
    
    Args:
        vae: trained VAE model
        device: torch device
        batch_size: number of initial frames to sample
        h5_path: path to preprocessed H5 file
    
    Returns:
        encoded frames: [batch_size, latent_dim, height, width]
    """
    print(f"Sampling {batch_size} random frames from video...")
    
    # Use default H5 path if none provided
    if h5_path is None:
        h5_path = 'videos/pntCyf13iUQ.h5'
    
    # Create video dataset with individual frames
    video_dataset = create_dataset(
        h5_path=h5_path,
        num_frames=1000,  # Use a subset for faster sampling
        sequence_length=1  # Get individual frames
    )
    
    # Sample random frames
    indices = np.random.choice(len(video_dataset), size=batch_size, replace=False)
    frames = []
    
    for idx in indices:
        frame_seq = video_dataset[idx]  # Shape: [1, 3, 180, 320]
        frame = frame_seq[0]  # Get the single frame: [3, 180, 320]
        frames.append(frame)
    
    # Stack into batch
    batch_frames = t.stack(frames).to(device)  # [batch_size, 3, 180, 320]
    
    print(f"Sampled frames shape: {batch_frames.shape}")
    print(f"Frame range: [{batch_frames.min():.3f}, {batch_frames.max():.3f}]")
    
    # Encode with VAE
    with t.no_grad():
        mu, logvar = vae.encode(batch_frames)
        # Use mean for deterministic encoding
        encoded_frames = mu  # [batch_size, latent_dim, height, width]
    
    print(f"Encoded frames shape: {encoded_frames.shape}")
    print(f"Encoded range: [{encoded_frames.min():.3f}, {encoded_frames.max():.3f}]")
    
    return encoded_frames


def vector_field_sequence(t_val, x_flat, dit_model, current_sequence, device, shape):
    """
    Vector field function for ODE solver in sequence context
    
    Args:
        t_val: time value (scalar)
        x_flat: flattened state vector for next frame
        dit_model: trained DiT model
        current_sequence: current sequence context [batch_size, seq_len, latent_dim, height, width]
        device: torch device
        shape: shape of the next frame [batch_size, latent_dim, height, width]
    
    Returns:
        velocity field as flattened numpy array
    """
    # Reshape flat array back to frame shape
    next_frame = x_flat.reshape(shape)
    
    # Convert to torch tensor
    next_frame_tensor = t.from_numpy(next_frame).float().to(device)
    
    # Append to current sequence to create input for DiT
    batch_size, seq_len = current_sequence.shape[:2]
    input_sequence = t.cat([
        current_sequence,
        next_frame_tensor.unsqueeze(1)  # Add sequence dimension
    ], dim=1)  # [batch_size, seq_len+1, latent_dim, height, width]
    
    with t.no_grad():
        # Get DiT prediction for the full sequence
        predicted_sequence = dit_model(input_sequence)
        
        # Extract velocity for the next frame (last position)
        v = predicted_sequence[:, -1]  # [batch_size, latent_dim, height, width]
    
    # Convert back to numpy and flatten
    return v.cpu().numpy().flatten()


def generate_sequence_autoregressive(dit_model, initial_frames, target_length=64, window_size=32, device=None):
    """
    Generate sequences auto-regressively using DiT model with ODE solver and sliding window
    
    Args:
        dit_model: trained DiT model
        initial_frames: initial frame embeddings [batch_size, latent_dim, height, width]
        target_length: total number of frames to generate
        window_size: fixed window size for sequence context (default: 32)
        device: torch device
    
    Returns:
        generated_sequences: [batch_size, target_length, latent_dim, height, width] - all generated frames
        generation_steps: list of intermediate sequences for visualization
    """
    if device is None:
        device = next(dit_model.parameters()).device
    
    batch_size, latent_dim, height, width = initial_frames.shape
    print(f"Generating sequences auto-regressively with sliding window...")
    print(f"Initial frames: {initial_frames.shape}")
    print(f"Target length: {target_length}")
    print(f"Window size: {window_size}")
    
    # Start with initial frames - expand to sequence format
    current_window = initial_frames.unsqueeze(1)  # [batch_size, 1, latent_dim, height, width]
    
    # Store all generated frames (including initial)
    all_generated_frames = [initial_frames.unsqueeze(1)]  # List of [batch_size, 1, latent_dim, height, width]
    generation_steps = [current_window.clone()]
    
    for step in range(1, target_length):
        print(f"Generating frame {step+1}/{target_length} using ODE solver...")
        
        # Start from noise for the next frame
        next_frame_shape = (batch_size, latent_dim, height, width)
        x0 = np.random.randn(*next_frame_shape)
        x0_flat = x0.flatten()
        
        # Time span for integration (0 to 1)
        t_span = (0, 1)
        t_eval = np.array([1.0])  # Only need final result
        
        print(f"  Integrating ODE for frame {step+1}...")
        print(f"  Using context window of size: {current_window.shape[1]}")
        
        # Solve ODE using current sequence window as context
        solution = solve_ivp(
            fun=lambda t, x: vector_field_sequence(
                t, x, dit_model, current_window, device, next_frame_shape
            ),
            t_span=t_span,
            y0=x0_flat,
            method='DOP853',  # Dormand-Prince 8(5,3)
            t_eval=t_eval,
            rtol=1e-5,
            atol=1e-8
        )
        
        if not solution.success:
            raise RuntimeError(f"ODE solver failed at step {step+1}: {solution.message}")
        
        print(f"  ✅ ODE solved successfully")
        # Get final state and reshape
        final_state = solution.y[:, -1]
        next_frame_np = final_state.reshape(next_frame_shape)
        next_frame = t.from_numpy(next_frame_np).float().to(device)
        next_frame = next_frame.unsqueeze(1)  # Add sequence dimension
        
        # Store the generated frame
        all_generated_frames.append(next_frame)
        
        # Update sliding window
        current_window = t.cat([current_window, next_frame], dim=1)
        
        # If window exceeds window_size, remove the first frame (sliding window)
        if current_window.shape[1] >= window_size:
            current_window = current_window[:, 1:]  # Remove first frame, keep last window_size frames
            print(f"  🪟 Sliding window: removed oldest frame, window size now: {current_window.shape[1]}")
        
        # Store for visualization (store current window state)
        generation_steps.append(current_window.clone())
        
        # Print progress
        print(f"  Generated frames so far: {len(all_generated_frames)}")
        print(f"  Current window size: {current_window.shape[1]}")
        print(f"  Next frame range: [{next_frame.min():.3f}, {next_frame.max():.3f}]")
    
    # Concatenate all generated frames into final sequence
    final_sequence = t.cat(all_generated_frames, dim=1)  # [batch_size, target_length, latent_dim, height, width]
    
    print(f"Generated sequence shape: {final_sequence.shape}")
    print(f"Final window shape: {current_window.shape}")
    return final_sequence, generation_steps


def visualize_sequence_generation(sequences, generation_steps=None, save_path='generated_sequences.png'):
    """
    Visualize generated sequences and generation process
    
    Args:
        sequences: [batch_size, seq_len, latent_dim, height, width] - full generated sequence
        generation_steps: list of intermediate sliding windows
        save_path: path to save visualization
    """
    print("Visualizing generated sequences...")
    
    batch_size, seq_len = sequences.shape[:2]
    
    # Show generation process for first sequence (sliding window states)
    if generation_steps is not None:
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        
        # Show 4 different steps in generation (sliding window states)
        step_indices = [0, len(generation_steps)//4, len(generation_steps)//2, -1]
        step_labels = ['Initial', '25%', '50%', 'Final Window']
        
        for step_idx, (step_pos, label) in enumerate(zip(step_indices, step_labels)):
            step_window = generation_steps[step_pos][0]  # First batch item - current window
            window_length = step_window.shape[0]
            
            # Show up to 8 frames from current window
            for frame_idx in range(8):
                ax = axes[step_idx, frame_idx]
                
                if frame_idx < window_length:
                    # Visualize latent space (show first channel)
                    frame = step_window[frame_idx, 0]  # [height, width]
                    ax.imshow(frame.cpu().numpy(), cmap='viridis')
                    ax.set_title(f'Win[{frame_idx+1}]', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=8)
                
                ax.axis('off')
                
                if frame_idx == 0:
                    ax.set_ylabel(f'{label}\n(Win size: {window_length})', 
                                fontsize=10, rotation=90, labelpad=10)
        
        plt.suptitle('Sliding Window Generation Process (Window States)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_process.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Generation process saved to {save_path.replace('.png', '_process.png')}")
    
    # Show final sequences (first and last frames of each batch)
    fig, axes = plt.subplots(batch_size, 8, figsize=(20, 2*batch_size))
    
    for batch_idx in range(batch_size):
        # Show first 4 and last 4 frames to demonstrate full sequence
        frame_indices = list(range(4)) + list(range(max(4, seq_len-4), seq_len))
        frame_indices = frame_indices[:8]  # Ensure we don't exceed 8 frames
        
        for display_idx, frame_idx in enumerate(frame_indices):
            ax = axes[batch_idx, display_idx] if batch_size > 1 else axes[display_idx]
            
            # Show first channel of latent representation
            frame = sequences[batch_idx, frame_idx, 0]  # [height, width]
            ax.imshow(frame.cpu().numpy(), cmap='viridis')
            ax.axis('off')
            
            if batch_idx == 0:
                if display_idx < 4:
                    ax.set_title(f'Frame {frame_idx+1}', fontsize=8)
                else:
                    ax.set_title(f'Frame {frame_idx+1}', fontsize=8, color='red')
            
            if display_idx == 0:
                ax.set_ylabel(f'Seq {batch_idx+1}', fontsize=10, rotation=90, labelpad=10)
    
    plt.suptitle(f'Generated Sequences - {seq_len} total frames (First 4 + Last 4 shown)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Final sequences saved to {save_path}")


def decode_sequences_and_visualize(latent_sequences, vae, save_path='decoded_sequences.png'):
    """
    Decode latent sequences through VAE and visualize as video frames
    
    Args:
        latent_sequences: [batch_size, seq_len, latent_dim, height, width]
        vae: trained VAE model
        save_path: path to save visualization
    
    Returns:
        decoded_sequences: [batch_size, seq_len, 3, height, width]
    """
    device = next(vae.parameters()).device
    batch_size, seq_len = latent_sequences.shape[:2]
    
    print(f"Decoding {batch_size} sequences of length {seq_len}...")
    
    # Decode all frames
    decoded_sequences = []
    
    with t.no_grad():
        for batch_idx in range(batch_size):
            print(f"Decoding sequence {batch_idx+1}/{batch_size}...")
            
            sequence_frames = []
            for frame_idx in range(seq_len):
                # Get single frame: [latent_dim, height, width]
                latent_frame = latent_sequences[batch_idx, frame_idx].unsqueeze(0)  # [1, latent_dim, height, width]
                
                # Decode through VAE
                decoded_frame = vae.decode(latent_frame.to(device))  # [1, 3, 180, 320]
                decoded_frame = decoded_frame.cpu().squeeze(0)  # [3, 180, 320]
                
                sequence_frames.append(decoded_frame)
            
            # Stack frames into sequence
            decoded_sequence = t.stack(sequence_frames)  # [seq_len, 3, 180, 320]
            decoded_sequences.append(decoded_sequence)
    
    # Stack all sequences
    decoded_sequences = t.stack(decoded_sequences)  # [batch_size, seq_len, 3, 180, 320]
    
    # Clamp and normalize for display
    decoded_sequences = t.clamp(decoded_sequences, -1, 1)
    decoded_sequences = (decoded_sequences + 1) / 2  # [-1,1] -> [0,1]
    
    print(f"Decoded sequences shape: {decoded_sequences.shape}")
    
    # Visualize sequences
    fig, axes = plt.subplots(batch_size, min(8, seq_len), figsize=(min(8, seq_len)*3, batch_size*3))
    
    for batch_idx in range(batch_size):
        for frame_idx in range(min(8, seq_len)):
            ax = axes[batch_idx, frame_idx] if batch_size > 1 else axes[frame_idx]
            
            # Get frame and convert to displayable format
            frame = decoded_sequences[batch_idx, frame_idx]  # [3, 180, 320]
            frame_display = frame.permute(1, 2, 0)  # [180, 320, 3]
            
            ax.imshow(frame_display.numpy())
            ax.axis('off')
            
            if batch_idx == 0:
                ax.set_title(f'Frame {frame_idx+1}', fontsize=10)
            
            if frame_idx == 0:
                ax.set_ylabel(f'Sequence {batch_idx+1}', fontsize=12, rotation=90, labelpad=15)
    
    plt.suptitle(f'Decoded Video Sequences - {seq_len} frames each', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Decoded sequences saved to {save_path}")
    
    return decoded_sequences


def save_sequences_as_videos(decoded_sequences, output_dir='./output', fps=12):
    """
    Save decoded sequences as MP4 videos
    
    Args:
        decoded_sequences: [batch_size, seq_len, 3, height, width] in range [0, 1]
        output_dir: directory to save videos
        fps: frames per second for the videos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    batch_size, seq_len, channels, height, width = decoded_sequences.shape
    print(f"Saving {batch_size} sequences as MP4 videos...")
    print(f"Output directory: {output_dir}")
    print(f"Video specs: {seq_len} frames, {height}x{width}, {fps} FPS")
    
    for batch_idx in range(batch_size):
        video_path = os.path.join(output_dir, f'generated_sequence_{batch_idx+1}.mp4')
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height), True)
        
        # Check if video writer was successfully initialized
        if not video_writer.isOpened():
            print(f"  ❌ Failed to open video writer for {video_path}")
            print(f"  🔄 Trying alternative codec...")
            # Try alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path_alt = video_path.replace('.mp4', '.avi')
            video_writer = cv2.VideoWriter(video_path_alt, fourcc, float(fps), (width, height), True)
            video_path = video_path_alt
        
        if not video_writer.isOpened():
            print(f"  ❌ Failed to initialize video writer for sequence {batch_idx+1}")
            continue
        
        print(f"Saving sequence {batch_idx+1}/{batch_size} to {video_path}...")
        
        for frame_idx in range(seq_len):
            # Get frame: [3, height, width] in range [0, 1]
            frame = decoded_sequences[batch_idx, frame_idx]
            
            # Convert to numpy and scale to [0, 255]
            frame_np = frame.permute(1, 2, 0).numpy()  # [height, width, 3]
            
            # Clamp values to ensure they're in [0, 1] range
            frame_np = np.clip(frame_np, 0, 1)
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Verify frame dimensions
            if frame_bgr.shape[:2] != (height, width):
                print(f"  ⚠️  Frame {frame_idx} dimension mismatch: {frame_bgr.shape} vs expected ({height}, {width})")
                continue
            
            # Write frame to video
            success = video_writer.write(frame_bgr)
            if not success:
                print(f"  ⚠️  Failed to write frame {frame_idx}")
        
        video_writer.release()
        
        # Verify the file was created and has reasonable size
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            if file_size > 1000:  # At least 1KB
                print(f"  ✅ Saved: {video_path} ({file_size:,} bytes)")
            else:
                print(f"  ⚠️  Video file seems too small: {video_path} ({file_size} bytes)")
        else:
            print(f"  ❌ Video file not created: {video_path}")
    
    print(f"\n🎬 All {batch_size} videos saved to {output_dir}/")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, 'generation_info.txt')
    with open(summary_path, 'w') as f:
        f.write("Generated Video Sequences Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Number of sequences: {batch_size}\n")
        f.write(f"Frames per sequence: {seq_len}\n")
        f.write(f"Resolution: {height}x{width}\n")
        f.write(f"Frame rate: {fps} FPS\n")
        f.write(f"Duration per video: {seq_len/fps:.1f} seconds\n")
        f.write("\nGenerated files:\n")
        for i in range(batch_size):
            f.write(f"- generated_sequence_{i+1}.mp4\n")
    
    print(f"📋 Generation summary saved to {summary_path}")


def main():
    """Main auto-regressive sequence generation function"""
    # Configuration
    latent_dim = 16
    seq_len = 32
    batch_size = 4
    target_length = 64
    
    print("🚀 Auto-regressive Video Sequence Generation")
    print("=" * 50)
    
    print("Loading trained models...")
    dit_model, vae, device = load_models(
        latent_dim=latent_dim,
        seq_len=seq_len,
        d_model=256,
        n_layers=4,
        n_heads=8
    )
    
    print("\n1. Sampling initial frames from video...")
    initial_frames = sample_initial_frames(
        vae=vae,
        device=device,
        batch_size=batch_size,
        h5_path=None  # Uses default H5 file
    )
    print(f"Initial frames shape: {initial_frames.shape}")
    
    print(f"\n2. Generating sequences auto-regressively...")
    generated_sequences, generation_steps = generate_sequence_autoregressive(
        dit_model=dit_model,
        initial_frames=initial_frames,
        target_length=target_length,
        window_size=seq_len,  # Use the model's sequence length as window size
        device=device
    )
    
    print(f"\n3. Visualizing generation process...")
    visualize_sequence_generation(
        sequences=generated_sequences,
        generation_steps=generation_steps,
        save_path='autoregressive_sequences.png'
    )
    
    print(f"\n4. Decoding sequences through VAE...")
    decoded_sequences = decode_sequences_and_visualize(
        latent_sequences=generated_sequences,
        vae=vae,
        save_path='decoded_video_sequences.png'
    )
    
    print(f"\n5. Saving sequences as MP4 videos...")
    save_sequences_as_videos(
        decoded_sequences=decoded_sequences,
        output_dir='./output',
        fps=12  # Match dataset frame rate
    )
    
    print(f"\n✅ Generation complete!")
    print(f"   Generated {batch_size} sequences of length {target_length}")
    print(f"   Latent sequences shape: {generated_sequences.shape}")
    print(f"   Decoded sequences shape: {decoded_sequences.shape}")
    print(f"   Saved outputs:")
    print(f"   📊 Visualizations:")
    print(f"      - autoregressive_sequences.png")
    print(f"      - autoregressive_sequences_process.png") 
    print(f"      - decoded_video_sequences.png")
    print(f"   🎬 Videos:")
    print(f"      - ./output/generated_sequence_1.mp4")
    print(f"      - ./output/generated_sequence_2.mp4")
    print(f"      - ./output/generated_sequence_3.mp4")
    print(f"      - ./output/generated_sequence_4.mp4")
    print(f"      - ./output/generation_info.txt")


if __name__ == "__main__":
    main()
