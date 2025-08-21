import torch as t
import torch.nn as nn
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os
from model import create_latent_unet
from vae import create_video_vae
from utils import get_device


def load_models(latent_dim=16, flow_model_path=None, vae_checkpoint_path=None):
    """Load the trained flow model and VAE"""
    device = get_device()
    
    # Load flow model
    if flow_model_path is None:
        flow_model_path = f'latent_flow_model_dim{latent_dim}.pth'
    
    flow_model = create_latent_unet(latent_dim=latent_dim).to(device)
    flow_model.load_state_dict(t.load(flow_model_path, map_location=device))
    flow_model.eval()
    print(f"Loaded flow model from {flow_model_path}")
    
    # Load VAE
    if vae_checkpoint_path is None:
        vae_checkpoint_path = f'vae_final_dim{latent_dim}.pth'
        if not os.path.exists(vae_checkpoint_path):
            vae_checkpoint_path = f'vae_checkpoint_dim{latent_dim}.pth'
    
    vae = create_video_vae(latent_dim=latent_dim).to(device)
    
    try:
        if 'final' in vae_checkpoint_path:
            vae.load_state_dict(t.load(vae_checkpoint_path, map_location=device))
        else:
            checkpoint = t.load(vae_checkpoint_path, map_location=device)
            vae.load_state_dict(checkpoint['model_state_dict'])
        vae.eval()
        print(f"Loaded VAE from {vae_checkpoint_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint_path}")
    
    return flow_model, vae, device


def vector_field(t_val, x_flat, model, device, shape):
    """
    Vector field function for ODE solver
    
    Args:
        t_val: time value (scalar)
        x_flat: flattened state vector
        model: trained UNet model
        device: torch device
        shape: original tensor shape
    
    Returns:
        velocity field as flattened numpy array
    """
    # Reshape flat array back to original shape
    x = x_flat.reshape(shape)
    
    # Convert to torch tensor
    x_tensor = t.from_numpy(x).float().to(device)
    
    with t.no_grad():
        v = model(x_tensor)
    
    # Convert back to numpy and flatten
    return v.cpu().numpy().flatten()


def sample_flow_matching(flow_model, device, num_samples=16, latent_dim=16):
    """
    Sample from flow-matching model using ODE solver in latent space
    
    Args:
        flow_model: trained latent UNet model
        device: torch device
        num_samples: number of samples to generate
        latent_dim: latent space dimensions
    
    Returns:
        generated latent samples as tensor [num_samples, latent_dim, 18, 32]
    """
    # Start from noise in latent space
    x0 = np.random.randn(num_samples, latent_dim, 18, 32)
    shape = x0.shape
    x0_flat = x0.flatten()
    
    # Time span for integration (0 to 1)
    t_span = (0, 1)
    t_eval = np.array([0.0, 0.33, 0.67, 1.0])  # 4 points for visualization
    
    print(f"Integrating ODE from t=0 to t=1 at 4 evaluation points...")
    print(f"Latent space shape: {shape}")
    
    # Solve ODE using Dormand-Prince method
    solution = solve_ivp(
        fun=lambda t, x: vector_field(t, x, flow_model, device, shape),
        t_span=t_span,
        y0=x0_flat,
        method='DOP853',  # Dormand-Prince 8(5,3)
        t_eval=t_eval,
        rtol=1e-5,
        atol=1e-8
    )
    
    if not solution.success:
        print(f"ODE solver failed: {solution.message}")
        return None
    
    # Get final state and reshapec
    final_state = solution.y[:, -1]
    samples = final_state.reshape(shape)
    
    # Get intermediate states for visualization (4 time points)
    intermediates = []
    for i in range(4):  # t=0, 0.33, 0.67, 1.0
        intermediate = solution.y[:, i].reshape(shape)
        intermediates.append(t.from_numpy(intermediate).float())
    
    return t.from_numpy(samples).float(), intermediates


def visualize_samples(samples, intermediates=None, save_path='generated_samples.png'):
    """Visualize generated samples and intermediate states"""
    samples = samples.clamp(-1, 1)  # Clamp to valid range
    samples = (samples + 1) / 2  # Scale to [0, 1]
    
    if intermediates is not None:
        # Show 4 items with their intermediate states (4 time steps each)
        num_items = 4
        fig, axes = plt.subplots(num_items, 4, figsize=(12, 12))
        
        for item_idx in range(num_items):
            if item_idx < len(samples):
                for time_idx in range(4):
                    img = intermediates[time_idx][item_idx, 0]
                    img = img.clamp(-1, 1)
                    img = (img + 1) / 2
                    
                    ax = axes[item_idx, time_idx]
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    
                    if item_idx == 0:
                        time_labels = ['t=0 (noise)', 't=0.33', 't=0.67', 't=1 (data)']
                        ax.set_title(time_labels[time_idx], fontsize=10)
            else:
                for time_idx in range(4):
                    axes[item_idx, time_idx].axis('off')
        
        plt.suptitle('Flow Matching Generation Process', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_process.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Process visualization saved to {save_path.replace('.png', '_process.png')}")
    
    # Also show final results grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Final samples saved to {save_path}")


def decode_and_visualize(latent_samples, vae, intermediates=None):
    """Decode latent samples through VAE and visualize"""
    device = next(vae.parameters()).device
    
    # Decode final samples
    with t.no_grad():
        # Handle both tensor and numpy array inputs
        if isinstance(latent_samples, np.ndarray):
            latent_tensor = t.from_numpy(latent_samples).float().to(device)
        else:
            latent_tensor = latent_samples.float().to(device)
        
        decoded_samples = vae.decode(latent_tensor)  # [N, 3, 180, 320]
        decoded_samples = decoded_samples.cpu()
    
    # Clamp and normalize for display
    decoded_samples = t.clamp(decoded_samples, -1, 1)
    decoded_samples = (decoded_samples + 1) / 2  # [-1,1] -> [0,1]
    
    # Show intermediate process if available
    if intermediates is not None:
        num_items = 4
        fig, axes = plt.subplots(num_items, 4, figsize=(16, 12))
        
        for item_idx in range(min(num_items, len(decoded_samples))):
            for time_idx in range(4):
                # Decode intermediate latent
                intermediate_sample = intermediates[time_idx][item_idx:item_idx+1]
                if isinstance(intermediate_sample, np.ndarray):
                    intermediate_latent = t.from_numpy(intermediate_sample).float().to(device)
                else:
                    intermediate_latent = intermediate_sample.float().to(device)
                with t.no_grad():
                    intermediate_decoded = vae.decode(intermediate_latent)
                    intermediate_decoded = t.clamp(intermediate_decoded, -1, 1)
                    intermediate_decoded = (intermediate_decoded + 1) / 2
                
                # Display as RGB image
                img = intermediate_decoded[0].permute(1, 2, 0)  # [C,H,W] -> [H,W,C]
                
                ax = axes[item_idx, time_idx]
                ax.imshow(img.cpu().numpy())
                ax.axis('off')
                
                if item_idx == 0:
                    time_labels = ['t=0 (noise)', 't=0.33', 't=0.67', 't=1 (data)']
                    ax.set_title(time_labels[time_idx], fontsize=10)
        
        plt.suptitle('Flow Matching Generation Process (Decoded)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # Show final results grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(decoded_samples):
            # Convert to displayable format [H, W, C]
            img = decoded_samples[i].permute(1, 2, 0)
            ax.imshow(img.numpy())
            ax.axis('off')
            ax.set_title(f'Sample {i+1}', fontsize=8)
        else:
            ax.axis('off')
    
    plt.suptitle('Generated Video Frames', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return decoded_samples


def main():
    """Main sampling function"""
    latent_dim = 16
    
    print("Loading trained models...")
    flow_model, vae, device = load_models(latent_dim=latent_dim)
    
    print("Generating samples in latent space...")
    result = sample_flow_matching(flow_model, device, num_samples=16, latent_dim=latent_dim)
    
    if result is not None:
        latent_samples, intermediates = result
        print(f"Generated latent samples shape: {latent_samples.shape}")
        
        print("Decoding through VAE and visualizing...")
        decoded_samples = decode_and_visualize(latent_samples, vae, intermediates)
        print(f"Decoded samples shape: {decoded_samples.shape}")
    else:
        print("Sampling failed!")


if __name__ == "__main__":
    main()
