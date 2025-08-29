import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from dit import create_dit_flow_model
from encoding_dataset import create_encoding_dataset
from utils import get_device
from einops import rearrange
from safetensors.torch import save_file, load_file

def load_encoded_data(batch_size=64, encoded_h5_path=None, latent_dim=16, seq_len=32, 
                     num_sequences=None, sample_latents=True, max_frames=None):
    """Load encoded dataset from H5 file"""
    if encoded_h5_path is None:
        # Try to find encoded dataset automatically
        import glob
        encoded_files = glob.glob("encodings/*_encoded_dim*.h5")
        if encoded_files:
            encoded_h5_path = encoded_files[0]
            print(f"Auto-detected encoded dataset: {encoded_h5_path}")
        else:
            raise FileNotFoundError("No encoded dataset found. Please specify --encoded_h5_path or run encode_dataset.py first")
    
    dataset = create_encoding_dataset(
        h5_path=encoded_h5_path,
        sequence_length=seq_len,
        num_sequences=num_sequences,
        sample_latents=sample_latents,
        max_frames=max_frames
    )
    
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    print(f"Loaded encoded dataset: {len(dataset)} sequence samples")
    info = dataset.get_info()
    print(f"Dataset info:")
    print(f"  Total sequences: {info['total_sequences']}")
    print(f"  Sequence length: {info['sequence_length']}")
    print(f"  Latent dim: {info['latent_dim']}")
    print(f"  Model size: {info['model_size']}")
    print(f"  Original frame size: {info['original_frame_size']}")
    print(f"  VAE checkpoint: {info['vae_checkpoint_path']}")
    print(f"  Sample latents: {info['sample_latents']}")
    
    # Get latent statistics
    stats = dataset.get_latent_statistics()
    print(f"Latent statistics:")
    print(f"  Mu range: [{stats['mu_mean']:.3f} ± {stats['mu_std']:.3f}]")
    print(f"  Variance range: [{stats['variance_mean']:.3f} ± {stats['variance_std']:.3f}]")
    
    return train_loader


# flow matching next frame prediction loss
def compute_loss(model, batch):
    input = batch[:, :-1]
    target = batch[:, 1:]
    prior = t.randn_like(input)
    time = t.rand(input.shape[:2], device=input.device)
    time = rearrange(time, 'b t -> b t 1 1 1')
    input = input * time + prior * (1 - time) 
    v_pred = model(input)
    v_target = target - prior
    return nn.functional.mse_loss(v_pred, v_target)

def train_model(epochs=100, batch_size=32, lr=1e-4, encoded_h5_path=None, latent_dim=16, 
                seq_len=32, d_model=512, n_layers=6, n_heads=16, num_sequences=None, 
                sample_latents=True, max_frames=None):
    """Train the DiT flow matching model on encoded sequences"""
    device = get_device()
    
    # Load encoded data
    train_loader = load_encoded_data(batch_size, encoded_h5_path, latent_dim, seq_len, 
                                   num_sequences, sample_latents, max_frames)
    
    # Create DiT model
    model = create_dit_flow_model(
        input_spatial_shape=(32, 18),  # Height, width of embeddings
        latent_dim=latent_dim,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.1
    ).to(device)
    
    # Wrap model in DataParallel for multi-GPU training
    num_gpus = t.cuda.device_count()
    if num_gpus > 1:
        model = DataParallel(model)
        print(f"Using DataParallel with {num_gpus} GPUs")
        
        # Automatically adjust batch size for multi-GPU training
        original_batch_size = batch_size
        batch_size = batch_size * num_gpus
        print(f"Adjusted batch size: {original_batch_size} → {batch_size} (×{num_gpus} GPUs)")
        
        # Recreate DataLoader with adjusted batch size
        train_loader = DataLoader(
            train_loader.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
    else:
        print(f"Single GPU training on {device}")
    
    print(f"Created DiT model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
    
    # Load model if exists
    checkpoint_path = f'dit_flow_model_dim{latent_dim}_seq{seq_len}.safetensors'
    try:
        model_state_dict = load_file(checkpoint_path)  # Load to CPU first
        model.load_state_dict(model_state_dict)
        print(f"Loaded model from {checkpoint_path}")
        start_epoch = 0  # Always start from epoch 0 since we don't save training state
    except FileNotFoundError:
        start_epoch = 0
        print(f"No model found at {checkpoint_path}, starting from scratch")
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Expected batch shape: [batch_size, {seq_len}, {latent_dim}, 32, 18]")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch + epochs}')
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            
            # Verify data shape
            if batch_idx == 0:
                print(f"Batch shape: {data.shape}")
                print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = compute_loss(model, data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{start_epoch + epochs}, Average Loss: {avg_loss:.4f}')
        
        # Step scheduler
        scheduler.step(avg_loss)
        
        # Save only the model state dict
        save_file(model.state_dict(), checkpoint_path)
        print(f"Model saved as '{checkpoint_path}'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DiT flow matching model on encoded dataset")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=384, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Data parameters
    parser.add_argument("--encoded_h5", type=str, default=None, help="Path to encoded H5 file")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--num_sequences", type=int, default=None, help="Number of sequences to use")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to load")
    parser.add_argument("--sample_latents", action="store_true", help="Sample from distribution")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    
    args = parser.parse_args()
    
    print("Training DiT flow matching model on encoded dataset...")
    print(f"Parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoded_h5_path=args.encoded_h5,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_sequences=args.num_sequences,
        max_frames=args.max_frames,
        sample_latents=args.sample_latents
    )
