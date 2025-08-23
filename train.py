import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dit import create_dit_flow_model
from embedding_dataset import create_embedding_dataset
from utils import get_device
from einops import rearrange

def load_embedding_data(batch_size=64, num_frames=10000, latent_dim=16, seq_len=32):
    """Load embedding dataset - now returns sequences of length 32"""
    dataset = create_embedding_dataset(
        num_frames=num_frames,
        latent_dim=latent_dim,
        batch_size=32  # Batch size for VAE embedding process
    )
    
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    print(f"Loaded embedding dataset: {len(dataset)} sequence samples")
    info = dataset.get_embedding_info()
    print(f"Total embeddings: {info['num_embeddings']}")
    print(f"Sequences available: {len(dataset)} (length {seq_len})")
    print(f"Each sequence shape: [seq_len, latent_dim, height, width] = [32, 16, 32, 18]")
    print(f"Embedding range: [{info['range'][0]:.3f}, {info['range'][1]:.3f}]")
    
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

def train_model(epochs=100, batch_size=32, lr=1e-4, num_frames=10000, latent_dim=16, 
                seq_len=32, d_model=512, n_layers=6, n_heads=8):
    """Train the DiT flow matching model on embedding sequences"""
    device = get_device()
    
    # Load embedding data
    train_loader = load_embedding_data(batch_size, num_frames, latent_dim, seq_len)
    
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
    
    print(f"Created DiT model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
    
    # Load checkpoint if exists
    checkpoint_path = f'dit_flow_model_dim{latent_dim}_seq{seq_len}.pth'
    try:
        checkpoint = t.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    except FileNotFoundError:
        start_epoch = 0
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    
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
        
        # Save checkpoint with more info
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'model_config': {
                'latent_dim': latent_dim,
                'seq_len': seq_len,
                'd_model': d_model,
                'n_layers': n_layers,
                'n_heads': n_heads
            }
        }
        t.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved as '{checkpoint_path}'")


if __name__ == "__main__":
    train_model(
        epochs=50,
        batch_size=256 + 128,      # Smaller batch size for sequences
        lr=1e-4,           # Lower learning rate for stability
        num_frames=10000,
        latent_dim=16,
        seq_len=32,
        d_model=256,       # Smaller model for faster training
        n_layers=4,        # Fewer layers
        n_heads=8
    )
