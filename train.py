import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model import create_latent_unet
from embedding_dataset import create_embedding_dataset
from utils import get_device
from einops import rearrange

def load_embedding_data(batch_size=64, num_frames=10000, latent_dim=16):
    """Load embedding dataset"""
    dataset = create_embedding_dataset(
        num_frames=num_frames,
        latent_dim=latent_dim,
        batch_size=32  # Batch size for VAE embedding process
    )
    
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    print(f"Loaded embedding dataset: {len(dataset)} samples")
    info = dataset.get_embedding_info()
    print(f"Embedding shape: {info['embedding_shape']}")
    print(f"Embedding range: [{info['range'][0]:.3f}, {info['range'][1]:.3f}]")
    
    return train_loader


# flow matching loss
def compute_loss(model, batch):
    prior = t.randn_like(batch)
    time = t.rand(batch.shape[0], device=batch.device)
    time = rearrange(time, 'b -> b 1 1 1') # add channel dimension
    input = batch * time + prior * (1 - time) 
    v_pred = model(input)
    v_target = batch - prior
    return nn.functional.mse_loss(v_pred, v_target)


def train_model(epochs=100, batch_size=128, lr=1e-3, num_frames=10000, latent_dim=16):
    """Train the flow matching model on VAE embeddings"""
    device = get_device()
    
    # Load embedding data
    train_loader = load_embedding_data(batch_size, num_frames, latent_dim)
    
    # Create latent space model
    model = create_latent_unet(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.8)
    
    # Load checkpoint if exists
    checkpoint_path = f'latent_flow_model_dim{latent_dim}.pth'
    try:
        model.load_state_dict(t.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = compute_loss(model, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # Step scheduler
        scheduler.step(avg_loss)
        
        # Save model
        t.save(model.state_dict(), checkpoint_path)
        print(f"checkpoint saved as '{checkpoint_path}'")


if __name__ == "__main__":
    train_model(
        epochs=100,
        batch_size=64,  # Smaller batch size for latent space
        lr=1e-3,
        num_frames=10000,
        latent_dim=16
    )
