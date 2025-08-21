import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_device


class VideoVAE(nn.Module):
    """
    VAE for video frames: 320x180x3 -> 32x18x<latent_dim>
    Compression ratio: 10x in each spatial dimension
    """
    
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 320x180 -> 32x18 (exactly 10x reduction in each dim)
        self.encoder = nn.Sequential(
            # 320x180 -> 160x90
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # 160x90 -> 80x45
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # 80x45 -> 40x23 (45//2 = 22, need +1 for 23 due to padding)
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # 40x23 -> 20x12 (23//2 = 11, need +1 for 12 due to padding)  
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # 20x12 -> 10x6
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Adaptive pool to get exactly 32x18 (upsampling from 10x6)
        self.encoder_pool = nn.Upsample(size=(18, 32), mode='nearest')
        
        # Final conv layers for mu and logvar
        self.fc_mu = nn.Conv2d(128, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(128, latent_dim, 1)
        
        # Decoder: 18x32x<latent_dim> -> 180x320x3
        self.decoder_input = nn.Conv2d(latent_dim, 128, 1)
        
        # First downsample to match encoder path
        self.decoder_downsample = nn.Upsample(size=(6, 10), mode='nearest')  # 18x32 -> 6x10
        
        self.decoder = nn.Sequential(
            # 6x10 -> 12x20
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # 12x20 -> 23x40 (note: will be slightly off, need to crop)
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # 23x40 -> 45x80
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # 45x80 -> 90x160
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # 90x160 -> 180x320
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # Final layer
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] to match input normalization
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VAE initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
        print(f"Latent space: 32x18x{latent_dim} = {32*18*latent_dim:,} values")
        
    def encode(self, x):
        """
        Encode input to latent space
        Args:
            x: [B, 3, 180, 320] input frames
        Returns:
            mu, logvar: [B, latent_dim, 18, 32] mean and log variance
        """
        h = self.encoder(x)  # Should be [B, 128, 6, 10]
        h = self.encoder_pool(h)  # Upsample to [B, 128, 18, 32]
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent space to output
        Args:
            z: [B, latent_dim, 18, 32] latent codes
        Returns:
            reconstruction: [B, 3, 180, 320] reconstructed frames
        """
        h = self.decoder_input(z)  # [B, 128, 18, 32]
        h = self.decoder_downsample(h)  # [B, 128, 6, 10]
        reconstruction = self.decoder(h)  # [B, 3, 180, 320]
        
        # Ensure exact output size
        if reconstruction.shape[-2:] != (180, 320):
            reconstruction = F.interpolate(reconstruction, size=(180, 320), mode='bilinear', align_corners=False)
        
        return reconstruction
    
    def forward(self, x):
        """
        Full forward pass
        Args:
            x: [B, 3, 180, 320] input frames
        Returns:
            reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def sample(self, num_samples, device=None):
        """
        Sample from latent space
        Args:
            num_samples: number of samples to generate
            device: torch device
        Returns:
            samples: [num_samples, 3, 180, 320] generated frames
        """
        if device is None:
            device = next(self.parameters()).device
            
        z = torch.randn(num_samples, self.latent_dim, 18, 32, device=device)
        samples = self.decode(z)
        return samples


def vae_loss(reconstruction, target, mu, logvar, beta=1.0):
    """
    VAE loss function
    Args:
        reconstruction: reconstructed frames
        target: original frames  
        mu: mean of latent distribution
        logvar: log variance of latent distribution
        beta: weight for KL divergence (beta-VAE)
    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, target, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * beta * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss


def create_video_vae(latent_dim=8):
    """Create VideoVAE model"""
    return VideoVAE(latent_dim=latent_dim)


if __name__ == "__main__":
    # Test the VAE
    device = get_device()
    vae = create_video_vae(latent_dim=8).to(device)
    
    # Test with dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 180, 320).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    reconstruction, mu, logvar = vae(x)
    
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss
    total_loss, recon_loss, kl_loss = vae_loss(reconstruction, x, mu, logvar)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test sampling
    samples = vae.sample(2, device)
    print(f"Sample shape: {samples.shape}")
    
    print("VAE test completed successfully!")
