import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_device
from einops import rearrange
import math


class VideoVAE(nn.Module):
    """
    VAE for video frames: 320x180x3 -> 32x18x<latent_dim>
    Compression ratio: 10x in each spatial dimension
    """
    
    def __init__(self, latent_dim=8, model_size=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_size = model_size
        
        # Improved Encoder with BatchNorm and more layers
        self.encoder = nn.Sequential(
            # 320x180 -> 160x90
            nn.Conv2d(3, 32 * model_size, 4, stride=1, padding=1),
            nn.BatchNorm2d(32 * model_size),
            nn.ReLU(),

            # 160x90 -> 80x45
            nn.Conv2d(32 * model_size, 64 * model_size, 4, stride=1, padding=1),
            nn.BatchNorm2d(64 * model_size),
            nn.ReLU(),

            # 80x45 -> 40x23
            nn.Conv2d(64 * model_size, 128 * model_size, (4, 9), stride=2, padding=1),
            nn.BatchNorm2d(128 * model_size),
            nn.ReLU(),

            # Additional layer 1: 40x23 -> 20x12
            nn.Conv2d(128 * model_size, 128 * model_size, (8, 14), stride=1, padding=1),
            nn.BatchNorm2d(128 * model_size),
            nn.ReLU(),

            # Additional layer 2: 20x12 -> 10x6
            nn.Conv2d(128 * model_size, 128 * model_size, (6, 10), stride=2, padding=1),
            nn.BatchNorm2d(128 * model_size),
            nn.ReLU(),

            # Final layer: 10x6 -> 5x3
            nn.Conv2d(128 * model_size, 256 * model_size, (8, 8), stride=2, padding=1),
            nn.BatchNorm2d(256 * model_size),
            nn.ReLU(),
        )

        
        # Final conv layers for mu and logvar (updated for 256 channels)
        self.fc_mu = nn.Conv2d(256 * model_size, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(256 * model_size, latent_dim, 1)

        # Improved Decoder with BatchNorm
        self.decoder_input = nn.Conv2d(latent_dim, 256 * model_size, 1)

        self.decoder = nn.Sequential(
            # 18x32 -> 36x64
            nn.ConvTranspose2d(256 * model_size, 128 * model_size, 8, stride=2, padding=1),
            nn.BatchNorm2d(128 * model_size),
            nn.ReLU(),

            # 36x64 -> 72x128
            nn.ConvTranspose2d(128 * model_size, 128 * model_size, (8, 14), stride=2, padding=1),
            nn.BatchNorm2d(128 * model_size),
            nn.ReLU(),

            # 72x128 -> 144x256
            nn.ConvTranspose2d(128 * model_size, 64 * model_size, (8, 16), stride=1, padding=1),
            nn.BatchNorm2d(64 * model_size),
            nn.ReLU(),

            # 144x256 -> adjust to get closer to target size
            nn.ConvTranspose2d(64 * model_size, 32 * model_size, (7, 7), stride=2, padding=1),
            nn.BatchNorm2d(32 * model_size),
            nn.ReLU(),

            # Final refinement
            nn.Conv2d(32 * model_size, 16 * model_size, 4, padding=1),
            nn.BatchNorm2d(16 * model_size),
            nn.ReLU(),

            # Output layer
            nn.Conv2d(16 * model_size, 3, 3, padding=1),
            nn.Tanh()
        )

        # Initialize weights properly
        self._initialize_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VAE initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
        print(f"Latent space: 32x18x{latent_dim} = {32*18*latent_dim:,} values")
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize output layers with Kaiming as well
        nn.init.kaiming_normal_(self.fc_mu.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.kaiming_normal_(self.fc_logvar.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.fc_logvar.bias, -5)  # Moderate initial variance
        
    def encode(self, x):
        """
        Encode input to latent space
        Args:
            x: [B, 3, 180, 320] input frames
        Returns:
            mu, logvar: [B, latent_dim, 18, 32] mean and log variance
        """
        h = self.encoder(x)  # Should be [B, 128, 6, 10]
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick with numerical stability"""
        # Clamp logvar to prevent extreme values
        logvar_clamped = torch.clamp(logvar, min=-20, max=20)
        std = torch.exp(0.5 * logvar_clamped)
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
        h = self.decoder_input(z)  # [B, 256, 18, 32]
        reconstruction = self.decoder(h)  # [B, 3, ~144, ~256]
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


def vae_loss(vae, frames, beta=0.0, gamma=0.001):
    """
    VAE loss function
    Args:
        vae: VAE model
        frames: original frames (B, T, 3, 180, 320)
        beta: weight for KL divergence (beta-VAE)
    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    t = frames.shape[1]
    input = rearrange(frames, 'b t c h w -> (b t) c h w')
    reconstruction, mu, logvar = vae(input)
    reconstruction = rearrange(reconstruction, '(b t) c h w -> b t c h w', t=t)
    mu = rearrange(mu, '(b t) c h w -> b t c h w', t=t)
    logvar = rearrange(logvar, '(b t) c h w -> b t c h w', t=t)
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, frames, reduction='mean')
    
    if t > 1:
        # Also optimize for reconstruction of the difference between consecutive frames
        frames_diff = frames[:, 1:] - frames[:, :-1]
        reconstruction_diff = reconstruction[:, 1:] - reconstruction[:, :-1]
        diff_loss = F.mse_loss(frames_diff, reconstruction_diff, reduction='mean')
    else:
        diff_loss = torch.zeros(1).to(frames.device)

    
    # Similarity loss based on temporal closeness using Gaussian weighting
    # Encourage latent codes of nearby frames to be similar
    sim_loss = 0
    sigma = t / 4  # Set sigma to quarter of sequence length
    for t1 in range(t):
        for t2 in range(t):
            # Gaussian weight based on temporal distance
            weight = math.exp(-0.5 * ((t1 - t2) / sigma) ** 2)
            sim_loss += weight * F.mse_loss(mu[:,t1], mu[:,t2], reduction='mean')
    sim_loss = gamma * sim_loss / (t * t)  # Normalize by number of pairs

    # KL divergence loss with clamping to prevent numerical instability
    logvar_clamped = torch.clamp(logvar, min=-20, max=20)  # Prevent extreme values
    kl_loss = -0.5 * beta * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
    
    # Total loss
    total_loss = recon_loss + kl_loss + sim_loss + diff_loss
    
    return total_loss, recon_loss, kl_loss, sim_loss, diff_loss


def create_video_vae(latent_dim=8, model_size=1):
    """Create VideoVAE model"""
    return VideoVAE(latent_dim=latent_dim, model_size=model_size)


if __name__ == "__main__":
    # Test the VAE
    device = get_device()
    vae = create_video_vae(latent_dim=16, model_size=1).to(device)
    
    # Test with dummy input
    batch_size = 4
    seq_len = 2
    x = torch.randn(batch_size, seq_len, 3, 180, 320).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Test loss
    total_loss, recon_loss, kl_loss, sim_loss, diff_loss = vae_loss(vae, x)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    print(f"Similarity loss: {sim_loss.item():.4f}")
    print(f"Diff loss: {diff_loss.item():.4f}")

    # Test sampling
    samples = vae.sample(2, device)
    print(f"Sample shape: {samples.shape}")
    
    print("VAE test completed successfully!")
