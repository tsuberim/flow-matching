import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):  # VAE latent: 16 channels
        super().__init__()
        
        # Encoder for VAE latents (32x18 input)
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)    # 32x18
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 16x9
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # 8x5
        self.enc4 = nn.Conv2d(256, 512, 3, stride=2, padding=1) # 4x3
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(512, 512, 3, padding=1)     # 4x3
        
        # Decoder with proper upsampling - match actual encoder sizes
        self.dec1 = nn.Upsample(size=(5, 8), mode='nearest')              # 3x4 -> 5x8
        self.dec1_conv = nn.Conv2d(512, 256, 3, padding=1)                # 5x8
        self.dec2 = nn.Upsample(size=(9, 16), mode='nearest')             # 5x8 -> 9x16
        self.dec2_conv = nn.Conv2d(512, 128, 3, padding=1)                # 9x16 (256+256 channels input)
        self.dec3 = nn.Upsample(size=(18, 32), mode='nearest')            # 9x16 -> 18x32
        self.dec3_conv = nn.Conv2d(256, 64, 3, padding=1)                 # 18x32 (128+128 channels input)
        self.dec4 = nn.Conv2d(128, out_channels, 3, padding=1)            # 18x32 (64+64 channels input)
        
        self.relu = nn.ReLU()
        
        # Count and print parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"UNet initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
    def forward(self, x, t=None):
        # Encoder
        x1 = self.relu(self.enc1(x))    # 32x18, 64
        x2 = self.relu(self.enc2(x1))   # 16x9, 128  
        x3 = self.relu(self.enc3(x2))   # 8x4, 256 (not 8x5!)
        x4 = self.relu(self.enc4(x3))   # 4x2, 512 (not 4x3!)
        
        # Debug print actual sizes
        # print(f"x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}")
        
        # Bottleneck
        bottleneck = self.relu(self.bottleneck(x4))  # 4x2, 512
        
        # Decoder with skip connections - use actual encoder sizes
        y1 = self.dec1(bottleneck)  # 3x4 -> 5x8
        y1 = self.relu(self.dec1_conv(y1))  # 5x8, 256
        
        y2 = self.dec2(torch.cat([y1, x3], dim=1))  # 5x8 -> 9x16
        y2 = self.relu(self.dec2_conv(y2))  # 9x16, 128
        
        y3 = self.dec3(torch.cat([y2, x2], dim=1))  # 9x16 -> 18x32
        y3 = self.relu(self.dec3_conv(y3))  # 18x32, 64
        
        y4 = self.dec4(torch.cat([y3, x1], dim=1))  # 18x32, 16
        
        return y4


def create_latent_unet(latent_dim=16):
    """Create UNet model for VAE latent space (32x18x16)"""
    return UNet(in_channels=latent_dim, out_channels=latent_dim)


def create_mnist_unet():
    """Create a simple UNet model for MNIST (28x28 grayscale images) - DEPRECATED"""
    return UNet(in_channels=1, out_channels=1)


def test_latent_unet():
    """Test the latent UNet with sample data"""
    print("Testing Latent UNet...")
    
    # Create model
    latent_dim = 16
    model = create_latent_unet(latent_dim=latent_dim)
    
    # Test data - VAE latent embeddings (batch_size=4, channels=16, height=18, width=32)
    batch_size = 4
    sample_input = torch.randn(batch_size, latent_dim, 18, 32)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Input range: [{sample_input.min():.3f}, {sample_input.max():.3f}]")
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check dimensions
    assert output.shape == sample_input.shape, f"Shape mismatch: {output.shape} != {sample_input.shape}"
    print("✅ Shape test passed")
    
    # Test with different batch sizes
    for test_batch_size in [1, 8, 16]:
        test_input = torch.randn(test_batch_size, latent_dim, 18, 32)
        test_output = model(test_input)
        assert test_output.shape == test_input.shape
        print(f"✅ Batch size {test_batch_size} test passed")
    
    # Test gradient flow
    model.train()
    loss_input = torch.randn(batch_size, latent_dim, 18, 32, requires_grad=True)
    output = model(loss_input)
    loss = output.mean()
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients, "No gradients found"
    print("✅ Gradient flow test passed")
    
    print("All tests passed! 🎉")
    return model


if __name__ == "__main__":
    test_latent_unet()