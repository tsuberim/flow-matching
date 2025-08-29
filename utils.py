import torch as t


def get_device():
    """
    Get the best available device: MPS > CUDA > CPU
    
    Returns:
        torch.device: The best available device
    """
    if t.backends.mps.is_available():
        device = t.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
        return device
    elif t.cuda.is_available():
        # Initialize CUDA context properly for multi-GPU
        device = t.device('cuda')
        
        # Set primary CUDA device and initialize context
        if t.cuda.device_count() > 1:
            # For multi-GPU, ensure primary context is set
            t.cuda.set_device(0)  # Set primary device
            print(f"Using device: {device} (NVIDIA GPU) - {t.cuda.device_count()} GPUs available")
        else:
            print(f"Using device: {device} (NVIDIA GPU)")
        
        return device
    else:
        device = t.device('cpu')
        print(f"Using device: {device}")
        return device
