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
        device = t.device('cuda')
        print(f"Using device: {device} (NVIDIA GPU)")
        return device
    else:
        device = t.device('cpu')
        print(f"Using device: {device}")
        return device
