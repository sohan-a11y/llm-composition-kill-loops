"""HRR binding ops (Plate 1995) via FFT. Verify correctness before using."""
import torch, math

def bind(x, y):
    """Circular convolution: x (*) y = irfft(rfft(x) * rfft(y))"""
    n = x.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(y, n=n), n=n)

def inv(y):
    """Approximate inverse = involution: (y1, yd, y_{d-1}, ..., y2)"""
    return torch.cat([y[..., :1], y[..., 1:].flip(-1)], dim=-1)

def unbind(b, y):
    return bind(b, inv(y))

def rand_hv(shape, d, device='cpu', gen=None):
    """i.i.d. N(0, 1/d) as Plate specifies."""
    return torch.randn(*shape, d, generator=gen, device=device) / math.sqrt(d)
