import torch
from surfel_cuda import rasterize
from test_cuda import reference

assert torch.cuda.is_available(), 'Requires an NVIDIA GPU'
N, H, W = 960, 128, 128
D = 'cuda'
means = (torch.rand(N, 2, device=D) * W).requires_grad_()
conics = torch.zeros(N, 3, device=D)
conics[:, 0] = .08
conics[:, 2] = .08
conics.requires_grad_()
colors = torch.rand(N, 3, device=D).requires_grad_()
opacities = torch.full((N,), .12, device=D, requires_grad=True)
params = (means, conics, colors, opacities)

def timed(fn, iterations=10):
    for _ in range(3):
        v = fn(*params); v.square().mean().backward()
        for p in params: p.grad = None
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        v = fn(*params); v.square().mean().backward()
        for p in params: p.grad = None
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations

# Both methods do all N*H*W interactions; reference constructs many intermediate tensors.
for name, fn in [('PyTorch', lambda *p: reference(*p, H, W, 0.0)),
                 ('CUDA', lambda *p: rasterize(*p, H, W, 0.0))]:
    print(name, f'{timed(fn):.2f} ms/step (render+backward)')
