import torch
from surfel_cuda import rasterize


def reference(means, conics, colors, opacities, height, width, background):
    yy, xx = torch.meshgrid(torch.arange(height, device=means.device),
                            torch.arange(width, device=means.device), indexing="ij")
    out = torch.zeros(height, width, 3, device=means.device)
    trans = torch.ones(height, width, 1, device=means.device)
    for i in range(means.shape[0]):
        dx, dy = xx - means[i, 0], yy - means[i, 1]
        a, b, c = conics[i]
        q = a*dx**2 + 2*b*dx*dy + c*dy**2
        # Exact same cutoff and clamp as CUDA implementation.
        alpha = (opacities[i] * torch.exp(-0.5*q)).clamp(0., 0.999)
        alpha = torch.where((q >= 0) & (q <= 18), alpha, 0).unsqueeze(-1)
        out = out + trans * alpha * colors[i]
        trans = trans * (1-alpha)
        # Test uses moderate alpha so CUDA's early termination never triggers.
    return out + background * trans


def test_forward_backward():
    assert torch.cuda.is_available()
    torch.manual_seed(3)
    device = 'cuda'
    n, h, w = 5, 12, 14
    means = torch.tensor([[2., 3.], [7., 6.], [10., 9.], [4., 8.], [9., 2.]], device=device)
    conics = torch.tensor([[.12, 0., .15], [.10, .01, .13], [.20, 0., .15],
                           [.11, -.01, .12], [.15, 0., .10]], device=device)
    colors = torch.rand(n, 3, device=device)
    opacities = torch.tensor([.2, .3, .4, .2, .3], device=device)
    inputs = [means, conics, colors, opacities]
    cuda_inputs = [t.clone().detach().requires_grad_() for t in inputs]
    ref_inputs = [t.clone().detach().requires_grad_() for t in inputs]
    out = rasterize(*cuda_inputs, h, w, .1)
    baseline = reference(*ref_inputs, h, w, .1)
    torch.testing.assert_close(out, baseline, atol=3e-5, rtol=3e-5)
    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()
    (baseline * upstream).sum().backward()
    for name, a, b in zip(('means', 'conics', 'colors', 'opacities'), cuda_inputs, ref_inputs):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-3, rtol=2e-3,
                                   msg=lambda m: f'{name}: {m}')
    print('Forward and backward match PyTorch reference')


if __name__ == '__main__':
    test_forward_backward()
