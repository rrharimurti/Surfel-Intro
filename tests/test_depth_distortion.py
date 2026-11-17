"""Run on a CUDA machine after build_ext --inplace:

    env PYTHONPATH=. python tests/test_depth_distortion.py

Checks CUDA RGB/depth-distortion and ALL FIVE input gradient tensors against
ordinary PyTorch autograd. Also checks the camera depth gradient numerically.
"""
import torch
from surfel_cuda import rasterize_with_distortion, rasterize_from_projected


def reference(means, conics, depths, colors, opacities, height, width, background):
    yy, xx = torch.meshgrid(
        torch.arange(height, device=means.device),
        torch.arange(width, device=means.device), indexing='ij'
    )
    dtype = means.dtype
    T = torch.ones((height, width), device=means.device, dtype=dtype)
    rgb = torch.zeros((height, width, 3), device=means.device, dtype=dtype)
    W = torch.zeros_like(T)
    M1 = torch.zeros_like(T)
    M2 = torch.zeros_like(T)
    for i in range(means.size(0)):
        dx = xx - means[i, 0]
        dy = yy - means[i, 1]
        a, b, c = conics[i]
        q = a * dx.square() + 2 * b * dx * dy + c * dy.square()
        alpha = (opacities[i] * torch.exp(-0.5 * q)).clamp(0, 0.999)
        alpha = torch.where((q >= 0) & (q <= 18), alpha, 0)
        w = T * alpha
        rgb = rgb + w.unsqueeze(-1) * colors[i]
        W = W + w
        M1 = M1 + w * depths[i]
        M2 = M2 + w * depths[i].square()
        T = T * (1 - alpha)
    distortion = W * M2 - M1.square()
    return rgb + background * T.unsqueeze(-1), distortion


def test_with_cuda():
    assert torch.cuda.is_available(), 'Requires NVIDIA CUDA GPU'
    device = 'cuda'
    torch.manual_seed(42)
    n, H, W = 5, 14, 13
    means = torch.tensor([[4., 3.], [7., 8.], [8., 5.], [3., 9.], [9., 10.]], device=device)
    conics = torch.tensor([[.11, .01, .17], [.12, 0, .09], [.15, -.01, .16],
                           [.10, .01, .10], [.13, 0, .12]], device=device)
    depths = torch.tensor([1.1, 1.4, 2.0, 2.7, 3.1], device=device)
    colors = torch.rand((n, 3), device=device)
    opacities = torch.tensor([.2, .4, .3, .25, .2], device=device)
    base = [means, conics, depths, colors, opacities]
    cuda_inputs = [x.clone().detach().requires_grad_(True) for x in base]
    ref_inputs = [x.clone().detach().requires_grad_(True) for x in base]
    rgb, dist = rasterize_with_distortion(*cuda_inputs, H, W, 1.)
    ref_rgb, ref_dist = reference(*ref_inputs, H, W, 1.)
    torch.testing.assert_close(rgb, ref_rgb, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dist, ref_dist, atol=3e-5, rtol=3e-5)
    rgb_grad = torch.randn_like(rgb)
    distortion_grad = torch.randn_like(dist) * 0.15
    (rgb * rgb_grad).sum().add((dist * distortion_grad).sum()).backward()
    (ref_rgb * rgb_grad).sum().add((ref_dist * distortion_grad).sum()).backward()
    for label, c, r in zip(('means', 'conics', 'depths', 'colors', 'opacities'), cuda_inputs, ref_inputs):
        torch.testing.assert_close(c.grad, r.grad, atol=3e-3, rtol=3e-3,
                                   msg=lambda msg: f'{label}: {msg}')
        print(label, 'gradient: PASS')

    # Finite differences of an unambiguously sorted depth value.
    eps = 1e-3
    with torch.no_grad():
        plus = depths.clone(); plus[2] += eps
        minus = depths.clone(); minus[2] -= eps
        Lp = rasterize_with_distortion(means, conics, plus, colors, opacities, H, W, 1.)[1].mean()
        Lm = rasterize_with_distortion(means, conics, minus, colors, opacities, H, W, 1.)[1].mean()
        numerical = ((Lp - Lm)/(2*eps)).item()
    z = depths.clone().detach().requires_grad_()
    loss = rasterize_with_distortion(means, conics, z, colors, opacities, H, W, 1.)[1].mean()
    loss.backward()
    analytic = z.grad[2].item()
    assert abs(analytic-numerical) < 3e-3 + 0.03*abs(numerical), (analytic, numerical)
    print('depth finite difference: PASS', analytic, numerical)
    print('All RGB/distortion forward and backward checks passed.')


if __name__ == '__main__':
    test_with_cuda()
