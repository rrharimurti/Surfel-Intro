"""Verify custom surfel CUDA forward/backward against native PyTorch.

Run from the repository root:
    env PYTHONPATH=. python verify_surfel_gradients.py

Requires a compiled surfel_cuda extension and a working CUDA GPU.
Checks the rasterizer and the gradient chain through projected ellipse axes.
"""

import torch
from surfel_cuda import rasterize_from_projected


def reference_render(means, axis_u, axis_v, depth, colors, opacities,
                     height, width, blur=0.25, background=1.0):
    """Independent, differentiable PyTorch implementation of extension math."""
    A = torch.stack((axis_u, axis_v), dim=-1)
    covariance = A @ A.transpose(-1, -2)
    covariance = covariance + blur * torch.eye(2, dtype=means.dtype, device=means.device)
    inv = torch.linalg.inv(covariance)

    order = torch.argsort(depth.reshape(-1), stable=True)
    m = means[order]
    qconic = inv[order]
    c = colors[order]
    o = opacities.reshape(-1)[order]

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=means.dtype, device=means.device),
        torch.arange(width, dtype=means.dtype, device=means.device),
        indexing="ij",
    )
    result = torch.zeros((height, width, 3), dtype=means.dtype, device=means.device)
    transmittance = torch.ones((height, width, 1), dtype=means.dtype, device=means.device)

    for i in range(m.shape[0]):
        dx, dy = xx - m[i, 0], yy - m[i, 1]
        a, b, cc = qconic[i, 0, 0], qconic[i, 0, 1], qconic[i, 1, 1]
        q = a * dx**2 + 2 * b * dx * dy + cc * dy**2
        alpha = (o[i] * torch.exp(-0.5 * q)).clamp(0.0, 0.999)
        alpha = torch.where((q >= 0.0) & (q <= 18.0), alpha, 0.0).unsqueeze(-1)
        result = result + transmittance * alpha * c[i]
        transmittance = transmittance * (1.0 - alpha)

    # With moderate opacities, the CUDA early-stop threshold cannot be reached.
    return result + background * transmittance


def clone_for_grad(tensors):
    return [t.detach().clone().requires_grad_(True) for t in tensors]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    torch.manual_seed(42)
    device = "cuda"
    h, w, n = 7, 8, 3
    means = torch.tensor([[2.4, 3.4], [5.2, 2.8], [3.8, 5.1]], device=device)
    axis_u = torch.tensor([[2.5, 0.20], [2.7, -0.10], [2.6, 0.25]], device=device)
    axis_v = torch.tensor([[0.10, 2.7], [0.15, 2.5], [-0.15, 2.4]], device=device)
    depth = torch.tensor([2.0, 1.0, 3.0], device=device)
    colors = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.7, 0.3],
                           [0.2, 0.3, 0.8]], device=device)
    opacities = torch.tensor([0.2, 0.35, 0.3], device=device)
    inputs = (means, axis_u, axis_v, colors, opacities)
    labels = ("center_2d", "axis_u", "axis_v", "colors", "opacities")
    cuda_inputs = clone_for_grad(inputs)
    ref_inputs = clone_for_grad(inputs)
    background, blur = 1.0, 0.25

    rendered = rasterize_from_projected(
        *cuda_inputs[:3], depth, *cuda_inputs[3:], h, w,
        blur=blur, background=background)
    reference = reference_render(
        *ref_inputs[:3], depth, *ref_inputs[3:], h, w,
        blur=blur, background=background)

    print("White-background forward max abs error:",
          (rendered - reference).abs().max().item())
    torch.testing.assert_close(rendered, reference, rtol=3e-4, atol=3e-4)

    # A nontrivial scalar loss is equivalent to choosing a gradient at the
    # image output and propagating it backwards through the custom kernel.
    upstream = torch.randn_like(rendered)
    (rendered * upstream).sum().backward()
    (reference * upstream).sum().backward()

    print("\nBackward gradients (CUDA vs PyTorch reference):")
    for name, cuda_t, ref_t in zip(labels, cuda_inputs, ref_inputs):
        diff = (cuda_t.grad - ref_t.grad).abs()
        print(f"  {name:12s} max_abs={diff.max().item():.6g} "
              f"mean_abs={diff.mean().item():.6g} "
              f"max_grad={ref_t.grad.abs().max().item():.6g}")
        torch.testing.assert_close(cuda_t.grad, ref_t.grad,
                                   rtol=5e-3, atol=3e-3)

    # Selected finite difference checks: use only stable points away from
    # alpha clamp / Gaussian truncation / depth-order ties.
    print("\nSelected CUDA finite differences (float32, epsilon=0.01):")
    epsilon = 0.01
    checks = [(0, (0, 0)), (1, (1, 0)), (2, (2, 1)),
              (3, (0, 2)), (4, (1,))]

    def scalar_loss(values):
        image = rasterize_from_projected(
            *values[:3], depth, *values[3:], h, w,
            blur=blur, background=background)
        return (image * upstream).sum().item()

    for which, coord in checks:
        plus = [x.detach().clone() for x in inputs]
        minus = [x.detach().clone() for x in inputs]
        plus[which][coord] += epsilon
        minus[which][coord] -= epsilon
        numerical = (scalar_loss(plus) - scalar_loss(minus)) / (2 * epsilon)
        analytic = cuda_inputs[which].grad[coord].item()
        delta = abs(numerical - analytic)
        print(f"  {labels[which]:12s} at {coord}: "
              f"analytic={analytic:+.6f}, numerical={numerical:+.6f}, "
              f"abs_diff={delta:.6f}")
        if delta > 0.01 + 0.05 * abs(numerical):
            raise AssertionError(f"Finite difference mismatch in {labels[which]}")

    print("\nPASS: Forward, projected-input backward, and finite differences agree.")
    print("NOTE: Depth-only ordering is discrete, so no direct depth gradient is provided.")


if __name__ == "__main__":
    main()
