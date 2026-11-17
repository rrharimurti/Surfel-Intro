"""Differentiable CUDA Gaussian surfel rasterization with depth distortion.

Backwards-compatible: rasterize(...) returns RGB, and rasterize_from_projected
returns RGB unless return_distortion=True.
"""
import torch
from torch.autograd import Function


class _RasterizeWithDistortion(Function):
    @staticmethod
    def forward(ctx, means, conics, depths, colors, opacities, height, width, background):
        from . import _C
        means, conics, depths, colors, opacities = (
            x.contiguous() for x in (means, conics, depths, colors, opacities)
        )
        rgb, distortion, moments = _C.forward(
            means, conics, depths, colors, opacities,
            int(height), int(width), float(background)
        )
        ctx.save_for_backward(means, conics, depths, colors, opacities, rgb, moments)
        ctx.height, ctx.width, ctx.background = int(height), int(width), float(background)
        return rgb, distortion

    @staticmethod
    def backward(ctx, grad_rgb, grad_distortion):
        from . import _C
        means, conics, depths, colors, opacities, rgb, moments = ctx.saved_tensors
        if grad_rgb is None:
            grad_rgb = torch.zeros_like(rgb)
        if grad_distortion is None:
            grad_distortion = torch.zeros((ctx.height, ctx.width),
                                          dtype=rgb.dtype, device=rgb.device)
        gm, gq, gz, gc, go = _C.backward(
            means, conics, depths, colors, opacities, rgb, moments,
            grad_rgb.contiguous(), grad_distortion.contiguous(),
            ctx.height, ctx.width, ctx.background
        )
        return gm, gq, gz, gc, go, None, None, None


def rasterize_with_distortion(means, conics, depths, colors, opacities,
                              height, width, background=0.0):
    """Inputs must be depth-sorted float32 CUDA tensors. Returns (RGB, distortion_map)."""
    return _RasterizeWithDistortion.apply(means, conics, depths, colors,
                                          opacities, height, width, background)


def rasterize(means, conics, colors, opacities, height, width, background=0.0):
    """Legacy RGB-only interface; input surfels must already be front-to-back sorted."""
    unused_depths = means.new_zeros((means.size(0),))
    rgb, _ = rasterize_with_distortion(means, conics, unused_depths, colors,
                                        opacities, height, width, background)
    return rgb


def rasterize_from_projected(center_2d, axis_u, axis_v, depth,
                             colors, opacities, height, width,
                             blur=0.25, background=0.0,
                             return_distortion=False, depth_scale=None):
    """Project-space Gaussian + CUDA rasterization.

    Returns RGB [H,W,3] or (RGB [H,W,3], depth_distortion [H,W]).
    If return_distortion=True, normalized depth z/scale is used for the loss.
    Default scale is a *detached* median absolute camera depth, reducing the
    need to retune lambda when the scene's distance scale changes.
    """
    A = torch.stack((axis_u, axis_v), dim=-1)
    covariance = A @ A.transpose(-1, -2)
    covariance = covariance + float(blur) * torch.eye(
        2, device=covariance.device, dtype=covariance.dtype
    )
    inv = torch.linalg.inv(covariance)
    conics = torch.stack((inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]), dim=-1)
    depth = depth.reshape(-1)
    order = torch.argsort(depth, stable=True)
    if depth_scale is None:
        depth_scale = depth.detach().abs().median().clamp_min(1e-3)
    normalized_depth = depth / depth_scale
    rgb, distortion = rasterize_with_distortion(
        center_2d[order], conics[order], normalized_depth[order],
        colors[order], opacities.reshape(-1)[order],
        height, width, background
    )
    if return_distortion:
        return rgb, distortion
    return rgb
