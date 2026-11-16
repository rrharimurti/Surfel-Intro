import torch
from torch.autograd import Function


class _Rasterize(Function):
    @staticmethod
    def forward(ctx, means, conics, colors, opacities, height, width, background):
        from . import _C
        means, conics, colors, opacities = (
            x.contiguous() for x in (means, conics, colors, opacities)
        )
        (out,) = _C.forward(means, conics, colors, opacities,
                            int(height), int(width), float(background))
        ctx.save_for_backward(means, conics, colors, opacities, out)
        ctx.height, ctx.width, ctx.background = height, width, background
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from . import _C
        means, conics, colors, opacities, out = ctx.saved_tensors
        gm, gq, gc, go = _C.backward(
            means, conics, colors, opacities, out, grad_output.contiguous(),
            int(ctx.height), int(ctx.width), float(ctx.background)
        )
        return gm, gq, gc, go, None, None, None


def rasterize(means, conics, colors, opacities, height, width, background=0.0):
    """Front-to-back sorted Gaussian surfels; returns [H,W,3] RGB.

    means: [N,2] pixel-space centers (x,y)
    conics: [N,3] inverse covariance (a,b,c) for [[a,b],[b,c]]
    colors: [N,3] RGB; opacities: [N], alpha before Gaussian weighting
    All tensors must be float32 CUDA, ordered front to back, and contiguous.
    """
    return _Rasterize.apply(means, conics, colors, opacities,
                            height, width, background)


def rasterize_from_projected(center_2d, axis_u, axis_v, depth,
                             colors, opacities, height, width,
                             blur=0.25, background=0.0):
    """Differentiable covariance construction + CUDA rasterization.

    Pass projected (scaled) axes as offsets FROM center_2d, not endpoints.
    Camera projection, axes, colors and opacity remain autograd-connected.
    """
    A = torch.stack((axis_u, axis_v), dim=-1)  # [N,2,2]
    covariance = A @ A.transpose(-1, -2)
    covariance = covariance + float(blur) * torch.eye(
        2, device=covariance.device, dtype=covariance.dtype
    )
    inv = torch.linalg.inv(covariance)
    conics = torch.stack((inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]), dim=-1)
    # Indices are not differentiable; gather is, so gradients reach all inputs.
    order = torch.argsort(depth.reshape(-1), stable=True)
    return rasterize(center_2d[order], conics[order], colors[order],
                     opacities.reshape(-1)[order], height, width, background)
