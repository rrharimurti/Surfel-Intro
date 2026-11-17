#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> raster_forward_cuda(
    torch::Tensor means, torch::Tensor conics, torch::Tensor depths,
    torch::Tensor colors, torch::Tensor opacities,
    int64_t height, int64_t width, double bg);
std::vector<torch::Tensor> raster_backward_cuda(
    torch::Tensor means, torch::Tensor conics, torch::Tensor depths,
    torch::Tensor colors, torch::Tensor opacities,
    torch::Tensor rendered, torch::Tensor moments,
    torch::Tensor grad_rgb, torch::Tensor grad_dist,
    int64_t height, int64_t width, double bg);

static void check_inputs(torch::Tensor means, torch::Tensor conics,
                         torch::Tensor depths, torch::Tensor colors,
                         torch::Tensor opacities) {
    TORCH_CHECK(means.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "float32 required");
    TORCH_CHECK(means.is_contiguous() && conics.is_contiguous() &&
                depths.is_contiguous() && colors.is_contiguous() &&
                opacities.is_contiguous(), "all inputs must be contiguous");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 2, "means must be [N,2]");
    auto n = means.size(0);
    TORCH_CHECK(conics.sizes() == torch::IntArrayRef({n, 3}), "conics must be [N,3]");
    TORCH_CHECK(depths.dim() == 1 && depths.size(0) == n, "depths must be [N]");
    TORCH_CHECK(colors.sizes() == torch::IntArrayRef({n, 3}), "colors must be [N,3]");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == n, "opacities must be [N]");
    TORCH_CHECK(conics.is_cuda() && depths.is_cuda() && colors.is_cuda() &&
                opacities.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(conics.scalar_type() == torch::kFloat32 &&
                depths.scalar_type() == torch::kFloat32 &&
                colors.scalar_type() == torch::kFloat32 &&
                opacities.scalar_type() == torch::kFloat32, "float32 required");
    TORCH_CHECK(means.device() == conics.device() && means.device() == depths.device() &&
                means.device() == colors.device() && means.device() == opacities.device(),
                "device mismatch");
}

std::vector<torch::Tensor> forward(torch::Tensor means, torch::Tensor conics,
                                   torch::Tensor depths, torch::Tensor colors,
                                   torch::Tensor opacities, int64_t height,
                                   int64_t width, double bg) {
    check_inputs(means, conics, depths, colors, opacities);
    TORCH_CHECK(height > 0 && width > 0, "invalid dimensions");
    TORCH_CHECK(bg >= 0.0 && bg <= 1.0, "background should be in [0,1]");
    return raster_forward_cuda(means, conics, depths, colors, opacities,
                               height, width, bg);
}
std::vector<torch::Tensor> backward(torch::Tensor means, torch::Tensor conics,
                                    torch::Tensor depths, torch::Tensor colors,
                                    torch::Tensor opacities, torch::Tensor rendered,
                                    torch::Tensor moments, torch::Tensor grad_rgb,
                                    torch::Tensor grad_dist, int64_t height,
                                    int64_t width, double bg) {
    check_inputs(means, conics, depths, colors, opacities);
    auto is_rgb_shape = [height, width](const torch::Tensor& t) {
    return t.dim() == 3 &&
           t.size(0) == height &&
           t.size(1) == width &&
           t.size(2) == 3;
    };
    TORCH_CHECK(
        is_rgb_shape(rendered) &&
        is_rgb_shape(moments) &&
        is_rgb_shape(grad_rgb),
        "RGB/moments gradients must be [H,W,3]"
    );
    TORCH_CHECK(
        grad_dist.dim() == 2 &&
        grad_dist.size(0) == height &&
        grad_dist.size(1) == width,
        "distortion gradient must be [H,W]"
    );
    for (auto tensor : {rendered, moments, grad_rgb, grad_dist}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "output tensors must be contiguous CUDA");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "output tensors must be float32");
        TORCH_CHECK(tensor.device() == means.device(), "device mismatch");
    }
    return raster_backward_cuda(means, conics, depths, colors, opacities,
                                rendered, moments, grad_rgb, grad_dist,
                                height, width, bg);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Surfel RGB + distortion forward (CUDA)");
    m.def("backward", &backward, "Surfel RGB + distortion backward (CUDA)");
}
