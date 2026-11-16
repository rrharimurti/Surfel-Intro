#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> raster_forward_cuda(
    torch::Tensor means, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, int64_t height, int64_t width, double bg);
std::vector<torch::Tensor> raster_backward_cuda(
    torch::Tensor means, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor rendered,
    torch::Tensor grad_output, int64_t height, int64_t width, double bg);

static void check_inputs(torch::Tensor means, torch::Tensor conics,
                         torch::Tensor colors, torch::Tensor opacities) {
    TORCH_CHECK(means.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "float32 required");
    TORCH_CHECK(means.is_contiguous() && conics.is_contiguous() &&
                colors.is_contiguous() && opacities.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 2, "means must be [N,2]");
    auto n = means.size(0);
    TORCH_CHECK(conics.sizes() == torch::IntArrayRef({n, 3}), "conics must be [N,3]");
    TORCH_CHECK(colors.sizes() == torch::IntArrayRef({n, 3}), "colors must be [N,3]");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == n, "opacities must be [N]");
    TORCH_CHECK(conics.is_cuda() && colors.is_cuda() && opacities.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(conics.scalar_type() == torch::kFloat32 && colors.scalar_type() == torch::kFloat32 &&
                opacities.scalar_type() == torch::kFloat32, "all tensors must be float32");
    TORCH_CHECK(means.device() == conics.device() && means.device() == colors.device() &&
                means.device() == opacities.device(), "device mismatch");
}
std::vector<torch::Tensor> forward(torch::Tensor means, torch::Tensor conics,
                                   torch::Tensor colors, torch::Tensor opacities,
                                   int64_t height, int64_t width, double bg) {
    check_inputs(means, conics, colors, opacities);
    TORCH_CHECK(height > 0 && width > 0, "invalid dimensions");
    TORCH_CHECK(bg >= 0.0 && bg <= 1.0, "background should be [0,1]");
    return raster_forward_cuda(means, conics, colors, opacities, height, width, bg);
}
std::vector<torch::Tensor> backward(torch::Tensor means, torch::Tensor conics,
                                    torch::Tensor colors, torch::Tensor opacities,
                                    torch::Tensor rendered, torch::Tensor grad_output,
                                    int64_t height, int64_t width, double bg) {
    check_inputs(means, conics, colors, opacities);
    TORCH_CHECK(rendered.is_cuda() && grad_output.is_cuda(), "rendered/grad must be CUDA");
    TORCH_CHECK(rendered.sizes() == torch::IntArrayRef({height, width, 3}), "rendered must be [H,W,3]");
    TORCH_CHECK(grad_output.sizes() == rendered.sizes(), "grad_output shape mismatch");
    TORCH_CHECK(rendered.is_contiguous() && grad_output.is_contiguous(), "rendered/grad must be contiguous");
    TORCH_CHECK(rendered.scalar_type() == torch::kFloat32 && grad_output.scalar_type() == torch::kFloat32, "float32 required");
    TORCH_CHECK(rendered.device() == means.device() && grad_output.device() == means.device(), "device mismatch");
    return raster_backward_cuda(means, conics, colors, opacities, rendered,
                                grad_output, height, width, bg);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Surfel rasterization forward (CUDA)");
    m.def("backward", &backward, "Surfel rasterization backward (CUDA)");
}
