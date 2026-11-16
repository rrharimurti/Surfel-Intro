#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Surfel arrays must be sorted front-to-back before calling these kernels.
// Inverse covariance [a,b,c] stands for [[a,b],[b,c]].
// One CUDA thread owns one pixel, avoiding any intermediate [N,H,W] tensor.
__global__ void render_kernel(const float* __restrict__ means,
                              const float* __restrict__ conics,
                              const float* __restrict__ colors,
                              const float* __restrict__ opacities,
                              float* __restrict__ output,
                              int n, int width, int pixels, float bg) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float x = float(p % width), y = float(p / width);
    float r = 0.f, g = 0.f, b = 0.f, T = 1.f;
    for (int i = 0; i < n; ++i) {
        float dx = x - means[2*i], dy = y - means[2*i+1];
        float a = conics[3*i], bb = conics[3*i+1], c = conics[3*i+2];
        float q = a*dx*dx + 2.f*bb*dx*dy + c*dy*dy;
        if (q < 0.f || q > 18.f) continue; // 3 sigma, avoids tiny contributions
        float weight = __expf(-0.5f*q);
        float alpha = fminf(fmaxf(opacities[i] * weight, 0.f), 0.999f);
        float w = T * alpha;
        r += w * colors[3*i]; g += w * colors[3*i+1]; b += w * colors[3*i+2];
        T *= (1.f - alpha);
        if (T < 1e-5f) break;
    }
    output[3*p] = r + T*bg;
    output[3*p+1] = g + T*bg;
    output[3*p+2] = b + T*bg;
}

// Exact reverse-mode derivatives of the *implemented, truncated* compositing
// recurrence (apart from floating point round-off). The saved output lets us
// recover the contribution behind the current splat without storing all Ts.
__global__ void backward_kernel(const float* __restrict__ means,
                                const float* __restrict__ conics,
                                const float* __restrict__ colors,
                                const float* __restrict__ opacities,
                                const float* __restrict__ rendered,
                                const float* __restrict__ grad,
                                float* __restrict__ gm,
                                float* __restrict__ gq,
                                float* __restrict__ gc,
                                float* __restrict__ go,
                                int n, int width, int pixels) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float x = float(p % width), y = float(p / width);
    float prefix_r = 0.f, prefix_g = 0.f, prefix_b = 0.f;
    float T = 1.f;
    float gr = grad[3*p], gg = grad[3*p+1], gb = grad[3*p+2];
    for (int i = 0; i < n; ++i) {
        float dx = x - means[2*i], dy = y - means[2*i+1];
        float a = conics[3*i], bb = conics[3*i+1], c = conics[3*i+2];
        float q = a*dx*dx + 2.f*bb*dx*dy + c*dy*dy;
        if (q < 0.f || q > 18.f) continue;
        float weight = __expf(-0.5f*q);
        float raw_alpha = opacities[i] * weight;
        float alpha = fminf(fmaxf(raw_alpha, 0.f), 0.999f);
        float w = T * alpha;
        float cr = colors[3*i], cg = colors[3*i+1], cb = colors[3*i+2];
        // Recover the *weighted* later splats plus background.
        float tail_r = rendered[3*p] - prefix_r - w*cr;
        float tail_g = rendered[3*p+1] - prefix_g - w*cg;
        float tail_b = rendered[3*p+2] - prefix_b - w*cb;
        float da = gr*(T*cr - tail_r/(1.f-alpha))
                 + gg*(T*cg - tail_g/(1.f-alpha))
                 + gb*(T*cb - tail_b/(1.f-alpha));
        atomicAdd(&gc[3*i], gr*w);
        atomicAdd(&gc[3*i+1], gg*w);
        atomicAdd(&gc[3*i+2], gb*w);
        // Gradient is zero where alpha was clamped.
        if (raw_alpha > 0.f && raw_alpha < 0.999f) {
            atomicAdd(&go[i], da*weight);
            float dq = -0.5f * da * raw_alpha;
            atomicAdd(&gm[2*i], -2.f*dq*(a*dx + bb*dy));
            atomicAdd(&gm[2*i+1], -2.f*dq*(bb*dx + c*dy));
            atomicAdd(&gq[3*i], dq*dx*dx);
            atomicAdd(&gq[3*i+1], dq*2.f*dx*dy);
            atomicAdd(&gq[3*i+2], dq*dy*dy);
        }
        prefix_r += w*cr; prefix_g += w*cg; prefix_b += w*cb;
        T *= (1.f-alpha);
        if (T < 1e-5f) break;
    }
}

std::vector<torch::Tensor> raster_forward_cuda(
    torch::Tensor means, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, int64_t height, int64_t width, double bg) {
    const c10::cuda::CUDAGuard device_guard(means.device());
    auto out = torch::empty({height, width, 3}, means.options());
    int pixels = int(height * width), threads = 256;
    render_kernel<<<(pixels+threads-1)/threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), colors.data_ptr<float>(),
        opacities.data_ptr<float>(), out.data_ptr<float>(), int(means.size(0)),
        int(width), pixels, float(bg));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out};
}
std::vector<torch::Tensor> raster_backward_cuda(
    torch::Tensor means, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor rendered,
    torch::Tensor grad_output, int64_t height, int64_t width, double bg) {
    const c10::cuda::CUDAGuard device_guard(means.device());
    auto gm = torch::zeros_like(means), gq = torch::zeros_like(conics);
    auto gc = torch::zeros_like(colors), go = torch::zeros_like(opacities);
    int pixels = int(height * width), threads = 256;
    backward_kernel<<<(pixels+threads-1)/threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), colors.data_ptr<float>(),
        opacities.data_ptr<float>(), rendered.data_ptr<float>(), grad_output.data_ptr<float>(),
        gm.data_ptr<float>(), gq.data_ptr<float>(), gc.data_ptr<float>(), go.data_ptr<float>(),
        int(means.size(0)), int(width), pixels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gm, gq, gc, go};
}
