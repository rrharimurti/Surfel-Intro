#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// All surfels must be sorted near-to-far, as done by the Python wrapper.
// Every CUDA thread owns a pixel; no dense [N,H,W] tensor is constructed.
// "depth" is in normalized camera-depth units (z / detached depth_scale).
// The depth distortion is sum_{i<j} w_i*w_j*(z_i-z_j)^2.
// The Gaussian footprint and early-termination logic match the original renderer.
__global__ void render_kernel(const float* __restrict__ means,
                              const float* __restrict__ conics,
                              const float* __restrict__ depths,
                              const float* __restrict__ colors,
                              const float* __restrict__ opacities,
                              float* __restrict__ output,
                              float* __restrict__ distortion,
                              float* __restrict__ moments,
                              int n, int width, int pixels, float bg) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float x = float(p % width), y = float(p / width);
    float r = 0.f, g = 0.f, b = 0.f, T = 1.f;
    float W = 0.f, M1 = 0.f, M2 = 0.f, D = 0.f;
    for (int i = 0; i < n; ++i) {
        float dx = x - means[2*i], dy = y - means[2*i+1];
        float a = conics[3*i], bb = conics[3*i+1], c = conics[3*i+2];
        float q = a*dx*dx + 2.f*bb*dx*dy + c*dy*dy;
        if (q < 0.f || q > 18.f) continue;
        float alpha = fminf(fmaxf(opacities[i] * __expf(-0.5f*q), 0.f), 0.999f);
        float w = T * alpha;
        float z = depths[i];
        r += w*colors[3*i]; g += w*colors[3*i+1]; b += w*colors[3*i+2];
        // Incremental, numerically more stable form of W*M2 - M1*M1.
        D += w * (W*z*z + M2 - 2.f*z*M1);
        W += w;
        M1 += w*z;
        M2 += w*z*z;
        T *= 1.f - alpha;
        if (T < 1e-5f) break;
    }
    output[3*p] = r + T*bg;
    output[3*p+1] = g + T*bg;
    output[3*p+2] = b + T*bg;
    distortion[p] = D;
    moments[3*p] = W;
    moments[3*p+1] = M1;
    moments[3*p+2] = M2;
}

// dD/dw_i = M2 + W*z_i^2 - 2*M1*z_i;
// dD/dz_i = 2*w_i*(W*z_i - M1).
// w_j for j>i depends on alpha_i via transmittance. Its gradient is included
// using the analytic tail sum, with O(1) work per contributing surfel.
__global__ void backward_kernel(const float* __restrict__ means,
                                const float* __restrict__ conics,
                                const float* __restrict__ depths,
                                const float* __restrict__ colors,
                                const float* __restrict__ opacities,
                                const float* __restrict__ rendered,
                                const float* __restrict__ moments,
                                const float* __restrict__ grad_rgb,
                                const float* __restrict__ grad_dist,
                                float* __restrict__ gm,
                                float* __restrict__ gq,
                                float* __restrict__ gz,
                                float* __restrict__ gc,
                                float* __restrict__ go,
                                int n, int width, int pixels) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float x = float(p % width), y = float(p / width);
    float prefix_r = 0.f, prefix_g = 0.f, prefix_b = 0.f;
    float T = 1.f;
    float W = moments[3*p], M1 = moments[3*p+1], M2 = moments[3*p+2];
    float prefix_W = 0.f, prefix_M1 = 0.f, prefix_M2 = 0.f;
    float gr = grad_rgb[3*p], gg = grad_rgb[3*p+1], gb = grad_rgb[3*p+2];
    float gd = grad_dist[p];
    for (int i = 0; i < n; ++i) {
        float dx = x - means[2*i], dy = y - means[2*i+1];
        float a = conics[3*i], bb = conics[3*i+1], c = conics[3*i+2];
        float q = a*dx*dx + 2.f*bb*dx*dy + c*dy*dy;
        if (q < 0.f || q > 18.f) continue;
        float weight = __expf(-0.5f*q);
        float raw_alpha = opacities[i]*weight;
        float alpha = fminf(fmaxf(raw_alpha, 0.f), 0.999f);
        float w = T*alpha;
        float z = depths[i];
        float cr = colors[3*i], cg = colors[3*i+1], cb = colors[3*i+2];
        float tail_r = rendered[3*p] - prefix_r - w*cr;
        float tail_g = rendered[3*p+1] - prefix_g - w*cg;
        float tail_b = rendered[3*p+2] - prefix_b - w*cb;
        float da_rgb = gr*(T*cr - tail_r/(1.f-alpha))
                     + gg*(T*cg - tail_g/(1.f-alpha))
                     + gb*(T*cb - tail_b/(1.f-alpha));
        atomicAdd(&gc[3*i], gr*w);
        atomicAdd(&gc[3*i+1], gg*w);
        atomicAdd(&gc[3*i+2], gb*w);

        // Accumulate moments INCLUDING this surfel to recover the remaining tail.
        float next_W = prefix_W + w;
        float next_M1 = prefix_M1 + w*z;
        float next_M2 = prefix_M2 + w*z*z;
        float dw = M2 + W*z*z - 2.f*M1*z;
        float tail_dw = M2*(W-next_W) + W*(M2-next_M2)
                      - 2.f*M1*(M1-next_M1);
        float da_dist = T*dw - tail_dw/(1.f-alpha);
        // Depth has a direct derivative; sorting indices remain non-differentiable.
        atomicAdd(&gz[i], gd * (2.f*w*(W*z - M1)));
        float da = da_rgb + gd*da_dist;

        // Gaussian/opacity derivatives match the original implementation.
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
        prefix_W = next_W; prefix_M1 = next_M1; prefix_M2 = next_M2;
        T *= 1.f-alpha;
        if (T < 1e-5f) break;
    }
}

std::vector<torch::Tensor> raster_forward_cuda(
    torch::Tensor means, torch::Tensor conics, torch::Tensor depths,
    torch::Tensor colors, torch::Tensor opacities,
    int64_t height, int64_t width, double bg) {
    const c10::cuda::CUDAGuard device_guard(means.device());
    auto out = torch::empty({height, width, 3}, means.options());
    auto dist = torch::empty({height, width}, means.options());
    auto moments = torch::empty({height, width, 3}, means.options());
    int pixels = int(height*width), threads = 256;
    render_kernel<<<(pixels+threads-1)/threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), depths.data_ptr<float>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(), out.data_ptr<float>(),
        dist.data_ptr<float>(), moments.data_ptr<float>(), int(means.size(0)),
        int(width), pixels, float(bg));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, dist, moments};
}

std::vector<torch::Tensor> raster_backward_cuda(
    torch::Tensor means, torch::Tensor conics, torch::Tensor depths,
    torch::Tensor colors, torch::Tensor opacities,
    torch::Tensor rendered, torch::Tensor moments,
    torch::Tensor grad_rgb, torch::Tensor grad_dist,
    int64_t height, int64_t width, double bg) {
    const c10::cuda::CUDAGuard device_guard(means.device());
    auto gm = torch::zeros_like(means), gq = torch::zeros_like(conics);
    auto gz = torch::zeros_like(depths), gc = torch::zeros_like(colors);
    auto go = torch::zeros_like(opacities);
    int pixels = int(height*width), threads = 256;
    backward_kernel<<<(pixels+threads-1)/threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), depths.data_ptr<float>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(), rendered.data_ptr<float>(),
        moments.data_ptr<float>(), grad_rgb.data_ptr<float>(), grad_dist.data_ptr<float>(),
        gm.data_ptr<float>(), gq.data_ptr<float>(), gz.data_ptr<float>(),
        gc.data_ptr<float>(), go.data_ptr<float>(), int(means.size(0)),
        int(width), pixels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gm, gq, gz, gc, go};
}
