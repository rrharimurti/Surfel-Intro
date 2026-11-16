from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="surfel_cuda",
    version="0.1.0",
    packages=["surfel_cuda"],
    ext_modules=[CUDAExtension(
        name="surfel_cuda._C",
        sources=["surfel_cuda/binding.cpp", "surfel_cuda/rasterize.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
