import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5 8.0"

setup(
    name="synthetic_scene",
    version="0.1.0",
    packages=["synthetic_scene"],
    ext_modules=[
        CUDAExtension(
            name="synthetic_scene._cuda_renderer",
            sources=[
                "synthetic_scene/csrc/bindings.cpp",
                "synthetic_scene/csrc/render_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
