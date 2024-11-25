from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mycuda',
    ext_modules=[
        CUDAExtension('mycuda', ['./backward.cu',],
                      extra_compile_args={'cxx': [], 'nvcc': [
                          "-O3",
                          "-DCUDA_HAS_FP16=1",
                          "-D__CUDA_NO_HALF_OPERATORS__",
                          "-D__CUDA_NO_HALF_CONVERSIONS__",
                          "-D__CUDA_NO_HALF2_OPERATORS__",
                      ]}
                      ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)