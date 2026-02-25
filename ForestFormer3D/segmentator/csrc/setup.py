from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='libsegmentator',
    ext_modules=[
        CppExtension(
            name='libsegmentator',
            sources=['segmentator.cpp'],
            extra_compile_args=['-O3'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })