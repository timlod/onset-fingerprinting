import setuptools

import numpy

# Setup just to build the numpy extension
setuptools.setup(
    ext_modules=[
        setuptools.Extension(
            "online_cc",
            sources=["onset_fingerprinting/c/cross_corr.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=[
                "-Wall",
                "-O3",
                "-mavx2",  # or -msse
            ],
        ),
        setuptools.Extension(
            "online_cc",
            sources=["onset_fingerprinting/envelope_follower.c"],
            extra_compile_args=[
                "-Wall",
                "-O3",
            ],
        ),
    ]
)
