from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="testing",
    ext_modules=cythonize("skbio/alignment/align.pyx"),
    include_dirs=[numpy.get_include()],
)
