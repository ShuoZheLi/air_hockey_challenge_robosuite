# read the contents of your README file
from os import path

from setuptools import find_packages, setup


setup(
    name="robosuite",
    packages=[package for package in find_packages() if package.startswith("robosuite")],
    install_requires=[
        "numpy>=1.13.3",
        "numba>=0.49.1",
        "scipy>=1.2.3",
        "mujoco>=2.3.0",
        "Pillow",
        "opencv-python",
        "pynput",
        "termcolor",
    ],
)
