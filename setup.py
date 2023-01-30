import toml
from setuptools import find_namespace_packages, setup

cfg = toml.load("./pyproject.toml")

name = cfg["project"]["name"]

setup(
    packages=find_namespace_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={name: ["py.typed", "*.pyi", "*/*.pyi"]},
    cmake_install_dir=name)
