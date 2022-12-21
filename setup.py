from setuptools import setup
import os
import pathlib


def package_files(directory: str, ignore_ext: list = []) -> list:
    """Returns the filepath for all files in a directory. Borrowed from https://stackoverflow.com/a/36693250"""
    paths = []
    ignore_ext = [ext.lower() for ext in ignore_ext]
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(path, filename)
            if pathlib.Path(filepath).suffix.lower().strip(".") in ignore_ext:
                continue
            paths.append(filepath)
    return paths


urdf_files = package_files("ikflow/visualization_resources/")
urdf_files = [
    fname.strip("ikflow/") for fname in urdf_files
]  # filenames are relative to the root directory, but we want them relative to the root/ikflow/ directory
assert len(urdf_files) > 0, "No URDF files found"


setup(
    name="ikflow",
    version="0.0.0",
    author="Jeremy Morgan",
    author_email="jsmorgan6@gmail.com",
    scripts=[],
    url="https://github.com/jstmn/ikflow",
    license="LICENSE.txt",
    description="Open source implementation of the 'IKFlow' inverse kinematics solver",
    long_description=open("README.md").read(),
    install_requires=[
        "kinpy",
        "klampt",
        "torch",
        "pytorch-lightning",
        "FrEIA",
        "tensorboard",
        "wandb",
        "black",
        "jkinpylib==0.0.4",
        "more_itertools",
    ],
    include_package_data=True,
    packages=["ikflow"],
    package_data={"ikflow": ["model_descriptions.yaml"] + urdf_files},  # TODO: Add ikflow/visualization_resources/
)
