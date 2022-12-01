from setuptools import setup

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
    ],
    include_package_data=True,
    packages=["ikflow"],
    package_data={"": ["model_descriptions.yaml"]},
)
