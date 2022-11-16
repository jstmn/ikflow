from setuptools import setup

setup(
    name="ikflow",
    version="0.0.0",
    author="Jeremy Morgan",
    author_email="jsmorgan6@gmail.com",
    packages=[],
    scripts=[],
    url="http://pypi.python.org/pypi/ikflow/",
    license="LICENSE.txt",
    description="An awesome package that does something",
    long_description=open("README.md").read(),
    install_requires=["kinpy>=0.1.0", "klampt>=0.9.0", "FrEIA>=0.2", "confuse"],
    include_package_data=True,
    package_data={"": ["model_descriptions.yaml"]},
    # package_data={'': ['data/*.csv']},
)
