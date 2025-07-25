from setuptools import setup, find_packages

setup(
    name="SSL_ECG",
    version="0.1",
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
)