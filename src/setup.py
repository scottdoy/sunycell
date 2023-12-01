from setuptools import setup, find_packages

setup(
    name="sunycell", 
    packages=find_packages(),
    description="Toolkit for working with the SUNYCell image repository.",
    license='Apache Software License 2.0',
    version='0.1.0',
    install_requires=[
        'histomicstk',
        'python-dotenv',
        'pooch',
        'rasterio'
    ],
)

