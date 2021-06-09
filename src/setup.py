from setuptools import setup, find_packages

setup(
    name="sunycell", 
    description="Toolkit for working with the SUNYCell image repository.",
    license='Apache Software License 2.0',
    version='0.1a1',
    install_requires=[
        'histomicstk',
        'python-dotenv',
        'pooch'
    ]
)

