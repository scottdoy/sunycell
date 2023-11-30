# SUNYCell Software Resources

This repository is designed to maintain software for the SUNYCell project. 
For information on this software, including the SUNYCell biomedical image data repository, please contact [Scott Doyle](mailto:scottdoy@buffalo.edu).

## Installation

It is highly recommended to use a virtual environment (e.g. `conda` or `virtualenv`) to install this software.

To install this package into your current environment, from the repository directory, run:

```
pip install -e .
```

This will install the `sunycell` package into your current environment. 

The main dependency of the software is [histomicstk](https://github.com/DigitalSlideArchive/HistomicsTK).
There are some other utilities required as well, like `python-dotenv` for loading your secrets / API keys. 
See the `src/setup.py` for details.

## Running Examples

In order to access files on the SUNYCell DSA, you'll need to create a `.env` file somewhere in your project directory. 
This is a plaintext file that contains your secrets; in particular, the `*_URL` and `*_KEY` environment variables for different API endpoints.
Make sure that if you are running code locally, that these files are accessible.
DO NOT add them to your GitHub account!

## Example Notebooks

We will be expanding this section with demonstration / educational resources for different parts of the SUNYCell ecosystem.

- [Stain Normalization Examples](notebooks/stain_normalization.ipynb)
