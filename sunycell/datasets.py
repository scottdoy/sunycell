"""
This package handles sample datasets.
This includes both demo data and production data, if it exists.
Note that "production" data will consist of curated data samples that we will retrieve from a public source.
"""

import pooch
import sunycell

from PIL import Image

version = sunycell.__version__

# Create a new friend to manage your sample data storage
GOODBOY = pooch.create(
    # Folder where the data will be stored. For a sensible default, use the
    # default cache folder for your OS.
    path=pooch.os_cache("sunycell"),
    # Base URL of the remote data store. Will call .format on this string
    # to insert the version (see below).
    base_url="https://github.com/scottdoy/sunycell/raw/main/data/",
    # Pooches are versioned so that you can use multiple versions of a
    # package simultaneously. Use PEP440 compliant version number. The
    # version will be appended to the path.
    version=version,
    # If a version as a "+XX.XXXXX" suffix, we'll assume that this is a dev
    # version and replace the version with this string.
    version_dev="main",
    # An environment variable that overwrites the path.
    env="SUNYCELL_DATA_DIR",
    # The cache file registry. A dictionary with all files managed by this
    # pooch. Keys are the file names (relative to *base_url*) and values
    # are their respective SHA256 hashes. Files will be downloaded
    # automatically when needed (see fetch_gravity_data).
    registry={
        "he/008a.png": "sha512:8689d2e0a54e1730c4eca9a95db73f3a9066b31fded6942de69dcc8f7a7ee7d342edc62e9d146ecd5cceb1d40bc30886a6245e9704654d3a5a682b483a3922c6",
        "he/013a.png": "sha512:697060ae1edc819b39165cdebc32f92c988d2920dc56f93ed4fa2cef911da33ed69361b6be23c7d1135d67f5dd6a233ea7e8384d18fc7f0f1e6623a0b7bedac2",
    }
)


# Define functions that your users can call to get back the data in memory
def stainnorm_img_pair():
    """
    Load some sample tissue images for stain normalization
    """
    # Fetch the path to a file in the local storage. If it's not there,
    # we'll download it.
    target_fname = GOODBOY.fetch("he/008a.png")
    source_fname = GOODBOY.fetch("he/013a.png")

    # Load it with numpy/pandas/etc
    target_image = Image.open(target_fname)
    source_image = Image.open(source_fname)
    return target_image, source_image