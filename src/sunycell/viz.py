"""
This package handles data visualization and plotting.
We use some common techniques for displaying images, e.g. in matplotlib, plotly, and bokeh. 
This package will combine most of the default settings and themes in one place so they can be easily referenced by multiple users.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_image_stack(image_stack, ncols=3, figsize=(20,10), titles=None):
    """Plot a stack of images using nice defaults."""

    # Images should be in h x w x c x d where d is the number of images.
    image_stack = np.array(image_stack)

    # TODO: Process image to an appropriate range

    # Get statistics of the image
    h, w, c, d = np.shape(image_stack)

    # Calculate the rows and columns based on provided ncols
    nrows = d // ncols

    # If we have an extra row, add one
    nrows = nrows + 1 if d % ncols > 0 else nrows

    # Create a basic plot
    f, ax = plt.subplots(nrows, ncols, figsize=figsize)

    # Switch based on nrows, since nrows=1 gives a single axis list
    if nrows == 1:
        for ncol in np.arange(ncols):
            idx = ncol
            if idx <= (d-1):
                ax[ncol].imshow(image_stack[:,:,:,idx])
                ax[ncol].axis('off')
                if titles is not None:
                    ax[ncol].set_title(f'{titles[idx]}')
    else:
        for nrow in np.arange(nrows):
            for ncol in np.arange(ncols):
                idx = (nrow*ncols) + ncol
                if idx <= (d-1):
                    ax[nrow][ncol].imshow(image_stack[:,:,:,idx])
                    ax[nrow][ncol].axis('off')
                    if titles is not None:
                        ax[nrow][ncol].set_title(f'{titles[idx]}')
    
    plt.tight_layout()
    plt.show()

def plot_image(image, figsize=(10,10), title=None):
    """Plot a basic image using nice defaults."""

    # Ensure image is numpy
    image = np.array(image)

    # TODO: Process image to an appropriate range

    # Create a basic plot
    f, ax = plt.subplots(figsize=figsize)

    ax.imshow(image)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

