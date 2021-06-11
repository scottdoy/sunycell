"""
This package handles things related to tissue stains, like tissue detection and stain normalization.
"""
import numpy as np

from skimage import segmentation, color
from skimage.morphology import disk, opening, remove_small_objects
from skimage.measure import label

from scipy import ndimage as ndi

from histomicstk.saliency.tissue_detection import (
    get_tissue_mask)

def histogram_matching(source_image, target_image, nbins = 255):
    """Perform stain normalization through RGB histogram matching.
    """
    # Ensure that the source and target images are numpy arrays
    source_image_np = np.array(source_image)
    target_image_np = np.array(target_image)

    # Make a copy of the source to contain the matched image
    source_matched = source_image_np.copy()

    # Normalize along each color channel separately
    for channel_idx in range(source_image_np.shape[2]):
        
        src_hist, src_bins = np.histogram(source_image_np[:,:,channel_idx].flatten(), nbins, density=True)
        tar_hist, tar_bins = np.histogram(target_image_np[:,:,channel_idx].flatten(), nbins, density=True)

        # Source image normalization
        # Calculate the cumulative distribution function, then normalize
        cdf_source = src_hist.cumsum()
        cdf_source = (255 * cdf_source / cdf_source[-1]).astype(np.uint8)

        # Target image normalization
        # Calculate the cumulative distribution function, then normalize
        cdf_target = tar_hist.cumsum() 
        cdf_target = (255 * cdf_target / cdf_target[-1]).astype(np.uint8)

        # Interpolate 
        interp1 = np.interp(source_image_np[:,:,channel_idx].flatten(), src_bins[:-1], cdf_source)
        interp2 = np.interp(interp1, cdf_target, tar_bins[:-1])

        source_matched[:,:,channel_idx] = interp2.reshape((source_image_np.shape[0],source_image_np.shape[1]))
    
    return source_matched

def reinhard_matching(source_image, target_image):
    """Perform stain normalization through RGB histogram matching.

    Reinhard, E., et al. "Color transfer between images." IEEE Computer Graphics and Applications, vol. 21(5), 2001, pp. 34-41.
    """
    # Ensure that the source and target images are numpy arrays
    source_image_np = np.array(source_image)
    target_image_np = np.array(target_image)

    source_lab = color.rgb2lab(source_image_np)
    target_lab = color.rgb2lab(target_image_np)

    reinhard_norm = source_lab.copy()

    for channel_idx in range(source_lab.shape[2]):
        # Calculate channel stats for source and target
        source_mean = np.mean(source_lab[:,:,channel_idx].flatten())
        source_std = np.std(source_lab[:,:,channel_idx].flatten())
        
        target_mean = np.mean(target_lab[:,:,channel_idx].flatten())
        target_std = np.std(target_lab[:,:,channel_idx].flatten())
        
        # Compute the output channel
        reinhard_norm[:,:,channel_idx] = ((source_lab[:,:,channel_idx] - source_mean) * (target_std / source_std)) + target_mean

    # Convert back to rgb space for display
    # RGB values are floats between 0.0 and 1.0, so scale them back up
    reinhard_rgb = (color.lab2rgb(reinhard_norm) * 255).astype(np.uint8)

    return reinhard_rgb

def macenko_matching(source_image, target_image):
    """Perform stain normalization through RGB histogram matching.

    Macenko stain normalization:
    Macenko, M., et al. "A Method for Normalizing Histology Slides for Quantitative Analysis". IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250
    http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    """
    # Ensure that the source and target images are numpy arrays
    source_image_np = np.array(source_image)
    target_image_np = np.array(target_image)

    # Estimate the stain and sparse OD for both source and target
    source_stain, source_od_sparse = macenko_estimate_stain(source_image_np)
    target_stain, target_od_sparse = macenko_estimate_stain(target_image_np)

    # Deconvolve the source and target images with the estimated stain vector
    source_deconvolved, source_matrix = color_deconvolve(source_image_np, matrix=source_stain)
    target_deconvolved, target_matrix = color_deconvolve(target_image_np, matrix=target_stain)

    # Vectorize the deconvolved images
    source_deconvolved = np.reshape(source_deconvolved, (-1, 3))
    target_deconvolved = np.reshape(target_deconvolved, (-1, 3))

    # Get the top 99% of the images
    max_source = np.percentile(source_deconvolved, 99, axis=0);   
    max_target = np.percentile(target_deconvolved, 99, axis=0);   

    source_c = source_deconvolved / max_source
    source_c = source_c * max_target

    # Reconstruct the RGB image 
    macenko_norm = 255*np.exp(np.matmul(source_c, -target_matrix))
    macenko_norm = np.reshape(macenko_norm, (source_image_np.shape[0], source_image_np.shape[1], 3)).astype('uint8')
    return macenko_norm

def _random_patches(I, num_patches=30, patch_size=256):
    """Calculate a set of random patches from the input image I."""
    
    patches = []
    for patch_idx in np.arange(num_patches):
        x_coordinate = np.random.randint(patch_size//2+1, I.shape[0]-patch_size//2-1)
        y_coordinate = np.random.randint(patch_size//2+1, I.shape[1]-patch_size//2-1)
        
        patch = I[x_coordinate-patch_size//2:x_coordinate+patch_size//2,y_coordinate-patch_size//2:y_coordinate+patch_size//2,:]
        patches.append(patch)
    return patches

def _optical_density(input_data, intensity):
    return -np.log((input_data)/intensity)

def macenko_estimate_stain(I, light=255.0, beta=0.15, alpha=1):
    """Estimate the stain separation matrix using Macenko's method."""
    
    # Create a set of randomly-sampled patches from the image to reduce the amount of data to process
    I_patches = _random_patches(I, num_patches=30, patch_size=256)
    I_patches = np.reshape(np.stack(I).astype('double'), (-1, 3))
    
    od = _optical_density(I_patches+1, light)
    
    # Get rid of any pixels with OD < beta in any channel
    # This is similar to tissue detection -- any "non-dense" tissue is removed
    idx_bool = np.any(od < beta, 1)
    od_sparse = od[~idx_bool, :]
    
    # This is the step that takes a lot of memory
    _, eigenvectors = np.linalg.eigh(np.cov(od_sparse, rowvar=False))

    eigenvectors = eigenvectors[:, [2, 1]]
    if eigenvectors[0, 0] < 0: eigenvectors[:, 0] *= -1
    if eigenvectors[0, 1] < 0: eigenvectors[:, 1] *= -1
    T_hat = np.dot(od_sparse, eigenvectors)
    
    phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100-alpha)

    v1 = np.dot(eigenvectors, np.array([np.cos(min_phi), np.sin(min_phi)]))
    v2 = np.dot(eigenvectors, np.array([np.cos(max_phi), np.sin(max_phi)]))
    
    if v1[0] > v2[0]:
        stainvectors = np.array([v1, v2])
    else:
        stainvectors = np.array([v2, v1])

    return stainvectors, od_sparse

def color_deconvolve(input_image, intensity=255.0, matrix=np.array([[0.644211, 0.716556, 0.266844], [0.092789, 0.954111, 0.283111]])):
    """Perform color deconvolution of the given image with the given matrix."""
    
    I = np.array(input_image).astype('float')
    assert I.shape[2] >= 3, f"Input image must have 3 channels (RGB)."

    if I.shape[2] > 3:
        I = I[:,:,0:3]
    
    # If the stain matrix only has two vectors, add a third as the cross-product of the other two
    if matrix.shape[0] < 3:
        matrix = np.vstack((matrix, np.cross(matrix[0,:], matrix[1,:])))
        
    # Normalise to a unit vector
    matrix = matrix / np.sqrt(np.sum(matrix**2))

    # Vectorize
    I = np.reshape(I, (-1,3))

    # Optical Density
    od = _optical_density(I+1, intensity)
    od = np.reshape(od, (-1, 3))

    # Get stain concentrations
    # M is 3 x 3,  Y is N x 3, C is N x 3
    c = np.dot(od, np.linalg.pinv(matrix))
    
    # Reshape output
    deconvolved_image = np.reshape(c, (input_image.shape[0], input_image.shape[1], 3))
    return deconvolved_image, matrix

def get_tissue_boundaries(tissue_image, tissue_mask_kwargs=None):
    """Given an RGB image, detect tissue boundaries and return them."""

    if tissue_mask_kwargs is None:
        tissue_mask_kwargs = {
            'deconvolve_first': False,
            'n_thresholding_steps': 1, 
            'sigma': 0.4,
            'min_size': 5
        }

    # Use the HistomicsTK tissue masking function
    labeled, mask = get_tissue_mask(
        tissue_image,
        **tissue_mask_kwargs)

    #Erosion, Opening(instead of erosion)
    selem = disk(1)
    mask_proc = opening(mask, selem)

    # Remove small objects in the opened label image
    mask_proc = remove_small_objects(mask_proc, min_size = 5000)
    
    # Fill holes
    mask_proc = ndi.binary_fill_holes(mask_proc>0)

    # Re-labeled processed image mask
    labeled_proc = label(mask_proc, background = 0)
    
    # Grab the x and y coordinates of the tissue regions
    img_tissue_boundary = segmentation.find_boundaries(mask_proc)
    (tissue_y, tissue_x) = np.nonzero(img_tissue_boundary)
    return tissue_x, tissue_y