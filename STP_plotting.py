import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from STP_processing import *
from colormaps import *

def slice_to_contour(stp_image, mask, slice_range=None, slice=None, output="contour", gs=3, ar=1, cmap=green_cmp,
                     alpha=0.75):
    """_summary_

    Args:
        stp_image (np.array): 3d numpy array with intensity values of STP brain.
        mask (np.arrray): Mask for desired maximum area
        slice_range (tuple?, optional): Give slices to make maximum. Defaults to None.
        slice (int, optional): if given, will only output slice given by index. Defaults to None.
        output (str, optional): Can be 'maximum', 'guassian', or 'contour', will determine the output image.
                                Defaults to 'contour'
        gs (int, optional): Sigma to be used for gaussian blurring. Defaults to 3.
        ar (int, optional): aspect ratio for plot. Defaults to 1.
    """

    # apply mask to stp_image
    stp_masked = np.multiply(stp_image, mask)

    # apply ranges if given
    if slice_range:
        stp_masked = stp_masked[slice_range[0]:slice_range[1]]
    
    # get maximum
    maximum = stp_masked.max(axis=0)

    if slice:
        maximum = stp_image[slice]

    # apply gaussian filter
    blur = gaussian_filter(maximum, sigma=gs)


    if output=="maximum":
        return(maximum)
    elif output=="gaussian":
        return(blur)
    
    # else return controur plot
    contour = plt.contour(blur, cmap=cmap, alpha=alpha)
    # ax.set_aspect(ar)
    # ax.axis('off')

    return(contour)
    