import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from STP_processing import *
from colormaps import *
from scipy.ndimage import gaussian_filter # for applying gaussian filter for density plots
# from make_masks import areas

# areas to generate masks for
areas = ["grey", "CTX", "OMCc", "ACAc", "aud","TH", "STR", "CP", "AMY", "P", "PG", "MB", "PAG", "SCm", 
         "SNr", "HY", "CNU", "TEa", "ECT", "VISC", "AI", "GU", "BS"]


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


def plot_contour(omc_image, acc_image, mask_list, masks_to_plot, roi, 
                 mask_list_order=areas, view="front", species="STeg"):
    """Plot contour map of max projection of roi

    Args:
        omc_image (np.array): 3D STP images for OMC injection
        acc_image (np.array): 3D STP images for ACC injection
        mask_list (list): list of masks
        masks_to_plot (list): list of strings specifying the areas to plot in outline, and the order the areas should be laid
        roi (str): Region of interest to apply mask and plot max projection
        view (str, optional): what view, can be 'front', 'side', or 'top'. Defaults to "front".
        species (str, optional): Specify species so get correct color for outline. Defaults to "STeg".
    """
    # set prarmeters
    if view=="front":
        ar = 1
        transform = (0,1,2)
    elif view=="side":
        ar = 1/2.5
        transform = (2,1,0)
    elif view=="top":
        ar=2.5
        transform = (1,0,2)
    
    if species=="STeg":
        sp_cmp = orange_cmp
    elif species=="MMus":
        sp_cmp = blue_cmp

    # transform/rotate data
    omc_image = np.transpose(omc_image, transform)
    acc_image = np.transpose(acc_image, transform)
    mask_list = [np.transpose(array, transform) for array in mask_list]

    # create outline of max project slice
    outline = make_boundaries(plot_areas=masks_to_plot, mask_list=mask_list, roi=roi)
    
    # slice outline
    roi_index = areas.index(roi)
    roi_mask = mask_list[roi_index]

    fig, axs = plt.subplots()

    slice_to_contour(omc_image, roi_mask, cmap=green_cmp)
    slice_to_contour(acc_image, roi_mask, cmap=purple_cmp)
    axs.set_aspect(ar)
    axs.axis('off')
    plt.imshow(outline, cmap=sp_cmp, aspect=ar)

    return(fig)

def dot_bar_plot(df, title="", xaxis="Area", yaxis="Integrated Fluorescence", hueaxis="Species",
                 errorbar="se"):
    """
    Function to take pandas dataframe and plot individual values and mean/sem values
    Intent to use for plotting nodes by frequency (in fraction of neurons)

    Args:
        df (pandas.core.frame.DataFrame): pandas dataframe where rows are nodes and columns are:
         'Node Degreee', 'Normalized Frequency', 'Species', and 'mouse'
         - See output of df_to_nodes
        title (str): plot title
    """
    fig = plt.subplot()
    sns.stripplot(df, x=xaxis, y=yaxis, hue=hueaxis, dodge=True, jitter=False, size=3)
    t_ax = sns.barplot(df, x=xaxis, y=yaxis, hue=hueaxis, errorbar=errorbar, errwidth=1)
    for patch in t_ax.patches:
        clr = patch.get_facecolor()
        patch.set_edgecolor(clr)
        patch.set_facecolor((0,0,0,0))
    plt.setp(t_ax.patches, linewidth=1)
    plt.title(title, size=18)

    return(fig)

def stvmm_area_scatter(data, title="", to_plot="Fluorescence"):
    """Plots lab mouse v. singing moues scatter w/ unity line

    Args:
        data (pandas.dataframe): output of calc_fluor
        to_plot (str, optional): Label of column in data to plot. Defaults to "Fluorescence".
    """

    # separate by species
    data_st = data[data["species"]=="STeg"]
    data_mm = data[data["species"]=="MMus"]

    # calculate means
    st_mean = data_st.groupby("area").mean(numeric_only=True)
    st_se = data_st.groupby("area").sem(numeric_only=True)

    mm_mean = data_mm.groupby("area").mean(numeric_only=True)
    mm_se = data_mm.groupby("area").sem(numeric_only=True)

    fig = plt.subplot()

    plt.errorbar(st_mean[to_plot], mm_mean[to_plot], 
            xerr=st_se[to_plot], fmt='|', color="orange")
    plt.errorbar(st_mean[to_plot], mm_mean[to_plot], 
            yerr=mm_se[to_plot], fmt='|')

    # add area labels
    labels = list(st_mean.index)
    for i in range(len(labels)):
        plt.annotate(labels[i], (st_mean[to_plot][i], mm_mean[to_plot][i]))
    
    # adjust scale
    plt.xscale("log")
    plt.yscale("log")

    # plot unity line
    x = np.linspace(0,20000, 5)
    y = x
    plt.plot(x, y, color='red', linestyle="--", linewidth=0.5)

    # add axis labels
    plt.xlabel("Singing Mouse Integrated Fluorescence", color="tab:orange")
    plt.ylabel("Lab Mouse Integrated Fluorescence", color="tab:blue")

    # add title
    plt.title(title)

    return(fig)
