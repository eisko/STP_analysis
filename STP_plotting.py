import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from STP_processing import *
from colormaps import *
from scipy.ndimage import gaussian_filter # for applying gaussian filter for density plots
import math # needed for sqrt for ci95
import copy # needed to deepcopy dictionary
from matplotlib.lines import Line2D # for custom legend

# from make_masks import areas

# areas to generate masks for
areas = ["grey", "CTX", "OMCc", "ACAc", "aud","TH", "STR", "CP", "AMY", "P", "PG", "MB", "PAG", "SCm", 
         "SNr", "HY", "CNU", "TEa", "ECT", "VISC", "AI", "GU", "BS"]


def slice_to_contour(stp_image, mask, slice_range=None, slice=None, output="contour", gs=3, cmap=green_cmp,
                     alpha=0.75, ncontours=8, linewidths=1, fill=False):
    """_summary_

    Args:
        stp_image (np.array): 3d numpy array with intensity values of STP brain.
        mask (np.arrray): Mask for desired maximum area
        slice_range (tuple?, optional): Give slices to make maximum. Defaults to None.
        slice (int, optional): if given, will only output slice given by index. Defaults to None.
        output (str, optional): Can be 'maximum', 'guassian', or 'contour', will determine the output image.
                                Defaults to 'contour'
        gs (int, optional): Sigma to be used for gaussian blurring. Defaults to 3.
        cmap (cmap, optional): Colormap to use to determine contour colors. Defaults to custum greeen_cmp.
        alpha (float, optional): Opacity value for contour lines. Defaults to 0.75.
        ncontours (int, optional): Number of contour lines to draw. Defaults to 8 (plt.contour default is 8).
        fill (bool, optional): Whether to plot outlines or filled contours. Defaults to False.
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
    if fill:
        contour = plt.contourf(blur, ncontours, cmap=cmap, 
                               alpha=alpha, linewidths=linewidths,
                               origin="image")
    else:
        contour = plt.contour(blur, ncontours, cmap=cmap, 
                              alpha=alpha, linewidths=linewidths,
                              origin="image")
    # ax.set_aspect(ar)
    # ax.axis('off')

    return(contour)


def plot_contour_omc_acc(omc_image, acc_image, mask_list, masks_to_plot, roi, 
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

def plot_contour(images, mask_dict, masks_to_plot, roi=None, 
                 view="front", cmaps=None, ncontours=8, alpha=0.75, gs=3, linewidths=1,
                 cmap_outline="gray_r", fig_size=None, plot_fill=False, ref_fill=False):
    """Plot contour map of max projection of up to 3 images

    Args:
        images (list): List of 3D STP images to be plotted
        mask_dict (dict): Dictionary of masks to use
        masks_to_plot (list): list of strings specifying the areas to plot in outline, and the order the areas should be laid
        roi (str, optional): Region of interest to apply mask and plot max projection. Defaults to None.
        view (str, optional): what view, can be 'front', 'side', or 'top'. Defaults to "front".
        cmaps (list, optional): list of cmaps to be used to distinguish images. Defaults to None.
        ncontours (int, optional): Number of contour lines to draw. Dafaults to 8 (plt.contour default).
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

    # transform/rotate data
    if images:
        im_tr = [np.transpose(im, transform) for im in images]

    # transform/rotate masks
    mask_tr = copy.deepcopy(mask_dict)
    for area in mask_tr:
        if type(mask_tr[area])==list:
            for i in range(len(mask_tr[area])):
                mask_tr[area][i] = np.transpose(mask_tr[area][i], transform)
        else:
            mask_tr[area] = np.transpose(mask_tr[area], transform)

    # roi/slice outline
    if roi:
        roi_mask = mask_tr[roi]
        if type(roi_mask)==list:
            roi_plot = roi_mask[0]
        else:
            roi_plot = roi_mask
    else:
        roi_mask = mask_tr['grey']
        roi_plot = roi_mask

    mask_tr['roi'] = roi_mask
    mask_tr['roi_plot'] = roi_plot


    # create outline of max project slice
    # outline = make_boundaries_dict(plot_areas=masks_to_plot, mask_dict=mask_tr, roi="roi_plot")
    outline, fill = make_boundaries_dict(plot_areas=masks_to_plot, mask_dict=mask_tr,
                                   scaling_factor=1, return_fill=True)


    fig, axs = plt.subplots()

    if fig_size:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])

    
    if cmaps:
        colors=cmaps
    else:
        colors = [blue_cmp, orange_cmp, green_cmp]

    if type(roi_mask)==list:
        for i in range(len(images)):
            slice_to_contour(im_tr[i], roi_mask[i], cmap=colors[i], ncontours=ncontours,
                             alpha=alpha, gs=gs, linewidths=linewidths, fill=plot_fill)
    elif images:
        for i in range(len(images)):
            slice_to_contour(im_tr[i], roi_mask, cmap=colors[i], ncontours=ncontours,
                             alpha=alpha, gs=gs, linewidths=linewidths, fill=plot_fill)

    axs.set_aspect(ar) # make sure proper aspect ratio is maintained
    axs.axis('off')

    # plot reference so that it's editable
    if ref_fill:
        plt.contourf(fill, origin="image", vmin=0, vmax=10, cmap=cmap_outline, alpha=1)
    else:
        plt.contour(outline, origin="image", vmin=0, vmax=0.1, cmap=cmap_outline, 
                    alpha=0.5, linewidths=0.5)

    
    return(fig)


def plot_contour_species(mm_image, st_image, mask_dict, plot_areas, roi,
                          view="front", alpha_mm=0.75, alpha_st=0.75, 
                          fig_size=None):
    """Plot contour map of max projection of roi of to compare aligned
    Singing and lab mouse

    Args:
        mm_image (np.array): 3D STP images for MMus
        st_image (np.array): 3D STP images for STeg
        mask_dict (dict): dictionary of aligned masks where keys are areas,
                            and values are masks
        plot_areas (list): list of strings specifying the areas to plot in outline, 
                            and the order the areas should be laid
        roi (str): Region of interest to apply mask and plot max projection
        view (str, optional): what view, can be 'front', 'side', or 'top'. Defaults to "front".
        species (str, optional): Specify species so get correct color for outline. Defaults to "STeg".
        fig_size (tuple, optional): Tuple of integers for (w,h) of figure in inches. Defautls to None.
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

    # transform/rotate data
    st_image = np.transpose(st_image, transform)
    mm_image = np.transpose(mm_image, transform)
    mask_transpose = copy.deepcopy(mask_dict)
    for mask in mask_transpose:
        mask_transpose[mask] = np.transpose(mask_dict[mask], transform)

    # create outline of max project slice
    outline = make_boundaries_dict(plot_areas=plot_areas, mask_dict=mask_transpose, roi=roi)
    
    # slice outline
    roi_mask = mask_transpose[roi]

    fig, axs = plt.subplots()

    # set figure size if specified
    if fig_size:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])

    slice_to_contour(mm_image, roi_mask, cmap=blue_cmp, alpha=alpha_mm)
    slice_to_contour(st_image, roi_mask, cmap=orange_cmp, alpha=alpha_st)
    axs.set_aspect(ar)
    axs.axis('off')
    plt.imshow(outline, cmap="Greys", aspect=ar)

    # return(fig)

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

def stvmm_area_scatter(data, title="", to_plot="Fluorescence", log=True, 
                       err="sem", ax_limits=None, xlabel="Singing Mouse Integrated Fluorescence",
                       ylabel="Lab Mouse Integrated Fluorescence"):
    """Plots lab mouse v. singing moues scatter w/ unity line

    Args:
        data (pandas.dataframe): output of calc_fluor
        to_plot (str, optional): Label of column in data to plot. Defaults to "Fluorescence".
    """

    # separate by species
    data_st = data[data["species"]=="STeg"]
    data_mm = data[data["species"]=="MMus"]

    # calculate stats
    st_stats = data_st.groupby(["area"])[to_plot].agg(['mean', 'count', 'std', 'sem'])
    mm_stats = data_mm.groupby(["area"])[to_plot].agg(['mean', 'count', 'std', 'sem'])

    ci95 = []
    for i in st_stats.index:
        m, c, sd, se = st_stats.loc[i]
        ci95.append(1.96*sd/math.sqrt(c))
    st_stats['ci95'] = ci95

    ci95 = []
    for i in mm_stats.index:
        m, c, sd, se = mm_stats.loc[i]
        ci95.append(1.96*sd/math.sqrt(c))
    mm_stats['ci95'] = ci95

    fig = plt.subplot()

    plt.errorbar(st_stats['mean'], mm_stats['mean'], 
            xerr=st_stats[err], fmt='|', color="orange")
    plt.errorbar(st_stats['mean'], mm_stats['mean'], 
            yerr=mm_stats[err], fmt='|')

    # add area labels
    labels = list(st_stats.index)
    for i in range(len(labels)):
        plt.annotate(labels[i], (st_stats['mean'][i], mm_stats['mean'][i]))
    
    # get rid of upper and right axes
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set x and y lims so that starts at 0,0
    if ax_limits:
        plt.xlim(ax_limits)
        plt.ylim(ax_limits)


    # adjust scale
    if log:
        plt.xscale("log")
        plt.yscale("log")

    # plot unity line
    x = np.linspace(0,20000, 5)
    y = x
    plt.plot(x, y, color='red', linestyle="--", linewidth=0.5)


    # add axis labels
    plt.xlabel(xlabel, color="tab:orange")
    plt.ylabel(ylabel, color="tab:blue")

    # add title
    plt.title(title)

    return(fig)

def area_scatter(data, title="", to_plot="Fluorescence", log=True, 
                       err="sem", ax_limits=None, compare_group="species",
                       xgroup="STeg", ygroup="MMus", xcolor="black",
                       ycolor="black"):
    """Plots by area, group1 mean vs. group 2 mean, with xy error bars

    Args:
        data (pandas.dataframe): output of calc_fluor
        to_plot (str, optional): Label of column in data to plot. Defaults to "Fluorescence".
        log (bool, optional): Determine whether axis on log sale or not. Defaults to True.
        err (str, optional): What to use as error bars, i.e. "sem", "ci95", "std". Defaults to "sem".
        ax_limits (tuple, optional): what axis limits to use. Defaults to None.
        compare_group (str, optional): what column to use as 2 comparison groups. Defaults to "species".
        xgroup (str, optional): group to plot on x axis. Defaults to "STeg".
        ygroup (str, optional): grou pto plot on y axis. Defaults to "MMus".
        xcolor (str, optional): color for x error bars. Defaults to 'orange'.
        ycolor (str, optional): color for y error bars. Defaults to None (which plots blue).
    """

    # separate by species
    data_x = data[data[compare_group]==xgroup]
    data_y = data[data[compare_group]==ygroup]

    # calculate stats
    x_stats = data_x.groupby(["area"])[to_plot].agg(['mean', 'count', 'std', 'sem'])
    y_stats = data_y.groupby(["area"])[to_plot].agg(['mean', 'count', 'std', 'sem'])

    ci95 = []
    for i in x_stats.index:
        m, c, sd, se = x_stats.loc[i]
        ci95.append(1.96*sd/math.sqrt(c))
    x_stats['ci95'] = ci95

    ci95 = []
    for i in y_stats.index:
        m, c, sd, se = y_stats.loc[i]
        ci95.append(1.96*sd/math.sqrt(c))
    y_stats['ci95'] = ci95

    fig = plt.subplot()

    plt.errorbar(x_stats['mean'], y_stats['mean'], 
            xerr=x_stats[err], fmt='|', color=xcolor)
    plt.errorbar(x_stats['mean'], y_stats['mean'], 
            yerr=y_stats[err], fmt='|', color=ycolor)

    # add area labels
    labels = list(x_stats.index)
    for i in range(len(labels)):
        plt.annotate(labels[i], (x_stats['mean'][i], y_stats['mean'][i]))
    
    # set x and y lims so that starts at 0,0
    if ax_limits:
        plt.xlim(ax_limits)
        plt.ylim(ax_limits)


    # adjust scale
    if log:
        plt.xscale("log")
        plt.yscale("log")

    # plot unity line
    x = np.linspace(0,20000, 5)
    y = x
    plt.plot(x, y, color='red', linestyle="--", linewidth=0.5)


    # add axis labels
    plt.xlabel(xgroup+" "+to_plot, color=xcolor)
    plt.ylabel(ygroup+" "+to_plot, color=ycolor)

    # add title
    plt.title(title)

    return(fig)

def volcano_plot(df, x="log2_fc", y="nlog10_p", title=None, labels="area", p_05=True, 
                 p_01=True, p_bf=None, xlim=None, ylim=None, fig_size=(4,4)):
    """output volcano plot based on comparison of species proportional means

    Args:
        df (pd.DataFrame): output of proprotion_ttest
        x (str, optional): name of column to go on x axis. Defaults to "log2_fc".
        y (str, optional): Column to plot on y. Defaults to "nlog10_p".
        title (str, optional): what to put as title. Defaults to None.
        labels (str, optional): Column to use to label each point. Defaults to "area".
        p_05 (bool, optional): Determines whether to plot line at p(0.05). Defaults to True.
        p_01 (bool, optional): Determines whether to plot line at p(0.01). Defaults to True.
        p_bf (int, optional): Takes integer used as berforroni correction threshold. Defaults to None.

    """

    # areas = sorted(df['area'].unique())

    fig = plt.subplot()

    x=df[x]
    y=df[y]

    plt.scatter(x,y, s=25)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    # plot 0 axes
    plt.axline((0, 0), (0, 1),linestyle='--', linewidth=0.5)
    plt.axline((0, 0), (1, 0),linestyle='--', linewidth=0.5)

    # p_05
    if p_05:
        plt.axline((0, -np.log10(0.05)), (1,  -np.log10(0.05)),linestyle='--', color='r', alpha=0.75, linewidth=0.5)
        plt.text(-0.1, -np.log10(0.05)+.015, 'p<0.05', color='r', alpha=0.75)
    if p_01:
        plt.axline((0, -np.log10(0.01)), (1,  -np.log10(0.01)),linestyle='--', color='r', alpha=0.5, linewidth=0.5)
        plt.text(-0.1, -np.log10(0.01)+.015, 'p<0.01', color='r', alpha=0.75)
    if p_bf:
        plt.axline((0, -np.log10(p_bf)), (1,  -np.log10(p_bf)),linestyle='--', color='r', alpha=0.75, linewidth=0.5)
        plt.text(-0.1, -np.log10(p_bf)+.015, 'p<bf_01', color='r', alpha=0.75)


    for i in range(df.shape[0]):
        plt.text(x=df.loc[i,"log2_fc"]+0.01,y=df.loc[i,"nlog10_p"]+0.01,s=df.loc[i, labels], 
            fontdict=dict(color='black',size=10))


    plt.title(title)
    plt.xlabel('$log_{2}$($\dfrac{STeg}{MMus}$)')
    plt.ylabel('$-log_{10}(p\ value)$')

    # get rid of upper and right axes
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size[0],fig_size[1])


    return(fig)

def dot_plot_by_species(data, area=None, title=None, err="se", add_legend=True,
                              to_plot="Fluorescence", ylabel="Fluorescence", ylim=(0), 
                              fig_size=(3.5,3.5)):
    """Plot individual values, mean, and error by species in dot plot.

    Args:
        df (DataFrame): Output from calc_fluor_individ()
        area (str): area to plot
        err (str): error bar to plot for sns.pointplot(), can be "ci", "pi", "se", or "sd". Defaults to "se".
        title (str): Title to apply to plot
        add_legend (bool, optional): Specify whether to include legend or not. Defaults to True.
        to_plot (str, optional): what column name to plot. Defaults to "Fluorescence".
        ylim (int, optional): Lower bound for yaxis. Defaults to (0).
    """

    if area:
        df = data[data["area"]==area]
    else:
        df = data.copy()

    # means = area_df.groupby('species')['proportion'].mean() # need means for plotting lines

    fig, ax = plt.subplots()

    sns.stripplot(data=df, x="species", y=to_plot, hue="species", size=12, ax=ax)
    # violin = sns.violinplot(area_df, x='species',y="proportion",
    #             split=True, hue ='species', inner = None, 
    #             palette="pastel",legend=False)
    pts = sns.pointplot(data=df, x="species", y=to_plot, hue="species", units='brain', 
                          color='black', markers='+', ax=ax, errorbar=err) # plots mean and 95 confidence interval:

    # necessary to put mean/error markers on top
    plt.setp(pts.lines, zorder=100)
    plt.setp(pts.collections, zorder=100, label="")

    # needed to prevent cut-off of dots near top of graph
    plt.margins(0.15)

    # mm_line = mlines.Line2D([0, 1], [means["MMus"], means["MMus"]], color=blue_cmp.colors[150])
    # st_line = mlines.Line2D([0, 1], [means["STeg"], means["STeg"]], color=orange_cmp.colors[150])
    
    # ax.add_line(mm_line)
    # ax.add_line(st_line)

    plt.title(title, size=24)
    plt.ylim(ylim) # make sure y axis starts at 0
    plt.xlabel(None)
    plt.ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # increase text size
    plt.rcParams.update({'font.size': 12})

    # set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size[0],fig_size[1])

    if add_legend:
        legend = Line2D([], [], color="black", marker="+", linewidth=0, label="mean, "+err)
        plt.legend(handles=[legend], loc="lower right")
    else:
        plt.legend([],[], frameon=False)


    return(fig)