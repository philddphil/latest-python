# %% imports
import sys
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# %% defs
def ggplot_sansserif():
    """
    Set some parameters for ggplot
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 8
    plt.rcParams['lines.color'] = 'white'
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['axes.labelcolor'] = 'xkcd:black'
    plt.rcParams['xtick.color'] = 'xkcd:black'
    plt.rcParams['ytick.color'] = 'xkcd:black'
    plt.rcParams['axes.edgecolor'] = 'xkcd:black'
    plt.rcParams['savefig.edgecolor'] = 'xkcd:black'
    plt.rcParams['axes.facecolor'] = 'xkcd:dark gray'
    plt.rcParams['savefig.facecolor'] = 'xkcd:dark gray'
    plt.rcParams['grid.color'] = 'xkcd:dark gray'
    plt.rcParams['grid.linestyle'] = 'none'
    plt.rcParams['axes.titlepad'] = 6


def set_figure(name: str = 'figure',
               xaxis='x axis',
               yaxis='y axis',
               size: float = 4,
               ):
    """
    Set a figure up
    Parameters
    ----------
    name : str
        name of figure
    xaxis : str
       x axis label
    yaxis : str
       y axis label
    size : float

    Returns
    -------
    ax1 : axes._subplots.AxesSubplot
    fig1 : figure.Figure
    """
    ggplot_sansserif()
    fig1 = plt.figure(name, figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(111)
    fig1.patch.set_facecolor('xkcd:dark grey')
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel(yaxis)
    return ax1, fig1


def PPT_save_plot(fig, ax, name, dpi=600):
    # Set plot colours
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')
    ax.legend(
        fancybox=True,
        facecolor=(1.0, 1.0, 1.0, 1.0),
        labelcolor='black',
        loc='upper left',
        framealpha=0)
    ax.get_legend().remove()
    fig.tight_layout()

    # Loop to check for file - appends filename with _# if name already exists
    f_exist = True
    app_no = 0
    while f_exist is True:
        if os.path.exists(name + '.png') is False:
            ax.figure.savefig(name, dpi=dpi)
            f_exist = False
            print('Base exists')
        elif os.path.exists(name + '_' + str(app_no) + '.png') is False:
            ax.figure.savefig(name + '_' + str(app_no), dpi=dpi)
            f_exist = False
            print(' # = ' + str(app_no))
        else:
            app_no = app_no + 1
            print('Base + # exists')


def Gauss_hist(a, bins=10, rng=3, res=1000):
    """ Get Gaussian fit of histogram of data set a

    Args:
        a (array): set of data values
        bins (int, optional): number of bins in histogram. Defaults to 10.
        rng (int, optional): range of histogram. Defaults to 3.
        res (int, optional): resolution of plotted histogram fit. Defaults to 1000.

    Returns:
        x (array): range of x values (correspoding to data values)
        y (array): frequency of data values
    """
    μ = np.mean(a)
    σ = np.sqrt(np.var(a))
    n, bins = np.histogram(a, bins)
    x = np.linspace(μ - rng * σ, μ + rng * σ, res)
    y = Gaussian_1D(x, np.max(n), μ, σ)
    return x, y


def Gaussian_1D(x, A, x_c, x_w, bkg=0, N=1):
    """ See wikipedia
    https://en.wikipedia.org/wiki/Normal_distribution

    Args:
        x (array): x values
        A (float): peak value
        x_c (float): mean
        x_w (float): standard deviation
        bkg (int, optional): y offset. Defaults to 0.
        N (int, optional): order. Defaults to 1.

    Returns:
        array: y values
    """
    # Note the optional input N, used for super Gaussians (default = 1)
    x_c = float(x_c)
    G = A * np.exp(- (((x - x_c) ** 2) / (2 * x_w ** 2))**N) + bkg
    return G
    # Note the optional input N, used for super Gaussians (default = 1)
    x_c = float(x_c)
    G = A * np.exp(- (((x - x_c) ** 2) / (2 * σ ** 2))**N) + bkg
    return G



# %% Do some stuff
π = np.pi
phi = 0.1
ts = np.linspace(-6 * π, 6 * π, 1000)
Es0 = Gaussian_1D(ts, 1, 0, 2 * π, 0, 4) * np.cos(ts*2)
Es1 = Gaussian_1D(ts, 1, 0, 2 * π, 0) * np.cos(ts*2+π/4)
Es2 = Gaussian_1D(ts, 1, 0, 2 * π, 0) * np.cos(ts*2+π/3)
Es3 = Gaussian_1D(ts, 1, 0, 2 * π, 0) * np.cos(ts*2+π/2)
Es4 = Gaussian_1D(ts, 1, 0, 2 * π, 0) * np.cos(ts*2+2*π/3)
Es5 = Gaussian_1D(ts, 1, 0, 2 * π, 0) * np.cos(ts*2+3*π/4)
Es6 = Gaussian_1D(ts, 1, 0, 2 * π, 0) * np.cos(ts*2+π)


#  Plot some figures




ax1, fig1 = set_figure('wavepacket', 'x', 'y')
ax1.plot(ts, Es0**2, 
         '-',
         color='#ff652a',
         alpha=1,
         )
# ax1.plot(ts, Es1, 
#          '-',
#          color='#00bdcd7f',
#          alpha=0.4,
#          )
# ax1.plot(ts, Es2, 
#          '-',
#          color='#00bdcd7f',
#          alpha=0.45,
#          )
# ax1.plot(ts, Es3, 
#          '-',
#          color='#00bdcd7f',
#          alpha=0.5,
#          )
# ax1.plot(ts, Es4, 
#          '-',
#          color='#00bdcd7f',
#          alpha=0.45,
#          )
# ax1.plot(ts, Es5, 
#          '-',
#          color='#00bdcd7f',
#          alpha=0.4,
#          )
# ax1.plot(ts, Es6, 
#          '-',
#          color='#00bdcd7f',
#          alpha=0.3,
#          )
fig1.tight_layout()
ax1.axis('off')
# Save plot
os.chdir(r"G:\My Drive\Plots")
PPT_save_plot(fig1, ax1, 'wavepacket.svg')
