# %% imports
import numpy as np
import matplotlib.pyplot as plt
import os

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


def Gaussian_1D(x, A, x_c, σ, bkg=0, N=1):
    """_summary_

    Args:
        x (_type_): _description_
        A (_type_): _description_
        x_c (_type_): _description_
        bkg (int, optional): _description_. Defaults to 0.
        N (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # Note the optional input N, used for super Gaussians (default = 1)
    x_c = float(x_c)
    G = A * np.exp(- (((x - x_c) ** 2) / (2 * σ ** 2))**N) + bkg
    return G


# %% Do some stuff
x = np.linspace(-10, 10, 1000)
y = Gaussian_1D(x,1,0,1)
# %% plot figure
ax2, fig2 = set_figure()
ax2.plot(x,y,
         color='xkcd:red',
        )
plt.tight_layout()
plt.show()
# %% save figure
os.chdir(r"G:\My Drive\Plots")
PPT_save_plot(fig2, ax2, 'Gaussian.svg')
