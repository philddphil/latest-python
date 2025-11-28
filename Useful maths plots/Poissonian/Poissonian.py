
#%% Import some libraries
import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# %% Some defs
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


def PPT_save_plot(fig, ax, name, dpi=600, legend=True):
    """ saves a plot as 'name'.png (unless .svg is specified in the name). 
    Iterates name (name_0, name_1, ...) if file already exists with that name.

    Args:
        fig (plt.fig): figure pointer
        ax (plt.ax): axis pointer
        name (str): desired name of file. No extension produced .png with name. Add .svg as ext. if needed
        dpi (int, optional): _description_. Defaults to 600.
    """
    # Set plot colours
    plt.rcParams['text.color'] = 'xkcd:black'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:black')
    ax.yaxis.label.set_color('xkcd:black')
    ax.tick_params(axis='x', colors='xkcd:black')
    ax.tick_params(axis='y', colors='xkcd:black')
    # handle legends
    if legend==True:
        ax.legend(
            fancybox=True,
            facecolor=(1.0, 1.0, 1.0, 0.5),
            labelcolor='black',
            # loc='lower right',
            # loc='upper right',
            framealpha=1.0)

    else: 
        try:
            ax.get_legend().remove()
        except AttributeError:
            pass

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


# Poissonian distribution at values of k for mean value λ #####################
def Poissonian_1D(k, λ):
    P = []

    for i0, j0 in enumerate(k):
        P.append(np.exp(-λ) * (λ**j0) / sp.special.gamma(j0 + 1))

    return P


##############################################################################
# %% Do some stuff
n_max = 5
res = n_max + 1

n_bar1 = 1
n_bar2 = 1

n_ints = np.linspace(0, 10, 11)
n_cont = np.linspace(0, 10, 1024)

P_ints2 = Poissonian_1D(n_ints, n_bar2)
P_cont2 = Poissonian_1D(n_cont, n_bar2)

plot_path = r"C:\Plots\\"
x_d = n_ints
y_d2 = P_ints2

x_c = n_cont
# y_c1 = P_cont1
y_c2 = P_cont2


ax0, fig0 = set_figure('figure0', r'$\langle n \rangle$', 'Probability', 2.5)

ax0.bar([0,1],[0.5, 0.5],
        alpha=1,
        color='xkcd:green',
        label=r'|$\psi\rangle$')
ax0.legend()
ax0.set_xlim(-0.5, 10.5)
ax0.set_ylim(0, 1.1)

ax1, fig1 = set_figure('figure1', r'$\langle n \rangle$', 'Probability', 2.5)
ax1.bar(1, 1,
        alpha=1,
        color='xkcd:blue',
        label=r'|1$\rangle$')
ax1.legend()
ax1.set_xlim(-0.5, 10.5)    
ax1.set_ylim(0, 1.1)

ax2, fig2 = set_figure('figure2', r'$\langle n \rangle$', 'Probability', 2.5)

ax2.plot(x_c, y_c2, 
         alpha=0.8, 
         color='xkcd:dark red', 
         label=r'$\mathbb{N}$')

ax2.bar(x_d, y_d2,
        alpha=1,
        color='xkcd:red',
        label=r'|α$_{}\rangle$')
ax2.legend()
ax2.set_xlim(-0.5, 10.5)
ax2.set_ylim(0, 1.1)

os.chdir(plot_path)
plt.tight_layout()
plt.show()

# %% Save the plots
PPT_save_plot(fig0, ax0, 'plot0')
PPT_save_plot(fig1, ax1, 'plot1')
PPT_save_plot(fig2, ax2, 'plot2')

