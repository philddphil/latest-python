# %% import
import os
import sys
import glob
import time
import unicodedata
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime


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


def PM100Dcsv(file):
    """Read in a csv saved from a Thorlabs PM100D

    Args:
        file (path): path to .csv (including the filename itself)

    Returns:
        ts (list of datetimes): list of times of acquired data
        dts (list of floats): list of relative time of acquired data (s)
        Ps (list of floats): list of optical powers (W)
    """

    d0 = open(file, 'r', encoding='utf-8')
    x0 = d0.readlines()
    d0.close()

    ts = []
    dts = []
    Ps = []

    for i0, v0 in enumerate(x0[15:]):
        t = str(v0.split(",")[1].strip() + v0.split(",")[2])
        t2 = v0.split(",")[2]
        t2s = datetime.strptime(t2.strip(), "%H:%M:%S.%f")
        ts0 = datetime.strptime(t, '%d/%m/%Y %H:%M:%S.%f')
        ts.append(ts0)
        dts.append(ts0.timestamp())
        Ps.append(float(v0.split(',')[3]))

    dts = np.asarray(dts) - np.array(dts[0])

    return ts, dts, Ps


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


# %% stuff
path = (r"G:\Shared drives\Projects\Innovate\LYRA\WP2 Test Harness Development"
      r"\Data\QPhotonics Laser\Constant_LD_Current_Mode")
datafiles = glob.glob(path + r'\*.csv')
datafiles.sort(key=os.path.getmtime)
cmap = mpl.colormaps['plasma']
for i0, v0 in enumerate(datafiles[:]):
    # establish data
    p0 = v0
    ts, dts, Ps = PM100Dcsv(p0)
    path = Path(p0)
    # basic statistics
    Ps_mean = np.mean(Ps)
    Ps_std = np.std(Ps)
    # histogram dataset
    bin_n = 20
    bin_k = 5
    bin_cs = np.linspace(Ps_mean-bin_k*Ps_std, # type: ignore
                         Ps_mean+bin_k*Ps_std, # type: ignore
                         bin_n)
    bin_w = bin_cs[1]-bin_cs[0]
    bin_es = np.linspace(Ps_mean-bin_k*Ps_std - bin_w/2, # type: ignore
                         Ps_mean+bin_k*Ps_std + bin_w/2, # type: ignore
                         bin_n+1)
    Ps_hist, __ = np.histogram(Ps, bin_es)
    # plot figs
    ax0, fig0 = set_figure('time vs power', 'time / s', 'power / μW', 2)
    ax0.plot(dts, 1e6*np.asarray(Ps),
            '.',
            color=cmap(i0/len(datafiles))
            )
    fig0.tight_layout()
    
    ax1, fig1 = set_figure('power hist', 'power / μW', 'N / #', 2)
    ax1.bar(bin_cs,Ps_hist, 
            width=0.8*bin_w,
            color=cmap(i0/len(datafiles)))
    fig1.tight_layout()

    ax2, fig2 = set_figure('composite', 'normalised power', 'N / #', 2)
    ax2.bar((bin_cs/Ps_mean),Ps_hist, 
            width=0.8*bin_w,
            color=cmap(i0/len(datafiles)))
    fig2.tight_layout()

    plt.show()
    # %% save figure
    os.chdir(path.parent)
    # PPT_save_plot(fig0, ax0, str(slugify(path.stem)+'_timeseries'))
    # PPT_save_plot(fig1, ax1, str(slugify(path.stem)+'_hist'))
    fig0.clf()
    fig1.clf()
