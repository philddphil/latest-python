# %% 0. imports
import os
import numpy as np
from scipy import interpolate
from scipy import constants
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %% 1. defs
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


def PPT_save_plot(fig, ax, name, dpi=600, png=True, legend=True):
    """ saves a plot as 'name'.png or an .svg depending on final boolean. 
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
    if legend == False:
        ax.get_legend().remove()
    fig.tight_layout()

    # Loop to check for file - appends filename with _# if name already exists
    if png == True:
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
    else:
        f_exist = True
        app_no = 0
        while f_exist is True:
            if os.path.exists(name + '.svg') is False:
                ax.figure.savefig(name + '.svg', dpi=dpi)
                f_exist = False
                print('Base exists')
            elif os.path.exists(name + '_' + str(app_no) + '.svg') is False:
                ax.figure.savefig(name + '_' + str(app_no) + '.svg', dpi=dpi)
                f_exist = False
                print(' # = ' + str(app_no))
            else:
                app_no = app_no + 1
                print('Base + # exists')


def mode_nu(n,L,g1,g2):
    """ Calculate cavity mode frequency given cavity parameters 

    Args:
        n (float): mode number
        L (float): cavity length (m)
        g1 (float): mirror 1 cav parameter
        g2 (float): mirror 2 cav parameter

    Returns:
        nu_n (float): mode frequency (Hz)
    """
    return constants.c/(2*L) * (n+(1/constants.pi)*np.arccos(np.sqrt(g1*g2)))


def mode_nu_approx(n,L):
    """_summary_

    Args:
        n (int): longitudinal mode number
        L (float): cavity length (m)

    Returns:
        nu: mode frequency
    """
    return (constants.c*n)/(2*L)


def cav_g(L, R):
    """Calculate cavity g parameter from L and R

    Args:
        L (float): cavity length (m)
        R (float): cavity radius of curvature (m)

    Returns:
        g (float): 'cavity parameter' unitless
    """
    g = float(1-np.sqrt(L/R))
    return g


def linear(x, m, c):
    """ y = mx + c

    Args:
        x (array): array of x values to evaluate y at
        m (float): gradient of straight line
        c (float): y offset

    Returns:
        y (array): y values
    """
    y = (m*x) + c
    return y

# %% 2. preamble 
# Using unit of m & s, converting to MHz inline if needed
# Cavity length
L = 367e-6
# Radius of curvature
R = 220e-6

nu_1033 = 290215351403678.56
nu_1377_offset = 104e9
nu_1377 = (nu_1033 * 3/4) + nu_1377_offset

# Difference in length due to deposition
# - additional length in 1377 resonances experience 
#  w.r.t length 1033 experiences:

# L_offset = (constants.c/nu_1033)*(1/16)
L_offset = 0

# Set curved = true or false. Uses planar approximation, or not
curved = False

# %% 3. calculate mode frequencies
# %% 3.1 course assessment to find dual resonant lengths
L_min, L_max = 335e-6, 395e-6
Ls = np.linspace(L_min,L_max, int((L_max - L_min)*1e9))
L_lo_res = Ls[1]-Ls[0]
Dnu_lo_res = constants.c*int(np.mean(Ls)/1033e-9)*(np.mean(Ls)**-2)*L_lo_res
Window_range = 7
Window = Dnu_lo_res * Window_range

# Pre-allocate result arrays and lists
res_1033_GHz = np.zeros_like(Ls) 
res_1377_GHz = np.zeros_like(Ls)
res_both = np.zeros_like(Ls)
ns_1033_GHz, ns_1377_GHz, ns_both_1033, ns_both_1377 = [], [], [], []
nus_1033_GHz, nus_1377_GHz, nus_both_GHz = [], [], []
Ls_1033_GHz, Ls_1377_GHz, Ls_both_GHz = [], [], []
# supress weird unbound variable issue
g1 = None
g2 = None

for i0, v0 in enumerate(Ls[:]):
    ## For loop over lengths (coarse)
    n_1033 = int(2*v0/1033e-9) 
    ns_1033 = np.linspace(n_1033 - 20, n_1033 +20, 41).tolist()
    n_1377 = int((2*v0+L_offset)/1377e-9) 
    ns_1377 = np.linspace(n_1377 - 20, n_1377 +20, 41).tolist()

    g1 = cav_g(v0,R)
    g2 = cav_g(v0+L_offset,R)

    for i1, v1 in enumerate(ns_1033[:]):
        ## For loop over local mode numbers
        n_1033_loop = v1
        n_1377_loop = ns_1377[i1]
        
        if curved == True:
            
            ## curved cavity formula
            nu_1033_L = mode_nu(n_1033_loop,v0,g1,g1)
            nu_1377_L = mode_nu(n_1377_loop,v0+L_offset,g2,g2)  
        
        else:
            ## planar cavity formula
            nu_1033_L = mode_nu_approx(n_1033_loop,v0)
            nu_1377_L = mode_nu_approx(n_1377_loop,v0+L_offset)

        # mode_map.append([v0,nu_1033_L,nu_1377_L,n_1033_loop,nu_1377_L]
        if np.abs(nu_1033 - nu_1033_L) < Window:
            res_1033_GHz[i0] = 1
            if res_1033_GHz[i0-1] == 0:
                Ls_1033_GHz.append(v0)
                ns_1033_GHz.append(n_1033_loop)
                nus_1033_GHz.append(nu_1033_L)

        if np.abs(nu_1377 - nu_1377_L) < Window:
            res_1377_GHz[i0] = 1
            if res_1377_GHz[i0-1] == 0:
                Ls_1377_GHz.append(v0)
                ns_1377_GHz.append(n_1377_loop)
                nus_1377_GHz.append(nu_1377_L)
        
    if res_1033_GHz[i0] == 1 and res_1377_GHz[i0] == 1:
        ns_both_1033.append(ns_1033_GHz[-1])
        ns_both_1377.append(ns_1377_GHz[-1])
            
ns_both_1033 = list(dict.fromkeys(ns_both_1033))
ns_both_1377 = list(dict.fromkeys(ns_both_1377))
res_both_GHz = res_1033_GHz * res_1377_GHz
# %% 3.2 fine assessment of dual resonances to determine detunings
peaks = []
for i0, v0 in enumerate(Ls):
    if res_both_GHz[i0] == 1 and res_both_GHz[i0-1] == 0:
        peaks.append(Ls[i0])

dnu1 = []
dnu2 = []
dnu3 = []

for i0, v0 in enumerate(peaks[0:]):
    Ls_hi_res = np.linspace(v0 - 2*Window_range*L_lo_res, 
                            v0 + 2*Window_range*L_lo_res, 
                            int(1e4))
    L_hi_res = Ls_hi_res[1] - Ls_hi_res[0]
    nus_1033_L = []
    nus_1377_L = []
    
    for i1, v1 in enumerate(Ls_hi_res):
        
        if curved == True:               
            # curved cavity formula
            nus_1033_L.append(mode_nu(ns_both_1033[i0],
                                      v1, 
                                      g1, 
                                      g2)
                                      - nu_1033)
            nus_1377_L.append(mode_nu(ns_both_1377[i0],
                                      v1+L_offset, 
                                      g1, 
                                      g2)
                                      -nu_1377)
        
        else:        
            ## planar cavity formula
            nus_1033_L.append((mode_nu_approx(ns_both_1033[i0],
                                              v1))
                                              -nu_1033)
            nus_1377_L.append((mode_nu_approx(ns_both_1377[i0],
                                              v1+L_offset))
                                              -nu_1377)
    
    ## curve fitting method
    # m0 = (nus_1377_L[1]-nus_1377_L[0])/(Ls_hi_res[1]-Ls_hi_res[0])
    # c0 = nus_1377_L[0]
    # popt, gof = curve_fit(linear,Ls_hi_res,nus_1377_L,p0=[m0, c0])
    # L_1377_res = -1*popt[1]/popt[0]
    # L_1377_res_1033_nu = mode_nu_approx(ns_both_1033[i0],L_1377_res)
    # dnu1.append(mode_nu_approx(ns_both_1033[i0],L_1377_res)-nu_1033)

    ## argmin method
    # L_1377_res = Ls_hi_res[np.argmin(np.abs(nus_1033_L))]
    # L_1377_res_1033_nu = mode_nu_approx(ns_both_1033[i0],L_1377_res)
    # dnu2.append(mode_nu_approx(ns_both_1033[i0],L_1377_res)-nu_1033)

    ## interp1d method
    interp_1377 = interpolate.interp1d(nus_1377_L, Ls_hi_res)
    if curved == True:
        dnu3.append(mode_nu(ns_both_1033[i0],
                            interp_1377(0), 
                            g1, 
                            g2)
                            -nu_1033)
    else:
        dnu3.append(mode_nu_approx(ns_both_1033[i0],
                                   interp_1377(0))
                                   -nu_1033)
        
    # ax1, fig1 = set_figure('modes', 'ΔL / nm', 'Δν / GHz', 2)
    # ax1.plot(1e9*(Ls_hi_res-v0),
    #      np.array(nus_1033_L)*1e-9,
    #     #  '.',
    #      color='xkcd:blue',
    #      label='Δν of λ$_{ion}$'
    #      )

    # ax1.plot(1e9*(Ls_hi_res-v0),
    #      np.array(nus_1377_L)*1e-9,
    #     #  '.',
    #      color='xkcd:red',
    #      label='Δν of λ$_{lock}$'
    #      )
    
    # ax1.plot(1e9*(Ls_hi_res-v0), np.zeros_like(Ls_hi_res),
    #      ':',
    #      color='xkcd:black',
    #      lw=0.5,
    #      label='resonance')
    
    # ax1.legend(edgecolor='xkcd:black',
    #        loc='upper right'
    #        )
    # fig1.tight_layout()

# %% 4. plots
ax0, fig0 = set_figure('modes', 'L / μm', 'y', 3)
ax0.plot(1e6*Ls,res_1033_GHz,
         '-',
         color='xkcd:blue',
         lw=0.5,
         label='1033')
ax0.plot(1e6*Ls,res_1377_GHz,
         '-',
         color='xkcd:red',
         lw=0.5,
         label='1377')
ax0.plot(1e6*Ls,res_both_GHz,
         '-',
         color='xkcd:green',
         lw=0.5,
         label='both')

for i0, v0 in enumerate(peaks):
    ax0.text(1e6*v0, 0.6, 
            str(int(ns_both_1033[i0])),
            size=8,
            # color='xkcd:blue',
            )
    ax0.text(1e6*v0, 0.5, 
            str(int(ns_both_1377[i0])),
            size=8,
            # color='xkcd:red',
            )
    
ax0.legend(edgecolor='xkcd:black',
           loc='upper right'
           )
fig0.tight_layout()
# ax0.set_xlim([362,368])

ax2, fig2 = set_figure('dls', 'L / μm', 'dν / MHz', 3)
ax2.plot(np.array(peaks)*1e6,
         np.asarray(dnu3)*1e-6,
         '.',
         color='green')
# ax2.set_ylim([-0.1,1.5])
# ax2.set_xlim([362,368])


# ax3, fig3 = set_figure('mode map', 'L / μm', 'ν / PHz')

# for i0, v0 in enumerate(mode_map[:]):
#     ax3.plot(1e6*mode_map[i0][0],1e-12*mode_map[i0][1],
#              '.',
#              markersize=1,
#              color='xkcd:blue')
# ax3.plot([1e6*Ls[0],1e6*Ls[-1]],[1e-12*nu_1033,1e-12*nu_1033],
#          color='xkcd:blue',
#          )
# ax3a = ax3.twinx()
# ax3a.set_ylabel('wavelength / nm',
#                 )
# ax3.set_ylim([288.4,290.8])
# ax3a.set_ylim([1e9*constants.c/(1e12*288.4),1e9*constants.c/(1e12*290.8)])


# for i0, v0 in enumerate(mode_map[:]):
#     ax3.plot(1e6*mode_map[i0][0],1e-12*mode_map[i0][2],
#              '.',
#              markersize=1,
#              color='xkcd:red')
# ax3.plot([1e6*Ls[0],1e6*Ls[-1]],[1e-12*nu_1377,1e-12*nu_1377],
#          color='xkcd:red',
#          )
# ax3a = ax3.twinx()
# ax3a.set_ylabel('wavelength / nm',
#                 )
# ax3.set_ylim([216.5,218.5])
# ax3a.set_ylim([1e9*constants.c/(1e12*216.5),1e9*constants.c/(1e12*218.5)])

plt.show()
# %% 5. save plot
os.chdir(r"G:\My Drive\Plots")
# PPT_save_plot(fig0, ax0, 'cav spec', 600, False)
# PPT_save_plot(fig1, ax1, 'fine detuning plot', 600, False)
PPT_save_plot(fig2, ax2, 'detunings with length', 600, False)
