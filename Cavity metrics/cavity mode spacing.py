# %% imports
import os
import numpy as np
from scipy import interpolate
from scipy import constants
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

    return 1-np.sqrt(L/R)


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

# %% preamble 
# Using unit of m & s, converting to MHz inline if needed
L = 367e-6

# L_offset = 0
# including 15 micron increase for 1030 DBR
R = 220e-6

nu_1033 = 290215351403678.56
nu_1377_offset = 0
nu_1377 = (nu_1033 * 3/4) + nu_1377_offset

# Difference in length due to deposition
L_offset = (constants.c/nu_1033)*(1/16)
L_offset = 0

# Set curved = true or false
curved = False

# %% calculate mode frequencies
L_min, L_max = 362e-6, 368e-6
# L_min, L_max = 320e-6, 400e-6
coarse_sensitivity = 20
Ls = np.linspace(L_min,L_max, int((L_max - L_min)*1000e6))
L_lo_res = Ls[1]-Ls[0]
Dnu_lo_res = constants.c*int(np.mean(Ls)/1033e-9)*(np.mean(Ls)**-2)*L_lo_res
Window = Dnu_lo_res * coarse_sensitivity

res_1033_GHz = np.zeros_like(Ls) 
res_1377_GHz = np.zeros_like(Ls)
res_both = np.zeros_like(Ls)
ns_1033_GHz, ns_1377_GHz, ns_both_1033, ns_both_1377 = [], [], [], []
nus_1033_GHz, nus_1377_GHz, nus_both_GHz = [], [], []
Ls_1033_GHz, Ls_1377_GHz, Ls_both_GHz = [], [], []
mode_map = []

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
# %% assess dual resonance
peaks = []
for i0, v0 in enumerate(Ls):
    if res_both_GHz[i0] == 1 and res_both_GHz[i0-1] == 0:
        peaks.append(Ls[i0])

ax1, fig1 = set_figure('modes', 'ΔL / nm', 'Δν / PHz', 3)
dnu1 = []
dnu2 = []
dnu3 = []

for i0, v0 in enumerate(peaks[0:]):
    Ls_hi_res = np.linspace(v0 - 2*coarse_sensitivity*L_lo_res, 
                            v0 + 2*coarse_sensitivity*L_lo_res, 
                            10000)
    L_hi_res = Ls_hi_res[1] - Ls_hi_res[0]
    nus_1033_L = []
    nus_1377_L = []
    
    for i1, v1 in enumerate(Ls_hi_res):
        
        if curved == True:             
            # curved cavity formula
            nus_1033_L.append(mode_nu(ns_both_1033[i0],v1, g1, g2)-nu_1033)
            nus_1377_L.append(mode_nu(ns_both_1377[i0],v1+L_offset, g1, g2)-nu_1377)
        
        else:        
            ## planar cavity formula
            nus_1033_L.append((mode_nu_approx(ns_both_1033[i0],v1))-nu_1033)
            nus_1377_L.append((mode_nu_approx(ns_both_1377[i0],v1+L_offset))-nu_1377)
    
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
    ## planar cavity formula
    # dnu3.append(mode_nu_approx(ns_both_1033[i0],interp_1377(0))-nu_1033)
    ## curved cavity formula
    dnu3.append(mode_nu(ns_both_1033[i0],interp_1377(0), g1, g2)-nu_1033)
    
    # ax1.plot(1e9*(Ls_hi_res-v0),
    #      np.array(nus_1033_L)*1e-9,
    #      '.',
    #      color='xkcd:blue',
    #      )

    # ax1.plot(1e9*(Ls_hi_res-v0),
    #      np.array(nus_1377_L)*1e-9,
    #      '.',
    #      color='xkcd:red'
    #      )
    
    # ax1.plot(1e9*(L_1377_res-v0),0,
    #          'o',
    #          mfc='none',
    #          color='xkcd:dark red',
    #          )
    
    # ax1.plot(1e9*(L_1377_res-v0),

    #          1e-9*(L_1377_res_1033_nu-nu_1033),
    #          'o',
    #          mfc='none',
    #          color='xkcd:dark blue',
    #          )

    # ax1.set_xlim([10,15])

    # ax1.set_ylim([-3,3])

plt.show()

# %% plots
ax0, fig0 = set_figure('modes', 'L / μm', 'y', 3)
ax0.plot(1e6*Ls,2*res_1033_GHz,
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

# for i0, v0 in enumerate(Ls_1033_GHz):
#     if int(i0)%2 ==0:
#         ax0.text(1e6*v0, (2*i0)/(len(Ls_1033_GHz)), 
#                 int(ns_1033_GHz[i0]),
#                 size=6)

# for i0, v0 in enumerate(Ls_1033_GHz):

#     ax0.text(1e6*v0, 1.5, 
#             int(ns_1033_GHz[i0]),
#             size=6)

# for i0, v0 in enumerate(Ls_1377_GHz):
#     ax0.text(1e6*v0, 0.5, 
#             int(ns_1377_GHz[i0]),
#             size=6)

for i0, v0 in enumerate(peaks):
    ax0.text(1e6*v0, 0.6, 
            int(ns_both_1033[i0]),
            size=6)
    ax0.text(1e6*v0, 0.5, 
            int(ns_both_1377[i0]),
            size=6,
            )
    
# ax0.legend(edgecolor='xkcd:black',
#            loc='upper right'
#            )
# fig0.tight_layout()
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
# %% save plot
PPT_save_plot(fig0, ax0, 'cav spec')
# PPT_save_plot(fig1, ax1, 'fine detuning plot')
PPT_save_plot(fig2, ax2, 'large range detunings')
