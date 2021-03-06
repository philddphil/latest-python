##############################################################################
# Import some libraries
##############################################################################
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import prd_file_import
import prd_plots
import prd_maths


###############################################################################
# Data processing defs
###############################################################################
# Cosmic ray removal ##########################################################
def cos_ray_rem(data, thres):
    data_proc = copy.copy(data)
    grad_data = np.pad(np.gradient(data), 3, 'minimum')
    for n0, m0 in enumerate(grad_data):
        if m0 > thres:
            # if gradient of data[n] is above threshold, relace it with mean
            # of data[n-2] & data[n+2]
            data_proc[n0 - 2] = np.mean([data[n0 - 4], data[n0]])
    return data_proc


# Select subset of HH data set, centred at τ (ps) with surrounding # points
def HH_data_subset(τs, cts, τ, points):
    τ_nearest, τ_idx = prd_maths.find_nearest(τs, τ)
    print(τ_nearest, τ_idx)
    τs_ss = τs[int(τ_idx - points / 2):int(τ_idx + points / 2)]
    cts_ss = cts[int(τ_idx - points / 2):int(τ_idx + points / 2)]
    return τs_ss, cts_ss


# Gaussian fits on spectral sequence used on 11th of June. See onenote labook
def spec_seq_Gauss_fit_20190611(d, popt, idx_pk, roi, pk_lb, Ps):
    print(d)
    λs, ctss, lbs = prd_file_import.load_spec_dir(d)
    prd_plots.ggplot()
    cs = prd_plots.palette()
    size = 4
    colors = plt.cm.viridis(np.linspace(0, 1, len(λs[0:])))

    fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('Wavelength shifted with excitation power')
    ax1.set_ylabel('Counts')
    ax1.set_title('Spectra with fits - ' + pk_lb)
    fig1.tight_layout()

    fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
    ax2 = fig2.add_subplot(1, 1, 1)
    fig2.patch.set_facecolor(cs['mnk_dgrey'])
    ax2.set_xlabel('Power (μW)')
    ax2.set_ylabel('Fit amplitude')
    ax2.set_title('Fit amplitude with power - ' + pk_lb)
    fig2.tight_layout()

    fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
    ax3 = fig3.add_subplot(1, 1, 1)
    fig3.patch.set_facecolor(cs['mnk_dgrey'])
    ax3.set_xlabel('Power (μW)')
    ax3.set_ylabel('Peak location, (λ$_c$ - nm)')
    ax3.set_title('Peak location with power - ' + pk_lb)
    fig3.tight_layout()

    fig4 = plt.figure('fig4', figsize=(size * np.sqrt(2), size))
    ax4 = fig4.add_subplot(1, 1, 1)
    fig4.patch.set_facecolor(cs['mnk_dgrey'])
    ax4.set_xlabel('Power (P - μW)')
    ax4.set_ylabel('Peak width (σ - nm)')
    ax4.set_title('Peak widths with power - ' + pk_lb)
    fig4.tight_layout()

    μs = []
    σs = []
    As = []

    for i0, val0 in enumerate(λs[0:]):
        P = Ps[i0]
        x = λs[i0]
        y = ctss[i0]
        x_roi = x[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
        y_roi = y[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]

        x_fit = np.linspace(min(x_roi), max(x_roi), 1000)

        popt, pcov = curve_fit(prd_maths.Gaussian_1D,
                               x_roi, y_roi, p0=[popt])
        As.append(popt[0])
        μs.append(popt[1])
        σs.append(popt[2])

        # Final plots
        prd_plots.ggplot()

        colors = plt.cm.viridis(np.linspace(0, 1, len(λs)))
        ax1.plot(x - popt[1] + i0, y, '--', alpha=0.5,
                 color=colors[i0],
                 label='',
                 lw=0)

        ax1.plot(x_roi - popt[1] + i0, y_roi, '.',
                 c=colors[i0],
                 alpha=0.3)

        ax1.plot(x_fit - popt[1] + i0, prd_maths.Gaussian_1D(
            x_fit - popt[1] + i0, *(popt[0], i0, popt[2], popt[3])),
            label='fit',
            c=colors[i0],
            lw=0.5)
        ax1.set_xlim((x[idx_pk - int(0.3 * roi)], x[idx_pk + int(0.3 * roi)]))
        ax1.set_xlim((-1, 12))
        fig1.tight_layout()

        ax2.plot(P, popt[0], 'o', c=colors[i0])
        fig2.tight_layout()

        ax3.plot(P, popt[1], 'o', c=colors[i0])
        fig3.tight_layout()

        ax4.plot(P, popt[2], 'o', c=colors[i0])
        fig4.tight_layout()

    plt.show()

    # ax1.figure.savefig(pk_lb + ' ' + ax1.get_title() + ' dark' + '.png')
    # ax2.figure.savefig(pk_lb + ' ' + ax2.get_title() + ' dark' + '.png')
    # ax3.figure.savefig(pk_lb+ ax3.get_title() + ' dark' + '.png')
    # ax4.figure.savefig(pk_lb + ' ' + ax4.get_title() + ' dark' + '.png')

    prd_plots.PPT_save_2d(fig1, ax1, pk_lb + ' ' + ax1.get_title())
    prd_plots.PPT_save_2d(fig2, ax2, pk_lb + ' ' + ax2.get_title())
    prd_plots.PPT_save_2d(fig3, ax3, pk_lb + ' ' + ax3.get_title())
    prd_plots.PPT_save_2d(fig4, ax4, pk_lb + ' ' + ax4.get_title())

    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    return As, μs, σs


# Gaussian fits on spectral sequence used on 5th of May. See onenote labook
def spec_seq_Gauss_fit_20190516(d, popt, idx_pk, roi, pk_lb):
    print(d)
    λs, ctss, lbs = prd_file_import.load_spec_dir(d)
    prd_plots.ggplot()
    cs = prd_plots.palette()
    size = 4
    colors = plt.cm.viridis(np.linspace(0, 1, len(λs[0:])))

    fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mnk_dgrey'])
    ax1.set_xlabel('Relative wavelength (λ - nm)')
    ax1.set_ylabel('Counts')
    ax1.set_title('Spectra with fits - ' + pk_lb)
    fig1.tight_layout()

    fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
    ax2 = fig2.add_subplot(1, 1, 1)
    fig2.patch.set_facecolor(cs['mnk_dgrey'])
    ax2.set_xlabel('Power (μW)')
    ax2.set_ylabel('Fit amplitude')
    ax2.set_title('Fit amplitude with power - ' + pk_lb)
    fig2.tight_layout()

    fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
    ax3 = fig3.add_subplot(1, 1, 1)
    fig3.patch.set_facecolor(cs['mnk_dgrey'])
    ax3.set_xlabel('Power (μW)')
    ax3.set_ylabel('Peak location, (λ$_c$ - nm)')
    ax3.set_title('Peak location with power - ' + pk_lb)
    fig3.tight_layout()

    fig4 = plt.figure('fig4', figsize=(size * np.sqrt(2), size))
    ax4 = fig4.add_subplot(1, 1, 1)
    fig4.patch.set_facecolor(cs['mnk_dgrey'])
    ax4.set_xlabel('Power (P - μW)')
    ax4.set_ylabel('Peak width (σ - nm)')
    ax4.set_title('Peak widths with power - ' + pk_lb)
    fig4.tight_layout()

    μs = []
    σs = []
    As = []
    Ps = []

    for i0, val0 in enumerate(λs[0:]):
        lb = lbs[i0].split()[1]
        regex = re.compile(r'\d+')
        Id = float(regex.findall(str(lb))[0])
        P = 0.33 * Id - 8.01
        x = λs[i0]
        y = ctss[i0]
        x_roi = x[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]
        y_roi = y[int(idx_pk - roi / 2):int(idx_pk + roi / 2)]

        x_fit = np.linspace(min(x_roi), max(x_roi), 1000)

        popt, pcov = curve_fit(prd_maths.Gaussian_1D,
                               x_roi, y_roi, p0=[popt])
        As.append(popt[0])
        μs.append(popt[1])
        σs.append(popt[2])
        Ps.append(P)
        # Final plots
        prd_plots.ggplot()

        colors = plt.cm.viridis(np.linspace(0, 1, len(λs)))
        ax1.plot(x - popt[1] + i0, y, '--', alpha=0.5,
                 color=colors[i0],
                 label='',
                 lw=0)

        ax1.plot(x_roi - popt[1] + i0, y_roi, '.',
                 c=colors[i0],
                 alpha=0.3)

        ax1.plot(x_fit - popt[1] + i0, prd_maths.Gaussian_1D(
            x_fit - popt[1] + i0, *(popt[0], i0, popt[2], popt[3])),
            label='fit',
            c=colors[i0],
            lw=0.5)
        # ax1.set_xlim((x[idx_pk - int(0.3 * roi)], x[idx_pk + int(0.3 *
        # roi)]))
        ax1.set_xlim(auto=True)
        fig1.tight_layout()

        ax2.plot(P, popt[0], 'o', c=colors[i0])
        fig2.tight_layout()

        ax3.plot(P, popt[1], 'o', c=colors[i0])
        fig3.tight_layout()

        ax4.plot(P, popt[2], 'o', c=colors[i0])
        fig4.tight_layout()

    plt.show()

    # ax1.figure.savefig(pk_lb + ' ' + ax1.get_title() + ' dark' + '.png')
    # ax2.figure.savefig(pk_lb + ' ' + ax2.get_title() + ' dark' + '.png')
    # ax3.figure.savefig(pk_lb + ' ' + ax3.get_title() + ' dark' + '.png')
    # ax4.figure.savefig(pk_lb + ' ' + ax4.get_title() + ' dark' + '.png')

    prd_plots.PPT_save_2d(fig1, ax1, pk_lb + ' ' + ax1.get_title())
    prd_plots.PPT_save_2d(fig2, ax2, pk_lb + ' ' + ax2.get_title())
    prd_plots.PPT_save_2d(fig3, ax3, pk_lb + ' ' + ax3.get_title())
    prd_plots.PPT_save_2d(fig4, ax4, pk_lb + ' ' + ax4.get_title())

    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    return As, μs, σs, Ps
