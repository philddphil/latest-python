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


# %% Do some stuff
x = np.linspace(0.9, 1.1, 500)
A = 50
SNR = 1 / 20
bins = 15
value = 10
noise = np.random.normal(0, A * SNR, x.shape) + value
Gauss_x, Gauss_y = prd_maths.Gauss_hist(noise, bins)

# %% Plot some figures
plot_path = r"G:\My Drive"
# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))

###### xy plot ###############################################################
size = 4

fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(111)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('measurement number')
ax2.set_ylabel('value')
plt.plot(noise, '.', alpha=1, color=cs['gglred'], label='')
plt.plot(noise, alpha=0.2, color=cs['gglred'], label='')
[ymin, ymax] = ax2.get_ylim()
plt.tight_layout()

size = 4
fig3 = plt.figure('fig3', figsize=(size, size))
ax3 = fig3.add_subplot(111)
fig3.patch.set_facecolor(cs['mnk_dgrey'])
ax3.set_xlabel('freq of occurances (#)')
ax3.set_ylabel('value')
plt.hist(noise, bins, edgecolor=cs['mnk_dgrey'], color=cs['ggred'], alpha=0.5,
         orientation="horizontal")
plt.plot(Gauss_y, Gauss_x, alpha=1, color=cs['gglblue'], label='')
ax3.set_ylim(ymin, ymax)
plt.tight_layout()
# plt.hist(G_noise, 10, alpha=1, color=cs['ggdred'], lw=0.5, label='decay')
# plt.plot(x2, y2, '.', alpha=0.4, color=cs['gglblue'], label='')
# plt.plot(x2, y2, alpha=1, color=cs['ggblue'], lw=0.5, label='excite')

###### xyz plot ##############################################################
# size = 4
# fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
# ax3 = fig3.add_subplot(111, projection='3d')
# fig3.patch.set_facecolor(cs['mnk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# scatexp = ax3.scatter(*coords, z, '.', alpha=0.4, color=cs['gglred'], label='')
# surffit = ax3.contour(*coords, z, 10, cmap=cm.jet)

# ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
# # os.chdir(p0)

# ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# set_zlim(min_value, max_value)

plt.show()
plot_file_name = plot_path + r'\hist.png'
prd_plots.PPT_save_2d(fig3, ax3, plot_file_name)
plot_file_name = plot_path + r'\tseries.png'
prd_plots.PPT_save_2d(fig2, ax2, plot_file_name)
