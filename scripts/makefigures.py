
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from itertools import cycle

################################################################################
### setup

# inches_per_pt = 1. / 72.27               # Convert pt to inch
# inches_per_mm = 1. / 25.4

# onecol_in = 83.5 * inches_per_mm
# twocol_in = 173.5 * inches_per_mm

usecolor = False
textsize_title = 7
textsize_small = 6
textsize_mini = 5

twocol_in = 4.25
# onecol_in = 2.
onecol_in = 0.5 * twocol_in

params = {'axes.labelsize': textsize_small,
          'text.fontsize': textsize_small,
          'legend.fontsize': textsize_small,
          'xtick.labelsize': textsize_small,
          'ytick.labelsize': textsize_small}
plt.rcParams.update(params)

if usecolor:
    colors = [[0, 0, 0], [0, 0, 1], [0, 0.5, 0], [0.8, 0, 0]]
else:
    colors = [[0, 0, 0]]

styles = [[1, 1e-8],
          [3.5, 3.5],
          [1.5, 2, 3.5, 2],
          [1, 2]]
widths = [1.5, 1, 1, 2]
styleobj = (colors,widths,styles)

bw_cdict = [(0, 0.8, 0.8), (1, 0, 0)]
bw_cdict = dict([(c, list(bw_cdict)) for c in ['red', 'green', 'blue']])
bw_cmap = matplotlib.colors.LinearSegmentedColormap('bw', bw_cdict, 256)

g_axis_i = 0
g_rows = 0
g_cols = 0

def init_subplots(rows, cols):
    global g_axis_i, g_rows, g_cols
    g_axis_i = 0
    g_rows = rows
    g_cols = cols

def next_subplot():
    global g_axis_i
    g_axis_i += 1
    return plt.subplot(g_rows, g_cols, g_axis_i)

def makesubplot(x,y,plotfun,colors,widths,styles):
    colors = cycle(colors)
    widths = cycle(widths)
    styles = cycle(styles)

    for i in range(y.shape[1]):
        l, = plotfun( x, y[:,i],
                      color=next(colors), linewidth=next(widths) )

        l.set_dashes(next(styles))

def titletext(s, ax=None, color='k', x_offset=0.0, y_offset=0.0):
    if ax is None: ax = plt.gca()
    ax.text(0.01+x_offset, 1.05-y_offset, s,
            fontsize=textsize_title, color=color, transform=ax.transAxes,)
    # ax.text(0.06+x_offset, 0.88-y_offset, s,
    #         fontsize=textsize_title, color=color, transform=ax.transAxes,)
    #         # backgroundcolor='white')

def tight_layout(top=0.94, **kwargs):
    plt.tight_layout(pad=0.1, rect=(0,0,1,top), **kwargs)

def legend(s,loc):
    plt.legend(s, loc=loc, labelspacing=0.1, frameon=False)

def legendbox(s,loc,bbox):
    plt.legend(s, loc=loc, bbox_to_anchor=bbox, labelspacing=0.1, frameon=False)

def legendupperleft(s):
    legendbox(s,'upper left',(0,0.9))

def floattolatex(f, fstr='%0.1f'):
    logn = np.floor(np.log10(f))
    numn = f / (10**logn)
    return r'$%s \times$ $10^{%d}$' % (fstr % numn, logn)

def floattolatex10(f):
    logn = np.log10(f)
    return r'$10^{%0.1f}$' % (logn)

def floattoe(f):
    mantissa, exp = ("%0.1e" % f).split('e')
    return "%se%d" % (mantissa, int(exp))

def make_mask(data, targs, rtol=1e-3):
    mask = np.zeros(data.shape, dtype='bool')
    for t in targs:
        i = np.argmin(np.abs(data - t))
        mask[i] = 1

    r = np.abs((data[mask] - targs) / targs)
    assert (r < rtol).all()
    return mask

noise_label = r'noise ($\sigma_\eta$)'
hetero_label = r'heterogeneity ($b_r$)'
info_label = 'information [bits]'
info_spike_label = 'information rate [bits/spike]'
phase_label = 'standard deviation of phase [radians]'

#########################################################################################
### Mutual information plot (x-axis: heterogeneity)

def infoheteroplot(lifdatafile, fhndatafile, target):
    lifdata = np.load(lifdatafile)
    fhndata = np.load(fhndatafile)

    noise_targs = 1e-1 * np.array([1e-3, 1e-2, 1e-1, 1e0])
    lifmask = make_mask(lifdata['Nstds'], noise_targs, rtol=1e-3)
    fhnmask = make_mask(fhndata['Nstds'], noise_targs, rtol=1e-3)

    bw = 1. / (2 * np.pi * 20e-3)
    lifinfo = 2 * bw * (lifdata['infos'] / lifdata['spikes_sec']).mean(-1)[lifmask,:]
    fhninfo = 2 * bw * (fhndata['infos'] / fhndata['spikes_sec']).mean(-1)[fhnmask,:]

    legendstr = []
    for sigma in noise_targs:
        legendstr.append(r'$\sigma_\eta =$ ' + floattolatex10(sigma))

    plt.figure(6,figsize=(twocol_in,onecol_in))
    plt.clf()
    rows, cols = (1,2)
    ylims = [0,2]

    ####################
    plt.subplot(rows,cols,1)
    makesubplot(lifdata['bths'], lifinfo.T, plt.semilogx, *styleobj)
    plt.ylim(ylims)
    plt.xlabel(hetero_label)
    plt.ylabel(info_spike_label)
    titletext("LIF neurons")

    ax = plt.subplot(rows,cols,2)
    makesubplot(fhndata['bths'], fhninfo.T, plt.semilogx, *styleobj)
    plt.ylim(ylims)
    plt.xlabel(hetero_label)
    ax.set_yticklabels([])
    titletext("FHN neurons")
    legend(legendstr, 'upper left')

    tight_layout()
    plt.savefig(target)

#########################################################################################
### Mutual information contour plot (x-axis: heterogeneity, y-axis: noise)

def infocontourplot(lifdatafile, fhndatafile, target):
    data_lif = np.load(lifdatafile)
    data_fhn = np.load(fhndatafile)

    zlevels = np.linspace(0,2.0,21)
    zlabel = info_label

    def make_info_contour(ax, data, perspike=False):
        bias = data['bths']
        noise = data['Nstds']
        if not perspike:
            info = data['infos']
        else:
            bw = 1. / (2 * np.pi * 20e-3)
            info = 2*bw*data['infos'] / data['spikes_sec']

        ### smoothing
        X, Y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
        G = np.exp(-0.5*(X**2 + Y**2)/(0.75)**2)
        G /= G.sum()
        info_mean = sp.signal.convolve2d(info.mean(-1), G, mode='same', boundary='symm')

        ax.set_xscale('log')
        ax.set_yscale('log')

        if usecolor:
            contours = ax.contourf(bias, noise, info_mean, levels=zlevels)
            cbar = plt.colorbar(contours, ax=ax)
            cbar.set_label(zlabel)
        else:
            cs = ax.contour(bias, noise, info_mean, levels=zlevels, cmap=bw_cmap)
            ax.clabel(cs, zlevels[::3], inline=1, inline_spacing=1,
                      fontsize=textsize_mini, fmt="%0.2f")

    plt.figure(7,figsize=(twocol_in,twocol_in))
    plt.clf()

    rows = 2
    cols = 2
    init_subplots(rows, cols)

    ax = plt.subplot(rows,cols,1)
    make_info_contour(ax, data_lif)
    ax.xaxis.set_visible(0)
    ax.set_ylabel(noise_label)
    titletext("LIF neurons [bits]")

    ax = plt.subplot(rows,cols,2)
    make_info_contour(ax, data_fhn)
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)
    titletext("FHN neurons [bits]")

    ax = plt.subplot(rows,cols,3)
    make_info_contour(ax, data_lif, perspike=True)
    ax.set_xlabel(hetero_label)
    ax.set_ylabel(noise_label)
    titletext("LIF neurons [bits/spike]")

    ax = plt.subplot(rows,cols,4)
    make_info_contour(ax, data_fhn, perspike=True)
    ax.set_xlabel(hetero_label)
    ax.yaxis.set_visible(0)
    titletext("FHN neurons [bits/spike]")

    tight_layout(top=0.97, h_pad=1)
    plt.savefig(target)

################################################################################
### Mutual information plot (x-axis: noise, y-axis: info)

def infonoiseplot(lifdatafile, fhndatafile, target):
    lifdata = np.load(lifdatafile)
    fhndata = np.load(fhndatafile)

    ylims = [0,1.4]
    ylims2 = [0,1.4]
    legendstr = ['LIF neurons', 'FHN neurons']

    plt.figure(1,figsize=(twocol_in,onecol_in))
    plt.clf()

    rows = 1
    cols = 2

    lifbind = 0
    fhnbind = 0

    lifinfo = lifdata['infos'].mean(-1)[:,lifbind]
    fhninfo = fhndata['infos'].mean(-1)[:,fhnbind]
    bw = 1. / (2 * np.pi * 20e-3)
    lifinfo2 = 2 * bw * (lifdata['infos'] / lifdata['spikes_sec']).mean(-1)[:,lifbind]
    fhninfo2 = 2 * bw * (fhndata['infos'] / fhndata['spikes_sec']).mean(-1)[:,fhnbind]

    ####################
    plt.subplot(rows,cols,1)
    noise = lifdata['Nstds'].flatten()
    makesubplot(noise, np.array([lifinfo, fhninfo]).T, plt.semilogx, *styleobj)

    plt.ylim(ylims)
    plt.xlabel(noise_label)
    plt.ylabel(info_label)
    titletext("A")
    plt.legend(legendstr, loc='lower left', bbox_to_anchor=(0.02,0.02), frameon=False)

    ####################
    plt.subplot(rows,cols,2)
    noise = fhndata['Nstds'].flatten()
    makesubplot(noise, np.array([lifinfo2, fhninfo2]).T, plt.semilogx, *styleobj)

    plt.ylim(ylims2)
    plt.xlabel(noise_label)
    plt.ylabel(info_spike_label)
    titletext("B")

    tight_layout()
    plt.savefig(target)

################################################################################
### Phase plot

def phaseplot(lifphasefile, fhnphasefile, target):
    lifdata = np.load(lifphasefile)
    fhndata = np.load(fhnphasefile)

    ylims = [0,np.pi/2]

    legendstr = []
    for b, b2 in zip(lifdata['bths'], fhndata['bths']):
        assert b == b2
        if abs(b) < 1e-8:
            legendstr.extend( [r'$b_i =$ $0$'] )
        else:
            legendstr.extend( [r'$b_i \in [%0.2f,%0.2f]$' % (b,b)] )

    plt.figure(2,figsize=(twocol_in,onecol_in))
    plt.clf()
    rows = 1
    cols = 2
    fhn_ind, lif_ind = (2,1)

    ####################
    ax = plt.subplot(rows,cols,lif_ind)
    noise = lifdata['Nstds']
    phase = lifdata['phases'].mean(-1)
    makesubplot(noise, phase, plt.semilogx, *styleobj)

    plt.ylim(ylims)
    plt.xlabel(noise_label)
    plt.ylabel(phase_label)
    titletext("LIF neurons")
    legend(legendstr, loc='lower right')

    yticks = np.pi/8 * np.arange(5)
    yticklabels = [r'$0$', r'$\frac{1}{8}\pi$', r'$\frac{1}{4}\pi$',
                   r'$\frac{3}{8}\pi$', r'$\frac{1}{2}\pi$']
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=textsize_small+1)

    ####################
    ax = plt.subplot(rows,cols,fhn_ind)
    noise = fhndata['Nstds']
    phase = fhndata['phases'].mean(-1)
    makesubplot(noise, phase, plt.semilogx, *styleobj)

    plt.ylim(ylims)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    plt.xlabel(noise_label)
    titletext("FHN neurons")

    tight_layout()
    plt.savefig(target)

################################################################################
### raster plot to show synchronization

def syncrasterplot(firingfile, target):
    lifdata = np.load(firingfile)
    results = lifdata['noisehetero']
    Nstds = lifdata['Nstds']
    bths = lifdata['bths']

    plt.figure(8, figsize=(twocol_in, 1*twocol_in))
    plt.clf()
    rows = len(Nstds)
    cols = len(bths)

    Nlabels = ['low\nnoise', 'moderate\nnoise', 'high\nnoise']
    Hlabels = ['low\nheterogeneity', 'moderate\nheterogeneity', 'high\nheterogeneity']

    gs1 = gridspec.GridSpec(1, 1, bottom=0.75, top=0.97)
    gs2 = gridspec.GridSpec(3, 3, bottom=0.08, top=0.71)

    ### plot the input signal at the top
    ax1 = plt.subplot(gs1[0,0])
    ax1.plot(lifdata['t'], lifdata['S'][0], 'k-')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([-0.3, 0.3])
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('input signal')
    ax1.set_yticks(np.linspace(-0.2, 0.2, 5))

    ### plot the responses for various levels
    for i in xrange(rows):
        for j in xrange(cols):
            # ax = plt.subplot(rows+1, cols, (i+1)*cols + j + 1)
            ax = plt.subplot(gs2[i,j])

            ### plot spikes for each neuron
            times = results[i][j]
            times = times[:len(times)/2]

            height = 0.5
            for t in times:
                lines = ax.plot(t, height*np.ones(len(t)), 'k|')
                lines[0].set_markersize(1)
                lines[0].set_markeredgewidth(2)
                height += 1

            if i == len(Nstds) - 1:
                ax.set_xlabel('time [s]')
                ax.set_xticks(0.2*np.arange(5))
            else:
                ax.set_xticklabels([])

            if i == 0:
                ax.set_title(Hlabels[j], fontsize=textsize_title)

            if j == cols - 1:
                ax.set_ylabel('neuron number', labelpad=10)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
                ax.yaxis.label.set_horizontalalignment('center')
            else:
                ax.set_yticklabels([])

            if j == 0:
                ax.set_ylabel(Nlabels[i], fontsize=textsize_title, labelpad=15)
                ax.yaxis.label.set_rotation(0)
                ax.yaxis.label.set_verticalalignment('center')

            for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
                tick.tick1line.set_visible(0)
                tick.tick2line.set_visible(0)

            ax.set_ylim([0, len(times) + 1])

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    p = ax1.get_position()
    s = 0.5
    ax1.set_position([p.x0, p.y0 + (1-s)*p.height, p.width, s*p.height])

    # tight_layout(top=1)
    plt.savefig(target)

################################################################################
### Noisy tuning curves

def tuningnoisyplot(liftuningfile, fhntuningfile, target):
    lifdata = np.load(liftuningfile)
    fhndata = np.load(fhntuningfile)

    xlims = [-0.2,0.2]
    ylims = [0, 30]
    xlabel = r'input signal ($s(t)$)'
    ylabel = r'output firing rate [Hz]'
    rotation = 20

    def legend_string(Nstds):
        legendstr = []
        for Nstd in Nstds:
            if abs(Nstd) < 1e-8:
                legendstr.append(r'$\sigma_\eta =$ $0$')
            else:
                legendstr.append(r'$\sigma_\eta =$ $\mathrm{%s}$' % floattoe(Nstd))
        return legendstr

    plt.figure(4,figsize=(twocol_in,onecol_in))
    plt.clf()
    rows, cols = (1,2)
    fhn_ind, lif_ind = (2,1)

    ####################
    plt.subplot(rows,cols,lif_ind)
    Svals = lifdata['Svals']
    rates = lifdata['rates'].mean(-1)
    makesubplot(Svals, rates.T, plt.plot, *styleobj)

    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xticks(rotation=rotation, fontsize=textsize_mini)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    titletext("LIF neurons")
    legend(legend_string(lifdata['Nstds']), 'lower right')

    ####################
    plt.subplot(rows,cols,fhn_ind)
    Svals = fhndata['Svals']
    rates = fhndata['rates'].mean(-1)
    makesubplot(Svals, rates.T, plt.plot, *styleobj)

    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xticks(rotation=rotation, fontsize=textsize_mini)
    plt.gca().set_yticklabels([])
    plt.xlabel(xlabel)
    titletext("FHN neurons")
    legend(legend_string(fhndata['Nstds']), 'lower right')

    tight_layout()
    plt.savefig(target)

################################################################################
### Heterogeneous tuning curves

def tuningheteroplot(target):
    import colorsys

    ########## Tuning curve generation ##########
    alpha = 15
    tauRC = 20e-3
    tauRef = 33e-3
    Jth = 1

    xmin = -0.2
    xmax = 0.2
    xlims = (xmin, xmax)
    x = np.linspace( xmin, xmax, 100 )

    np.random.seed(2)
    bthmin = -0.15
    bthmax = 0.15

    ### keep generating biases until we have decent spacing between them
    mindiff = 0
    while mindiff < 0.001:
        bth = bthmin + (bthmax - bthmin)*np.random.rand(9)
        bth.sort()
        bth[0] = -0.15
        mindiff = np.min(np.diff(bth))

    b = Jth - alpha*bth
    nb = len(b)

    j = alpha*x[:,None] + b[None,:]
    a = np.zeros_like(j)
    a[j > Jth] = 1 / (tauRef - tauRC*np.log( 1 - Jth/j[j > Jth] ))


    ########## Plots ##########

    ### plot tuning curves together
    plt.figure(figsize=(twocol_in,onecol_in))
    m = 1
    n = 2

    ylims = (0,30)

    decoder = (1./nb) * np.ones(nb)

    lw = 1.5   # linewidth
    lw2 = 1.5

    if usecolor:
        sty = [styles[0]]
        colorhue = np.linspace(0,0.9,nb)
        colors1 = []
        colors2 = []
        for i in range(nb):
            colors1.append( colorsys.hsv_to_rgb( colorhue[i], 1.0, 0.8 ) )
            colors2.append( colorsys.hsv_to_rgb( colorhue[i], 0.4, 0.8 ) )
    else:
        sty = [styles[0], styles[1], styles[3]]
        numrep = int(np.ceil(nb / len(sty)))
        sty = sty * numrep
        colors1 = [[0.4]*3]*numrep + [[0.5]*3]*numrep + [[0.6]*3]*numrep
        colors2 = colors1

    if usecolor: save_dir = 'color/'
    else:        save_dir = 'bw/'

    ### individual tuning curves plot
    plt.subplot(m,n,1)

    for i in range(nb):
        l, = plt.plot(x, a[:,i], linewidth=lw2, color=colors1[i])
        l.set_dashes(sty[i])

    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xticks(rotation=30, fontsize=textsize_mini)

    plt.xlabel('input signal')
    plt.ylabel('firing rate [Hz]')
    titletext('LIF tuning curves')

    ### tuning curve sum plot
    plt.subplot(m,n,2)

    # stacked tuning curves
    decodermod = decoder.copy()
    for i in range(nb-1,-1,-1):
        mask = (x - bth[i] >= -0.01)
        l, = plt.plot(x[mask], np.dot(decodermod, a.T)[mask],
                     color=colors2[i], linewidth=lw2)
        l.set_dashes(sty[i])
        decodermod[i] = 0

    # dotted line on top
    plt.plot(x, np.dot(decoder,a.T), linestyle='-', color='k', linewidth=lw)

    # reference tuning curve
    l, = plt.plot(x, a[:,0], color='k', linewidth=lw)
    l.set_dashes(styles[1])

    # reference line
    pt1 = np.array([-0.15,0])
    pt2 = np.array([0.2,a[-1,0]])
    ptdiff = pt2 - pt1

    refline = ptdiff[1]/ptdiff[0] * (x - pt1[0]) + pt1[1]
    refline[x < pt1[0]] = pt1[1]
    l, = plt.plot(x, refline, linewidth=lw, color='k')
    l.set_dashes(styles[2])

    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xticks(rotation=30, fontsize=textsize_mini)

    plt.xlabel('input signal')
    plt.ylabel('population firing rate [Hz]')
    titletext('Population response')

    tight_layout()
    plt.savefig(target)
