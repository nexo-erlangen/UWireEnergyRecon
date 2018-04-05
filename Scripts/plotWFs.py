import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import h5py
import random
import os

note = 'Th-SSMS-Data'

def main():
    folderIN = '/home/woody/capm/sn0515/PhD/Th_U-Wire/Th228_Wfs_SS+MS_S5_Data_FullWfs/'
    # folderIN = '/home/vault/capm/sn0515/PhD/Th_U-Wire/ThMC/'
    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f))]
    number = 100
    generator = generate_event(files)
    for idx in range(number):
        print 'plot waveform \t', idx
        wf, _ = generator.next()
        if idx != 78: continue
        # if idx not in [53, 66]: continue
        # if idx == 53: ch = 13
        # if idx == 66: ch = 23
        plot_waveforms(wf, idx)
    return

def plot_waveforms(wf, idx):
    cut_1 = 512
    length = 1024
    cut_2 = cut_1 + length
    time = range(0, 2048)
    for j in range(76):
        plt.plot(time[ : cut_1], wf[ : cut_1 , j] + 20. * j, color='lightgray')
        plt.plot(time[ cut_1 : cut_2], wf[ cut_1 : cut_2, j] + 20. * j, color='k')
        plt.plot(time[ cut_2 : ], wf[ cut_2 :, j] + 20. * j, color='lightgray')
    plt.axvline(x=cut_1, lw=0, color='black')
    plt.axvline(x=cut_2, lw=0, color='black')
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('Amplitude + offset [a.u.]')
    # plt.ylabel('Amplitude + Offset')
    plt.xlim(xmin=0, xmax=2048)
    plt.ylim(ymin=-20, ymax=1520)
    # plt.axes().set_aspect(0.7)
    plt.axes().set_aspect(0.5)
    plt.yticks([])
    plt.savefig('/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/waveforms/' + str(idx) + '_wvf_' + str(note) + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()

    # for j in range(76):
    #     if j == ch: lw = 2
    #     else: lw = 1
    #     plt.plot(time[ cut_1 : cut_2], wf[ cut_1 : cut_2, j] + 20. * j, lw=lw, color='k')
    # plt.yticks([])
    # plt.xticks([])
    # plt.xlim(xmin=512, xmax=512+1024)
    # plt.ylim(ymin=-20, ymax=1520)
    # plt.axes().set_aspect(0.7)
    # plt.savefig('/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/waveforms/empty_' + str(idx) + '_wvf_' + str(note) + '.png', bbox_inches='tight')
    # plt.close()
    # plt.clf()

    # ax = plt.gca()
    # aspect = 1. / ax.get_data_ratio()
    aspect = 1024. / 76. / 1.4
    wf_crop = (np.swapaxes(wf, 0, 1))[ : , cut_1 : cut_2 , 0]
    abs_min = abs(np.min(wf_crop))
    abs_max = abs(np.max(wf_crop))
    if abs_max>abs_min:
        elev_max = abs_max
        elev_min = -abs_max
    else:
        elev_max = abs_min
        elev_min = -abs_min
    mid_val = 0
    logthresh = -0

    im = plt.imshow(wf_crop, origin='lower', aspect=aspect, vmin=elev_min, vmax=elev_max,
                    norm=mpl.colors.SymLogNorm(linthresh=(10**-logthresh), linscale=.1, vmin=elev_min, vmax=elev_max), cmap=plt.get_cmap('RdBu_r'))
                     #, clim=(elev_min, elev_max)) #, norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))

    maxlog = int(np.ceil(np.log10(elev_max)))
    minlog = int(np.ceil(np.log10(-elev_min)))

    # generate logarithmic ticks
    tick_locations = ([-(10 ** x) for x in xrange(minlog, -int(np.ceil(logthresh)) - 1, -1)]
                      + [0.0]
                      + [(10 ** x) for x in xrange(-int(np.ceil(logthresh)), maxlog + 1)])

    # cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=tick_locations) #, ticks=mpl.ticker.LogLocator(subs=range(2,10)))
    plt.yticks([])
    plt.xticks([])
    # cbar = plt.colorbar(im, fraction=0.025, pad=0.04)
    # cbar.set_label('Amplitude')
    # plt.xlabel('Time [$\mu$s]')
    # plt.ylabel('Channel')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/images/' + str(idx) + '_ims' + str(note) + '.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    return

def generate_event(files):
    start = 0
    length = 2048
    while 1:
        # random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            E_true_i = np.asarray(f.get('trueEnergy'))[:100]
            wfs_i = np.asarray(f.get('wfs'))[:100, :, start:start + length]
            wfs_gains = np.asarray(f.get('gains'))
            f.close()
            wfs_i = np.asarray(wfs_i / wfs_gains[:, None])
            wfs_i = np.swapaxes(wfs_i, 1, 2)
            wfs_i = wfs_i[..., np.newaxis]
            lst = range(len(E_true_i))
            # random.shuffle(lst)
            for i in lst:
                E_true = E_true_i[i]
                wfs = wfs_i[i]
                yield (wfs, E_true)

class MidpointNormalize(mpl.colors.Normalize):
    """
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def color(all, idx):
    cmap = mpl.cm.get_cmap('Dark2')
    norm = mpl.colors.Normalize(vmin=0.0, vmax=len(all) - 1)
    return cmap(norm(idx))

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()