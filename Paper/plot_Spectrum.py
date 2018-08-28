#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
#mpl.use('PDF')
import matplotlib.pyplot as plt
from math import atan2,degrees
from sys import path
import os
path.append('/home/vault/capm/sn0515/PhD/Th_U-Wire/Scripts/')
import script_plot as plot

BIGGER_SIZE = 18
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes

def main():
    # FINAL NETWORK
    Model_GOOD = "/180308-1100/180309-1055/180310-1553/180311-2206/180312-1917/180313-2220/"
    Epoch_GOOD = 99
    Source_GOOD = 'th'
    Position_GOOD = 'S5'

    # PREPROCESSING
    folderOUT = "/home/vault/capm/sn0515/PhD/Th_U-Wire/Paper/"
    Epoch_Th = str(Epoch_GOOD).zfill(3)
    folderIN_MC_Th = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model_GOOD + "/1validation-data/" + Source_GOOD + "ms-" + Position_GOOD + "/"
    fileIN_Th = "spectrum_events_" + Epoch_Th + "_" + Source_GOOD + "ms-" + Position_GOOD + ".p"

    folderIN_MC_Uni = '/home/vault/capm/sn0515/PhD/Th_U-Wire/Data_MC/Uniform_Wfs_SS+MS_S5_MC/'
    filesIN_Uni = [os.path.join(folderIN_MC_Uni, f) for f in os.listdir(folderIN_MC_Uni) if os.path.isfile(os.path.join(folderIN_MC_Uni, f))]

    print 'folderIN_MC_Uni\t', folderIN_MC_Uni
    # print 'filesIN_Uni\t', filesIN_Uni
    print 'folderIN_MC_Th\t', folderIN_MC_Th
    print 'fileIN_Th\t', fileIN_Th

    data_MC_Th = get_events(folderIN_MC_Th + fileIN_Th)['E_True']
    data_MC_Uni = get_events_hdf5(filesIN_Uni, folderOUT+'Uniform_data.p')

    print data_MC_Uni.shape
    print data_MC_Th.shape
    # exit()

    fileOUT = "spectrum_Uniform_228Th.pdf"

    plot_spectrum(data_TH=data_MC_Th, data_UNI=data_MC_Uni,
                  ylabel_TH='$^{228}$Th', ylabel_UNI='Uniform',
                  ycolor_TH='green', ycolor_UNI='blue', fOUT=(folderOUT + fileOUT))

    print '===================================== Program finished =============================='

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def get_events(fileIN):
    import cPickle as pickle
    try:
        return pickle.load(open(fileIN, "rb"))
    except IOError:
        print 'file not found' ; exit()

def get_events_hdf5(filesIN, fileOUT):
    import cPickle as pickle
    try:
        # raise
        return pickle.load(open(fileOUT, "rb"))['uniform']
    except:
        import h5py
        for idx, filename in enumerate(filesIN):
            f = h5py.File(str(filename), 'r')
            data_i = np.asarray(f['trueEnergy'])
            f.close()
            if idx == 0:
                data = data_i
            else:
                data = np.concatenate((data, data_i))
            print idx, data.shape
            # if idx >= 5: break
        data_dict = {'uniform': data}
        pickle.dump(data_dict, open(fileOUT, "wb"))
        return data

def plot_spectrum(data_TH, data_UNI, ylabel_TH, ylabel_UNI, ycolor_TH, ycolor_UNI, fOUT):
    # Create figure

    lowE = 700
    upE = 3100
    bins = 250

    hist_TH , bin_edges = np.histogram(data_TH, bins=bins, range=(lowE, upE), density=True)
    hist_UNI , bin_edges = np.histogram(data_UNI, bins=bins, range=(lowE, upE), density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.ion()

    fig, ax1 = plt.subplots(1, figsize=(8.5,5.66))

    ax1.set(ylabel='Probability')
    ax1.set(xlabel='True Energy [keV]')
    ax1.set_xlim([lowE, upE])
    ax1.set_ylim([5.e-6, 2.e-2])
    # ax1.set_ylim([1.e-4, 1.e-1])

    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    ax1.set_yscale('log')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.05)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    ax1.step([0.1], [0.1], label=ylabel_TH, color=ycolor_TH, where='mid')
    ax1.step(bin_centres, hist_UNI, label=ylabel_UNI, color=ycolor_UNI, where='mid')
    ax1.step(bin_centres, hist_TH, color=ycolor_TH, where='mid')
    ax1.legend(loc='best')
    plt.show()
    raw_input("")
    plt.savefig(fOUT)
    return

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()