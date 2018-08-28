#!/usr/bin/env python

import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.mlab as mlab
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
from sys import path
path.append('/home/vault/capm/sn0515/PhD/Th_U-Wire/Scripts')
import script_plot as plot
import scipy.special as scp

BIGGER_SIZE = 12
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes

def get_events(fileIN):
    import cPickle as pickle
    try:
        return pickle.load(open(fileIN, "rb"))
    except IOError:
        print 'file not found' ; exit()

def main():
    # FINAL NETWORK
    Model = "/180308-1100/180309-1055/180310-1553/180311-2206/180312-1917/180313-2220/"
    Epoch = 99
    Source = 'th'

    Multi = 'ms'
    Position = 'S5'
    Calibration = True
    # Calibration = False


    # PREPROCESSING
    Epoch = str(Epoch).zfill(3)
    folderOUT = "/home/vault/capm/sn0515/PhD/Th_U-Wire/Paper/"
    fileOUT = "energy_position_"+Epoch+"_"+Source+Multi+"_"+Position
    # MC
    folderIN = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model + "/1validation-data/" + Source + Multi + "-" + Position + "/"
    fileIN = "spectrum_events_" + Epoch + "_" + Source + Multi + "-" + Position + ".p"

    # DATA
    # folderIN = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model + "/0physics-data/" + Epoch + "/" + Source + "ms-" + Position + "/"
    # fileIN = "spectrum_events_" + Epoch + "_" + Source + "ms-" + Position + ".p"

    print 'folderIN\t', folderIN
    print 'fileIN\t', fileIN

    data = get_events(folderIN+fileIN)

    print data.keys()
    print data['posZ'].size

    if Calibration:
        fOUT_cal = "_calibrated"
        isSS = data['isSS'] == True
        for key in data.keys():
            if key == 'isSS': continue
            data[key] = {'SS': data[key][isSS],
                         'MS': data[key][np.invert(isSS)],
                         'SSMS': data[key]}
        for E_List_str in ['E_True', 'E_CNN', 'E_EXO']:
            if E_List_str != 'E_True':
                for Multi in ['SS', 'MS']:
                    if len(data[E_List_str][Multi]) != 0:
                        CalibrationFactor = plot.calibrate_spectrum(data=data[E_List_str][Multi], name='', peakpos=2614.5, isMC=True, fOUT=None, peakfinder='max')
                        data[E_List_str][Multi] = data[E_List_str][Multi] / CalibrationFactor
        for key in data.keys():
            if key == 'isSS': continue
            data[key]['SSMS'] = np.concatenate((data[key]['SS'],data[key]['MS']))
        # for Multi in ['SS', 'MS', 'SSMS']:
        for Multi in ['SS']:
            PlotTest(trE=data["E_True"][Multi],
                    prE=data["E_EXO"][Multi],
                    posX=data['posX'][Multi],
                     posY=data['posY'][Multi],
                     posZ=data['posZ'][Multi],
                    indE=data['reconEnergyInd'][Multi],
                    mEn=data['missedEnergy'][Multi],
                    mCl=data['missedCluster'][Multi],
                    fileOUT=folderOUT + fileOUT + fOUT_cal + "_Standard_" + Multi + ".pdf")
    else:
        print 'test'

    print '===================================== Program finished =============================='

def PlotTest(trE, prE, posX, posY, posZ, indE, mEn, mCl, fileOUT):
    pos = (posX, posY, posZ)
    th = (trE>2613) & (trE<2616)
    # trE, prE, indE = trE[th], prE[th], indE[th]
    # mEn, mCl = mEn[th], mCl[th]
    # posX, posY, posZ = posX[th], posY[th], posZ[th]

    z_bins = np.linspace(-200,200,40)

    # z_clu = np.asarray([
    #                        np.divide(float(mCl[
    #                                            (mCl > 50) & (pos >= z_bins[iBin]) & (pos < z_bins[iBin + 1])
    #                                            ].size),
    #                                  float(prE[
    #                                            (pos >= z_bins[iBin]) & (pos < z_bins[iBin + 1])
    #                                            ].size))
    #                        for iBin in range(len(z_bins[:-1]))])

    plt.clf()
    # make Figure
    fig = plt.figure()

    # set size of Figure
    fig.set_size_inches(w=15*0.8, h=10*0.8)

    # add Axes
    ax = [None]*3
    ax[0]  = fig.add_axes([0.10, 0.10, 0.26, 0.90])
    ax[1]  = fig.add_axes([0.40, 0.10, 0.26, 0.90], sharex=ax[0], sharey=ax[0])
    ax[2]  = fig.add_axes([0.70, 0.10, 0.26, 0.90], sharex=ax[0], sharey=ax[0])

    z_bins_c = (z_bins[:-1] + z_bins[1:]) / 2.
    label = ['x', 'y', 'z']
    color = ['blue', 'green', 'red']


    # plt.plot(z_bins_c, z_ind, marker='o', label='induction Tag')
    for i in range(3):
        print label[i]
        if i == 0:
            ax[0].set_xticks([-100, 0, 100])
            # ax[0].set_yticks([0, 5, 10])
            ax[0].set_xticklabels([-100, 0, 100])
            # ax[0].set_yticklabels([0, 5, 10])
            ax[0].set_xlim([-200, 200])
            ax[0].set_ylim([0, 13])
            ax[0].set_ylabel(r'Event Fraction [$\%$]', fontsize=16)
        else:
            plt.setp(ax[i].get_yticklabels(), visible=False)

        ax[i].grid()
        ax[i].set_xlabel(r'%s [mm]'%(label[i]), fontsize=16)

        for j, thresh in enumerate([30,40,50]):
            z_ene = np.asarray([
                                   np.divide(float(mEn[
                                                       (mEn > thresh) & (pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])
                                                       ].size),
                                             float(prE[
                                                       (pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])
                                                       ].size))
                                   for iBin in range(len(z_bins[:-1]))])
            z_ene_err = np.asarray([np.sqrt(
                np.divide(float(mEn[(mEn > thresh) & (pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])].size),
                          float(prE[(pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])].size) ** 2)
                +
                np.divide(float(mEn[(mEn > thresh) & (pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])].size) ** 2,
                          float(prE[(pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])].size) ** 3)
            )
                                    for iBin in range(len(z_bins[:-1]))])
            ene_mean = np.mean([np.divide(float(mEn[mEn > thresh].size), float(prE.size))])
            ax[i].axhline(y=100. * ene_mean, color='k')
            ax[i].fill_between(z_bins_c, 100. * ene_mean, 100. * z_ene, step='mid', alpha=0.4, color=color[j])
            if i == j:
                ax[i].errorbar(z_bins_c, 100.* z_ene, yerr=100. * z_ene_err, marker='o', label='> %s keV' % (str(thresh)), color=color[j])
            else:
                ax[i].errorbar(z_bins_c, 100. * z_ene, yerr=100. * z_ene_err, marker='o', color=color[j])

        z_ind = np.asarray([
                               np.divide(float(indE[
                                                   (indE>0) & (pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])
                                               ].size),
                                         float(prE[
                                                   (pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])
                                               ].size))
                               for iBin in range(len(z_bins[:-1]))])
        z_ind_err = np.asarray([np.sqrt(
            np.divide(float(indE[(indE > thresh) & (pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])].size),
                      float(prE[(pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])].size) ** 2)
            +
            np.divide(float(indE[(indE > thresh) & (pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])].size) ** 2,
                      float(prE[(pos[i] >= z_bins[iBin]) & (pos[i] < z_bins[iBin + 1])].size) ** 3)
        )
                                for iBin in range(len(z_bins[:-1]))])

        ind_mean = np.mean([np.divide(float(indE[indE>0].size),float(prE.size))])
        ax[i].axhline(y=100. * ind_mean, color='k')
        ax[i].fill_between(z_bins_c, 100. * ind_mean, 100. * z_ind, step='mid', alpha=0.4, color='k')
        if i == 1:
            ax[i].errorbar(z_bins_c, 100. * z_ind, yerr=100. * z_ind_err, marker='o', label='Ind Tag', color='k')
        else:
            ax[i].errorbar(z_bins_c, 100. * z_ind, yerr=100. * z_ind_err, marker='o', color='k')


        ax[i].legend(loc="upper center", fontsize=14)
        if i == 1:
            ax[i].set_title("missed Energy & Induction Tag", fontsize=18)
    # ax[1].set_yscale("log", nonposy='clip')
    fig.savefig(fileOUT, bbox_inches='tight')
    # fig.savefig(fileOUT[:-4] + "_40" + fileOUT[-4:], bbox_inches='tight')


def std_uncertainty(data):
    n = float(data.size)
    if n == 0.0: return 0.0
    return np.std(data) * np.sqrt(np.exp(1.) * (1. - 1. / n) ** (n - 1.) - 1.)

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()