#!/usr/bin/env python

import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.mlab as mlab
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
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


    # PREPROCESSING
    Epoch = str(Epoch).zfill(3)
    folderOUT = "/home/vault/capm/sn0515/PhD/Th_U-Wire/Paper/"
    fileOUT = "induction_"+Epoch+"_"+Source+Multi+"_"+Position
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
                        if E_List_str == 'E_EXO':
                            data["missedEnergy"][Multi] = data["missedEnergy"][Multi] / CalibrationFactor
                            data["missedCluster"][Multi] = data["missedCluster"][Multi] / CalibrationFactor
                            data["reconEnergyInd"][Multi] = data["reconEnergyInd"][Multi] / CalibrationFactor
        for key in data.keys():
            if key == 'isSS': continue
            data[key]['SSMS'] = np.concatenate((data[key]['SS'],data[key]['MS']))
        # for Multi in ['SS']:
        for Multi in ['SS', 'MS', 'SSMS']:
            doPlot(trE=data["E_True"][Multi],
                   prEXO=data["E_EXO"][Multi],
                   prCNN=data["E_CNN"][Multi],
                   mEn=data["missedEnergy"][Multi],
                   rEnInd=data["reconEnergyInd"][Multi],
                   mCl=data["missedCluster"][Multi],
                   fileOUT=folderOUT+fileOUT+fOUT_cal+"_"+Multi+".pdf")
    else:
        isSS = data['isSS'] == True
        i = 50000
        doPlot(trE=data["E_True"][isSS][:i],
               prEXO=data["E_EXO"][isSS][:i],
               prCNN=data["E_CNN"][isSS][:i],
               mEn=data["missedEnergy"][isSS][:i],
               rEnInd=data["reconEnergyInd"][isSS][:i],
               mCl=data["missedCluster"][isSS][:i],
               fileOUT=folderOUT+fileOUT+".pdf")

    print '===================================== Program finished =============================='

def doPlot(trE, prEXO, prCNN, mEn, rEnInd, mCl, fileOUT):
    # Plot Range
    limit = 500
    bins = 200

    resEXO = prEXO- trE
    resCNN = prCNN - trE

    norm = float(resEXO[abs(resEXO) <= limit].size)

    hist_resEXO, bin_edges = np.histogram(resEXO, bins=bins, range=(-limit, limit))
    hist_resCNN, bin_edges = np.histogram(resCNN, bins=bins, range=(-limit, limit))
    hist_Ind, bin_edges = np.histogram(resEXO[(rEnInd > 0)], bins=bins, range=(-limit, limit))
    hist_mEn, bin_edges = np.histogram(resEXO[(mEn > 50) | (mCl > 50)], bins=bins, range=(-limit, limit))
    hist_resEXOREMAIN, bin_edges = np.histogram(resEXO[~(rEnInd > 0)], bins=bins, range=(-limit, limit))
    hist_resEXOCOMBINE, bin_edges = np.histogram(resEXO[~((rEnInd > 0) | (mEn > 50) | (mCl > 50))],
                                                 bins=bins, range=(-limit, limit))

    # cmap = plt.get_cmap('Reds')
    # cNorm = colors.Normalize(vmin=0, vmax=100)
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    plt.clf()
    plt.step(bin_edges[:-1], hist_resEXO / norm, label='EXO init', lw=2, color='k')
    plt.fill_between(bin_edges[:-1], 0.0, hist_Ind / norm, step='pre', label='Ind Tag', alpha=0.4, facecolor='green', color='green')
    plt.fill_between(bin_edges[:-1], 0.0, hist_mEn / norm, step='pre', label='missedEnergy', alpha=0.4, facecolor='red', color='red')
    plt.step(bin_edges[:-1], hist_resEXOREMAIN / norm, label='EXO w/o Ind Tag', lw=2, color='green')
    plt.step(bin_edges[:-1], hist_resEXOCOMBINE / norm, label='EXO final', lw=2, color='red')
    # for i in np.linspace(0, 100, 6):
    #     print i
    #     hist_mEn, bin_edges = np.histogram(resEXO[mEn > i], bins=bins, range=(-limit, limit))
    #     hist_mCl, bin_edges = np.histogram(resEXO[mCl > i], bins=bins, range=(-limit, limit))
    #     hist_resEXOREMAIN, bin_edges = np.histogram(resEXO[~((rEnInd > 0) | (mEn > i) | (mCl > i))],
    #                                                 bins=bins, range=(-limit, limit))
    #     colorVal = scalarMap.to_rgba(i)
    #     plt.fill_between(bin_edges[:-1], 0.0, hist_mCl / norm, step='pre', alpha=0.8, facecolor=colorVal)
    #     plt.fill_between(bin_edges[:-1], 0.0, hist_mEn / norm, step='pre', alpha=0.4, facecolor=colorVal)
    #     plt.step(bin_edges[:-1], hist_resEXOREMAIN / norm, color=colorVal)
    plt.step(bin_edges[:-1], hist_resCNN / norm, label='DNN', lw=2, color='dodgerblue')
    plt.axvline(x=0.0, lw=2, color='k')

    # scalarMap._A = []
    # cbar = plt.colorbar(scalarMap, fraction=0.025, pad=0.04)  # , ticks=mpl.ticker.LogLocator(subs=range(10)))
    # cbar.set_label('Missed Energy threshold')

    plt.gca().set_yscale('log')
    plt.xlim(xmin=-300, xmax=300)
    plt.ylim(ymin=5e-6, ymax=1.5e-1)
    plt.xlabel('Residual [keV]')
    plt.ylabel('Probability')
    plt.legend(loc="upper right")
    plt.savefig(fileOUT, bbox_inches='tight')
    # plt.show()
    # raw_input("")


    # ========================================================================================


    th = (trE>2613) & (trE<2616)
    trE, prEXO, prCNN = trE[th], prEXO[th], prCNN[th]
    mEn, rEnInd, mCl = mEn[th], rEnInd[th], mCl[th]

    norm = float(prEXO[abs(prEXO)-2615<=300].size)

    hist_resEXO, bin_edges = np.histogram(prEXO, bins=100, range=(-300+2615, 300+2615))
    hist_resCNN, bin_edges = np.histogram(prCNN, bins=100, range=(-300+2615, 300+2615))
    hist_Ind, bin_edges = np.histogram(prEXO[(rEnInd > 0)], bins=100, range=(-300+2615, 300+2615))
    hist_mEn, bin_edges = np.histogram(prEXO[(mEn > 50) | (mCl > 50)], bins=100, range=(-300+2615, 300+2615))
    # hist_mCl, bin_edges = np.histogram(prEXO[mCl > 50], bins=100, range=(-300 + 2615, 300 + 2615))
    hist_resEXOREMAIN, bin_edges = np.histogram(prEXO[~(rEnInd > 0)], bins=100, range=(-300 + 2615, 300 + 2615))
    hist_resEXOCOMBINE, bin_edges = np.histogram(prEXO[~((rEnInd > 0) | (mEn > 50) | (mCl > 50))],
                                                 bins=100, range=(-300 + 2615, 300 + 2615))

    # cmap = plt.get_cmap('Reds')
    # cNorm = colors.Normalize(vmin=0, vmax=100)
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    plt.clf()
    plt.step(bin_edges[:-1], hist_resEXO / norm, label='EXO init', lw=2, color='k')
    plt.fill_between(bin_edges[:-1], 0.0, hist_Ind / norm, step='pre', label='Ind Tag', alpha=0.4, facecolor='green', color='green')
    plt.fill_between(bin_edges[:-1], 0.0, hist_mEn / norm, step='pre', facecolor='white', color='white')
    plt.fill_between(bin_edges[:-1], 0.0, hist_mEn / norm, step='pre', label='missedEnergy', alpha=0.4, facecolor='red', color='red')
    plt.step(bin_edges[:-1], hist_resEXOREMAIN / norm, label='EXO w/o Ind Tag', lw=2, color='green')
    plt.step(bin_edges[:-1], hist_resEXOCOMBINE / norm, label='EXO final', lw=2, color='red')
    # for i in np.linspace(100,0,6):
    #     print i
    #     # hist_mEn, bin_edges = np.histogram(prEXO[mEn > i], bins=100, range=(-300 + 2615, 300 + 2615))
    #     # hist_mCl, bin_edges = np.histogram(prEXO[mCl > i], bins=100, range=(-300 + 2615, 300 + 2615))
    #     hist_resEXOREMAIN, bin_edges = np.histogram(prEXO[~((rEnInd > 0) | (mEn > i) | (mCl > i))],
    #                                                 bins=100, range=(-300 + 2615, 300 + 2615))
    #     colorVal = scalarMap.to_rgba(i)
    #     # plt.fill_between(bin_edges[:-1], 0.0, hist_mCl / norm, step='pre', alpha=0.8, facecolor=colorVal)
    #     # plt.fill_between(bin_edges[:-1], 0.0, hist_mEn / norm, step='pre', alpha=0.4, facecolor=colorVal)
    #     plt.step(bin_edges[:-1], hist_resEXOREMAIN / norm, color=colorVal)
    plt.step(bin_edges[:-1], hist_resCNN / norm, label='DNN', lw=2, color='dodgerblue')
    plt.axvline(x=2614.5, lw=2, color='k')

    # scalarMap._A = []
    # cbar = plt.colorbar(scalarMap, fraction=0.025, pad=0.04) #, ticks=mpl.ticker.LogLocator(subs=range(10)))
    # cbar.set_label('Missed Energy threshold')

    plt.gca().set_yscale('log')
    plt.xlim(xmin=2300, xmax=2800)
    plt.ylim(ymin=5e-5, ymax=1.5e-1)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.legend(loc="upper left")
    plt.savefig(fileOUT[:-4] + "_Th228" + fileOUT[-4:], bbox_inches='tight')
    # plt.show()





def std_uncertainty(data):
    n = float(data.size)
    return np.std(data)*np.sqrt(np.exp(1.) * (1. - 1. / n)**(n - 1.) - 1.)


# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()