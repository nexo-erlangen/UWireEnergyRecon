#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
#mpl.use('PDF')
import matplotlib.pyplot as plt
from math import atan2,degrees
from sys import path
path.append('/home/vault/capm/sn0515/PhD/Th_U-Wire/Scripts/')
import script_plot as plot

BIGGER_SIZE = 18
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes

def main():
    # FINAL NETWORK
    Model_GOOD = "/180802-1535/180803-1159/"
    Epoch_GOOD = 80
    Source_GOOD = 'ga'
    Position_GOOD = 'S5'

    # NETWORK W/ OVERTRAINING
    Model_BAD = "/180308-1102/180309-1055/180310-1553/180311-2201/180312-1917/"
    Epoch_BAD = 149
    Source_BAD = 'ga'
    Position_BAD = 'S5'

    Calibration = True

    # PREPROCESSING
    folderOUT = "/home/vault/capm/sn0515/PhD/Th_U-Wire/Paper/"
    Epoch_BAD = str(Epoch_BAD).zfill(3)
    Epoch_GOOD = str(Epoch_GOOD).zfill(3)
    folderIN_MC_BAD = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model_BAD + "/1validation-data/" + Source_BAD + "ms-" + Position_BAD + "/"
    folderIN_MC_GOOD = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model_GOOD + "/1validation-data/" + Source_GOOD + "ms-" + Position_GOOD + "/"
    fileIN_GOOD = "spectrum_events_" + Epoch_GOOD + "_" + Source_GOOD + "ms-" + Position_GOOD + ".p"
    fileIN_BAD = "spectrum_events_" + Epoch_BAD + "_" + Source_BAD + "ms-" + Position_BAD + ".p"

    print 'folderIN_MC_BAD\t', folderIN_MC_BAD
    print 'fileIN_BAD\t', fileIN_BAD
    print 'folderIN_MC_GOOD\t', folderIN_MC_GOOD
    print 'fileIN_GOOD\t', fileIN_GOOD

    # data_MC_BAD = get_events(folderIN_MC_BAD + fileIN_BAD)
    data_MC_GOOD = get_events(folderIN_MC_GOOD + fileIN_GOOD)

    # print data_MC_BAD["E_True"].shape, data_MC_BAD["E_CNN"].shape, data_MC_BAD["E_CNN"].shape
    print data_MC_GOOD["E_True"].shape, data_MC_GOOD["E_CNN"].shape, data_MC_GOOD["E_CNN"].shape

    if Calibration:
        # data_BAD = doCalibration(data_MC=data_MC_BAD)
        data_GOOD= doCalibration(data_MC=data_MC_GOOD)

        # for Multi in ['SS', 'MS', 'SSMS']:
        for Multi in ['SSMS', 'calib_SSMS']:
            plot_predict(x_bad=data_GOOD["E_True"][Multi], y_bad=data_GOOD["E_CNN"][Multi],
                         xlabel='True', ylabel_bad='DNN (Uniform training)',
                         ycolor_bad='green',
                         fOUT=(folderOUT + "spectrum_" + Source_BAD + "ms_ConvNN_" + Epoch_BAD + "_" + Multi + ".pdf"))
            plot_predict(x_bad=data_GOOD["E_True"][Multi], y_bad=data_GOOD["E_EXO"][Multi],
                         xlabel='MC', ylabel_bad='Good Conventional',
                         ycolor_bad='green',
                         fOUT=(folderOUT + "spectrum_" + Source_BAD + "ms_Standard_" + Epoch_BAD + "_" + Multi + ".pdf"))
            # plot_predict(x_bad=data_BAD["E_True"][Multi], y_bad=data_BAD["E_CNN"][Multi],
            #              x_good=data_GOOD["E_True"][Multi], y_good=data_GOOD["E_CNN"][Multi],
            #              xlabel='True', ylabel_bad='DNN   ($^{228}$Th training)', ylabel_good='DNN (Uniform training)',
            #              ycolor_bad='green', ycolor_good='blue',
            #              fOUT=(folderOUT + "spectrum_" + Source_BAD + "ms_ConvNN_" + Epoch_BAD + "_" + Multi + ".pdf"))
            # plot_predict(x_bad=data_BAD["E_True"][Multi], y_bad=data_BAD["E_EXO"][Multi],
            #              x_good=data_GOOD["E_True"][Multi], y_good=data_GOOD["E_EXO"][Multi],
            #              xlabel='MC', ylabel_bad='Bad Conventional', ylabel_good='Good Conventional',
            #              ycolor_bad='green', ycolor_good='firebrick',
            #              fOUT=(
            #              folderOUT + "spectrum_" + Source_BAD + "ms_Standard_" + Epoch_BAD + "_" + Multi + ".pdf"))
    else:
        fileOUT_MC_CNN = "spectrum_" + Source_BAD + "ms_ConvNN_" + Epoch_BAD + ".pdf"
        fileOUT_MC_Std = "spectrum_" + Source_BAD + "ms_Standard_" + Epoch_BAD + ".pdf"

        # plot_predict(x_bad=data_MC_BAD["E_True"], y_bad=data_MC_BAD["E_CNN"],
        #              x_good=data_MC_GOOD["E_True"], y_good=data_MC_GOOD["E_CNN"],
        #              xlabel='MC', ylabel_bad='Bad Neural Network', ylabel_good='Good Neural Network',
        #              ycolor_bad='green', ycolor_good='blue', fOUT=(folderOUT + fileOUT_MC_CNN))
        # plot_predict(x_bad=data_MC_BAD["E_True"], y_bad=data_MC_BAD["E_EXO"],
        #              x_good=data_MC_GOOD["E_True"], y_good=data_MC_GOOD["E_EXO"],
        #              xlabel='MC', ylabel_bad='Bad Conventional', ylabel_good='Good Conventional',
        #              ycolor_bad='firebrick', ycolor_good='blue', fOUT=(folderOUT + fileOUT_MC_Std))

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

def doCalibration(data_MC):
    data = {}
    for E_List_str in ['E_CNN', 'E_EXO', 'E_True']:
        data[E_List_str] = {'SS': data_MC[E_List_str][data_MC['isSS'] == True],
                            'MS': data_MC[E_List_str][data_MC['isSS'] == False],
                            'SSMS': data_MC[E_List_str]}
        for Multi in ['SS', 'MS']:
            CalibrationFactor = 1.0
            if len(data[E_List_str][Multi]) != 0:
                if E_List_str != 'E_True':
                    CalibrationFactor = plot.calibrate_spectrum(data=data[E_List_str][Multi], name='', isMC=True,
                                                                peakpos=2614.5, fOUT=None, peakfinder='max')
            data[E_List_str]['calib_' + Multi] = data[E_List_str][Multi] / float(CalibrationFactor)
        data[E_List_str]['calib_SSMS'] = np.concatenate((data[E_List_str]['calib_SS'], data[E_List_str]['calib_MS']))
    return data

# def plot_predict(x_bad, y_bad, x_good, y_good, xlabel, ylabel_bad, ylabel_good, ycolor_bad, ycolor_good, fOUT):
def plot_predict(x_bad, y_bad, xlabel, ylabel_bad, ycolor_bad, fOUT):
    if x_bad.size != y_bad.size: print 'arrays not same length. Press key' ; raw_input('')
    # if x_good.size != y_good.size: print 'arrays not same length. Press key'; raw_input('')
    # if sorted(x_bad) != sorted(x_good): print 'x arrays does not contain the same events. Press key'; raw_input('')

    # Create figure
    histX_bad , bin_edges = np.histogram(x_bad , bins=1200, range=(0, 12000), density=False)
    histY_bad , bin_edges = np.histogram(y_bad , bins=1200, range=(0, 12000), density=False)
    # histY_good, bin_edges = np.histogram(y_good, bins=1200, range=(0, 12000), density=False)
    norm_factor = float(len(x_bad))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    dE_bad = y_bad - x_bad
    lowE = 1000
    upE = 3100
    resE = 200
    gridsize = 200
    extent2 = [lowE, upE, -resE, resE]
    # plt.ion()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[2, 1]}, figsize=(8.5,8.5)) #, sharex=True) #, gridspec_kw = {'height_ratios':[3, 1]})

    ax2.axhline(y=0.0, ls='--', lw=2, color='black')

    ax1.set(ylabel='Probability')
    # ax2.set(xlabel=xlabel + ' Energy [keV]', ylabel='Residual [keV]')
    # ax2.set(xlabel=xlabel + ' Energy [keV]', ylabel='(%s - %s) [keV]' % ('DNN ($^{228}$Th)', xlabel))
    ax2.set(xlabel=xlabel + ' Energy [keV]', ylabel='(%s - %s) [keV]' % ('DNN (Uni)', xlabel))
    ax1.set_xlim([lowE, upE])
    ax1.set_ylim([10.e-5, 10.e-2])
    ax2.set_ylim([-resE, resE])

    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    ax2.xaxis.grid(True)
    ax2.yaxis.grid(True)
    ax1.set_yscale('log')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.05)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp(ax2, yticks=[-100, 0, 100])


    ax1.fill_between(bin_centres, 0.0, histX_bad / norm_factor, facecolor='black', alpha=0.2, interpolate=True)
    ax1.plot(bin_centres, histX_bad / norm_factor, label=xlabel, color='k', lw=0.5)
    ax1.step(bin_centres, histY_bad / norm_factor, label=ylabel_bad , color=ycolor_bad , where='mid')
    # ax1.step(bin_centres, histY_good/ norm_factor, label=ylabel_good, color=ycolor_good, where='mid')
    ax2.hexbin(x_bad, dE_bad, bins='log', extent=extent2, gridsize=(gridsize,gridsize/((upE-lowE)/(2*resE))), mincnt=1, cmap=plt.get_cmap('viridis'), linewidths=0.1)
    ax1.legend(loc='lower left')
    # plt.show()
    # raw_input("")
    plt.savefig(fOUT)
    return

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()