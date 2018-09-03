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

# BIGGER_SIZE = 16
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes

def get_events(fileIN):
    import cPickle as pickle
    try:
        return pickle.load(open(fileIN, "rb"))
    except IOError:
        print 'file not found' ; exit()

def main():
    # FINAL NETWORK
    Model = "/180802-1535/180803-1159/"
    Epoch = 80
    Source = 'th'

    Multi = 'ms'
    Position = 'S5'
    Calibration = True
    # Calibration = False


    # PREPROCESSING
    Epoch = str(Epoch).zfill(3)
    folderOUT = "/home/vault/capm/sn0515/PhD/Th_U-Wire/Paper/"
    fileOUT = "position_"+Epoch+"_"+Source+Multi+"_"+Position
    # MC
    # folderIN = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model + "/1validation-data/" + Source + Multi + "-" + Position + "/"
    # fileIN = "spectrum_events_" + Epoch + "_" + Source + Multi + "-" + Position + ".p"

    # DATA
    folderIN = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model + "/0physics-data/" + Epoch + "/" + Source + "ms-" + Position + "/"
    fileIN = "spectrum_events_" + Epoch + "_" + Source + "ms-" + Position + ".p"

    print 'folderIN\t', folderIN
    print 'fileIN\t', fileIN

    data = get_events(folderIN+fileIN)

    print data.keys()
    print data['posZ'].size

    # for i in xrange(100):
    #     print data['E_EXO'][i], data['E_EXOPur'][i], data['E_EXO'][i]- data['E_EXOPur'][i]
    # exit()

    if Calibration:
        fOUT_cal = "_calibrated"
        isSS = data['isSS'] == True
        for key in data.keys():
            if key == 'isSS': continue
            data[key] = {'SS': data[key][isSS],
                         'MS': data[key][np.invert(isSS)],
                         'SSMS': data[key]}
        for E_List_str in ['E_True', 'E_CNN', 'E_EXO']:
        # for E_List_str in ['E_True', 'E_EXOPur', 'E_EXO']:
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
        #     doPlot(trE=data["E_True"][Multi],
        #            prE=data["E_CNN"][Multi],
        #            posX=data['posX'][Multi],
        #            posY=data['posY'][Multi],
        #            posZ=data['posZ'][Multi],
        #            labeltr='True', labelpr='DNN',
        #            fileOUT=folderOUT+fileOUT+fOUT_cal+"_ConvNN_"+Multi+".pdf")
        #     doPlot(trE=data["E_True"][Multi],
        #            prE=data["E_EXO"][Multi],
        #            posX=data['posX'][Multi],
        #            posY=data['posY'][Multi],
        #            posZ=data['posZ'][Multi],
        #            labeltr = 'True', labelpr = 'Recon',
        #            fileOUT=folderOUT+fileOUT+fOUT_cal+"_Standard_"+Multi+".pdf")
            doPlot(trE=data["E_CNN"][Multi],
                   prE=data["E_EXO"][Multi],
                   posX=data['posX'][Multi],
                   posY=data['posY'][Multi],
                   posZ=data['posZ'][Multi],
                   labeltr = 'DNN', labelpr = 'Recon',
                   fileOUT=folderOUT+fileOUT+fOUT_cal+"_Data_"+Multi+".pdf")
            # doPlot(trE=data["E_EXO"][Multi],
            #        prE=data["E_EXOPur"][Multi],
            #        posX=data['posX'][Multi],
            #        posY=data['posY'][Multi],
            #        posZ=data['posZ'][Multi],
            #        labeltr='EXO', labelpr='EXOPur',
            #        fileOUT=folderOUT + fileOUT + fOUT_cal + "_Data_" + Multi + ".pdf")
    else:
        isSS = data['isSS'] == True
        doPlot(trE=data["E_True"][isSS],
               prE=data["E_CNN"][isSS],
               posX=data['posX'][isSS],
               posY=data['posY'][isSS],
               posZ=data['posZ'][isSS],
               labeltr='True', labelpr='DNN',
               fileOUT=folderOUT+fileOUT+"_ConvNN.pdf")
        doPlot(trE=data["E_True"][isSS],
               prE=data["E_EXO"][isSS],
               posX=data['posX'][isSS],
               posY=data['posY'][isSS],
               posZ=data['posZ'][isSS],
               labeltr='True', labelpr='Recon',
               fileOUT=folderOUT+fileOUT + "_Standard.pdf")

    print '===================================== Program finished =============================='

def doPlot(trE, prE, posX, posY, posZ, labeltr, labelpr, fileOUT):
    # Plot Range
    limit_Res = 150
    limit_Pos = 200
    pos_bins = 100
    resE = prE - trE

    hist1D_X, bin_edges = np.histogram(posX, range=(-limit_Pos, limit_Pos), bins=pos_bins, normed=True)
    hist1D_Y, bin_edges = np.histogram(posY, range=(-limit_Pos, limit_Pos), bins=pos_bins, normed=True)
    hist1D_Z, bin_edges = np.histogram(posZ, range=(-limit_Pos, limit_Pos), bins=pos_bins, normed=True)

    mean_X = np.asarray([np.mean(resE[(abs(resE) <= limit_Res) & (posX >= bin_edges[iBin]) & (posX < bin_edges[iBin + 1])])
                            for iBin in range(len(bin_edges[:-1]))])
    mean_Y = np.asarray([np.mean(resE[(abs(resE) <= limit_Res) & (posY >= bin_edges[iBin]) & (posY < bin_edges[iBin + 1])])
                          for iBin in range(len(bin_edges[:-1]))])
    mean_Z = np.asarray([np.mean(resE[(abs(resE) <= limit_Res) & (posZ >= bin_edges[iBin]) & (posZ < bin_edges[iBin + 1])])
                          for iBin in range(len(bin_edges[:-1]))])
    std_X = np.asarray([
                          np.std(
                              resE[
                                  (abs(resE) <= limit_Res) & (posX >= bin_edges[iBin]) & (posX < bin_edges[iBin + 1])
                              ]
                          )
                          for iBin in range(len(bin_edges[:-1]))
                          ])
    std_Y = np.asarray([np.std(resE[(abs(resE) <= limit_Res) & (posY >= bin_edges[iBin]) & (posY < bin_edges[iBin + 1])])
                         for iBin in range(len(bin_edges[:-1]))])
    std_Z = np.asarray([np.std(resE[(abs(resE) <= limit_Res) & (posZ >= bin_edges[iBin]) & (posZ < bin_edges[iBin + 1])])
                         for iBin in range(len(bin_edges[:-1]))])
    std_unc_X = np.asarray([std_uncertainty(resE[(abs(resE) <= limit_Res) & (posX >= bin_edges[iBin]) & (posX < bin_edges[iBin + 1])])
                            for iBin in range(len(bin_edges[:-1]))])
    std_unc_Y = np.asarray([std_uncertainty(resE[(abs(resE) <= limit_Res) & (posY >= bin_edges[iBin]) & (posY < bin_edges[iBin + 1])])
                            for iBin in range(len(bin_edges[:-1]))])
    std_unc_Z = np.asarray([std_uncertainty(resE[(abs(resE) <= limit_Res) & (posZ >= bin_edges[iBin]) & (posZ < bin_edges[iBin + 1])])
                            for iBin in range(len(bin_edges[:-1]))])
    N_X = np.asarray([float(resE[(abs(resE) <= limit_Res) & (posX >= bin_edges[iBin]) & (posX < bin_edges[iBin + 1])].size)
                      for iBin in range(len(bin_edges[:-1]))])
    N_Y = np.asarray([float(resE[(abs(resE) <= limit_Res) & (posY >= bin_edges[iBin]) & (posY < bin_edges[iBin + 1])].size)
                      for iBin in range(len(bin_edges[:-1]))])
    N_Z = np.asarray([float(resE[(abs(resE) <= limit_Res) & (posZ >= bin_edges[iBin]) & (posZ < bin_edges[iBin + 1])].size)
                      for iBin in range(len(bin_edges[:-1]))])
    weights_X = np.asarray([1. / hist1D_X[np.argmax(bin_edges >= p) - 1]
                            if abs(p) <= max(bin_edges) else 0.0 for p in posX])
    weights_Y = np.asarray([1. / hist1D_Y[np.argmax(bin_edges >= p) - 1]
                            if abs(p) <= max(bin_edges) else 0.0 for p in posY])
    weights_Z = np.asarray([1. / hist1D_Z[np.argmax(bin_edges >= p) - 1]
                            if abs(p) <= max(bin_edges) else 0.0 for p in posZ])

    hist2D_X, xbins, ybins = np.histogram2d(posX, resE, weights=weights_X, range=[[-limit_Pos, limit_Pos], [-limit_Res, limit_Res]],
                                            bins=pos_bins, normed=True)
    hist2D_Y, xbins, ybins = np.histogram2d(posY, resE, weights=weights_Y, range=[[-limit_Pos, limit_Pos], [-limit_Res, limit_Res]],
                                            bins=pos_bins, normed=True)
    hist2D_Z, xbins, ybins = np.histogram2d(posZ, resE, weights=weights_Z, range=[[-limit_Pos, limit_Pos], [-limit_Res, limit_Res]],
                                            bins=pos_bins, normed=True)

    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = "auto" #(xbins.max() - xbins.min()) / (ybins.max() - ybins.min())
    sig = 2
    color = 'firebrick' #'crimson'
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.

    plt.clf()
    # make Figure
    fig = plt.figure()

    # set size of Figure
    fig.set_size_inches(w=12*0.8, h=7*0.8)

    # add Axes
    ax1  = fig.add_axes([0.10, 0.10, 0.21, 0.60])
    ax11 = fig.add_axes([0.10, 0.75, 0.21, 0.20], sharex=ax1)
    ax2  = fig.add_axes([0.34, 0.10, 0.21, 0.60], sharex=ax1, sharey=ax1)
    ax21 = fig.add_axes([0.34, 0.75, 0.21, 0.20], sharex=ax2, sharey=ax11)
    ax3  = fig.add_axes([0.58, 0.10, 0.21, 0.60], sharex=ax1, sharey=ax1)
    ax31 = fig.add_axes([0.58, 0.75, 0.21, 0.20], sharex=ax3, sharey=ax11)
    ax41 = fig.add_axes([0.82, 0.10, 0.13, 0.60], sharey=ax1)


    ax1.set_xticks([-100,0,100])
    ax1.set_yticks([-100, -50, 0, 50, 100])
    ax1.set_xticklabels([-100, 0, 100])
    ax1.set_yticklabels([-100, -50, 0, 50, 100])
    ax1.set_xlim([-limit_Pos, limit_Pos])
    ax1.set_ylim([-limit_Res, limit_Res])
    ax1.grid()
    ax1.set_ylabel(r'$\mathregular{E_{%s} - E_{%s}}$ [keV]' % (labelpr, labeltr), fontsize = 14)
    ax1.set_xlabel(r'x [mm]', fontsize = 14)

    ax1.imshow(hist2D_X.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    ax1.axhline(y=0, color='k')
    for i in range(-sig, sig+1):
        # ax1.plot(bin_centres, mean_X + float(i) * std_X, color=color)
        ax1.fill_between(bin_centres,
                         (mean_X + float(i) * std_X - np.sqrt((std_X ** 2 / N_X) + (float(i) * std_unc_X) ** 2)),
                         (mean_X + float(i) * std_X + np.sqrt((std_X ** 2 / N_X) + (float(i) * std_unc_X) ** 2)), facecolor=color, lw=0.1, color=color)
    # cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    # cbar.set_label('Probability')

    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlim([-limit_Pos, limit_Pos])
    ax2.set_ylim([-limit_Res, limit_Res])
    ax2.grid()
    ax2.set_xlabel(r'y [mm]', fontsize=14)
    ax2.imshow(hist2D_Y.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    ax2.axhline(y=0, color='k')
    for i in range(-sig, sig+1):
        # ax2.plot(bin_centres, mean_Y + float(i) * std_Y, color=color)
        ax2.fill_between(bin_centres,
                         (mean_Y + float(i) * std_Y - np.sqrt((std_Y ** 2 / N_Y) + (float(i) * std_unc_Y) ** 2)),
                         (mean_Y + float(i) * std_Y + np.sqrt((std_Y ** 2 / N_Y) + (float(i) * std_unc_Y) ** 2)), facecolor=color, lw=0.1, color=color)

    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.set_xlim([-limit_Pos, limit_Pos])
    ax3.set_ylim([-limit_Res, limit_Res])
    ax3.grid()
    ax3.set_xlabel(r'z [mm]', fontsize=14)
    ax3.imshow(hist2D_Z.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    ax3.axhline(y=0, color='k')
    for i in range(-sig, sig+1):
        # ax3.plot(bin_centres, mean_Z + float(i) * std_Z, color=color)
        ax3.fill_between(bin_centres,
                         (mean_Z + float(i) * std_Z - np.sqrt((std_Z ** 2 / N_Z) + (float(i) * std_unc_Z) ** 2)),
                         (mean_Z + float(i) * std_Z + np.sqrt((std_Z ** 2 / N_Z) + (float(i) * std_unc_Z) ** 2)), facecolor=color, lw=0.1, color=color)


    plt.setp(ax11.get_xticklabels(), visible=False)
    plt.setp(ax11.get_yticklabels(), visible=False)
    ax11.yaxis.grid(False)
    ax11.xaxis.grid(True)
    ax11.set_yscale("log", nonposy='clip')
    ax11.step(bin_centres, hist1D_X, where='mid', color='k')

    ax21.step(bin_centres, hist1D_Y, where='mid', color='k')
    ax21.axis([-limit_Pos, limit_Pos, 1.e-5, 1.e-1])
    plt.setp(ax21.get_xticklabels(), visible=False)
    plt.setp(ax21.get_yticklabels(), visible=False)
    ax21.xaxis.grid(True)

    ax31.step(bin_centres, hist1D_Z, where='mid', color='k')
    ax31.axis([-limit_Pos, limit_Pos, 1.e-5, 1.e-1])
    plt.setp(ax31.get_xticklabels(), visible=False)
    plt.setp(ax31.get_yticklabels(), visible=False)
    ax31.xaxis.grid(True)


    ax41.set_xticks([])
    ax41.set_xticklabels([])
    plt.setp(ax41.get_xticklabels(), visible=False)
    plt.setp(ax41.get_yticklabels(), visible=False)
    ax41.set_ylim([-limit_Res, limit_Res])
    ax41.set_xlim([0.0, 0.02])
    ax41.yaxis.grid(True)
    ax41.hist(resE, bins=pos_bins, histtype="step", color="k", orientation="horizontal", range=(-limit_Res, limit_Res), normed=True)
    mean_E = np.mean(resE[(abs(resE) <= limit_Res) & (abs(posX) <= limit_Pos) & (abs(posY) <= limit_Pos) & (abs(posZ) <= limit_Pos)])
    std_E = np.std(resE[(abs(resE) <= limit_Res) & (abs(posX) <= limit_Pos) & (abs(posY) <= limit_Pos) & (abs(posZ) <= limit_Pos)])
    # std_unc_E = std_uncertainty(resE[(abs(resE) <= limit_Res) & (abs(posX) <= limit_Pos) & (abs(posY) <= limit_Pos) & (abs(posZ) <= limit_Pos)])
    # N_E = float(resE[(abs(resE) <= limit_Res) & (abs(posX) <= limit_Pos) & (abs(posY) <= limit_Pos) & (abs(posZ) <= limit_Pos)].size)
    for i in range(-sig, sig+1):
        ax41.axhline(y=(mean_E + float(i) * std_E), color=color)
        # ax41.axhspan((mean_E + float(i) * std_E - np.sqrt((std_E ** 2 / N_E) + (float(i) * std_unc_E) ** 2)),
        #              (mean_E + float(i) * std_E + np.sqrt((std_E ** 2 / N_E) + (float(i) * std_unc_E) ** 2)), facecolor=color, lw=0.1, color=color)
    # ax41.set_yscale("log", nonposy='clip')

    fig.savefig(fileOUT, bbox_inches='tight')



def std_uncertainty(data):
    n = float(data.size)
    if n == 0.0: return 0.0
    return np.std(data) * np.sqrt(np.exp(1.) * (1. - 1. / n) ** (n - 1.) - 1.)

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()