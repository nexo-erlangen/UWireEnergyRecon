#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import cPickle as pickle

BIGGER_SIZE = 18
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def make_plots(folderOUT, dataIn, epoch, sources, position):
    import os
    os.system("mkdir -p %s " % (folderOUT + "2prediction-spectrum/"))
    os.system("mkdir -p %s " % (folderOUT + "3prediction-scatter/" ))
    os.system("mkdir -p %s " % (folderOUT + "4residual-histo/"     ))
    os.system("mkdir -p %s " % (folderOUT + "5residual-scatter/"   ))
    os.system("mkdir -p %s " % (folderOUT + "6residual-mean/"      ))
    os.system("mkdir -p %s " % (folderOUT + "7residual-sigma/"))
    os.system("mkdir -p %s " % (folderOUT + "8residual-violin/"))

    name_CNN = 'DNN'
    name_EXO = 'EXO Recon'
    name_True = 'True'
    peakpos = 2614.5

    if ('th' in sources or 'ga' in sources) and position == 'S5':
        CalibrateSelf = True
    else:
        CalibrateSelf = False
        dataIn_Ref = pickle.load(open(folderOUT + '../thms-S5/spectrum_events_' + epoch + '_thms-S5.p', "rb"))

    data = {}
    data_Ref = {}
    for E_List_str in ['E_True', 'E_CNN', 'E_EXO']:
        data[E_List_str] = {'SS': dataIn[E_List_str][dataIn['isSS'] == True],
                            'MS': dataIn[E_List_str][dataIn['isSS'] == False],
                            'SSMS': dataIn[E_List_str]}
        if not CalibrateSelf:
            data_Ref[E_List_str] = {'SS': dataIn_Ref[E_List_str][dataIn_Ref['isSS'] == True],
                                    'MS': dataIn_Ref[E_List_str][dataIn_Ref['isSS'] == False],
                                    'SSMS': dataIn_Ref[E_List_str]}

        for Multi in ['SS', 'MS']:
            CalibrationFactor = 1.0
            # CalibrationOffset = 0.0
            if len(data[E_List_str][Multi]) != 0:
                if E_List_str != 'E_True':
                    if CalibrateSelf:
                        CalibrationFactor = calibrate_spectrum(data=data[E_List_str][Multi], name='', peakpos=peakpos, fOUT=None, isMC=True, peakfinder='max')
                    else:
                        CalibrationFactor = calibrate_spectrum(data=data_Ref[E_List_str][Multi], name='', peakpos=peakpos, fOUT=None, isMC=True, peakfinder='max')
                    # if CalibrateSelf:
                    #     CalibrationFactor, CalibrationOffset = doCalibration(data_True=data['E_True'][Multi], data_Recon=data[E_List_str][Multi])
                    # else:
                    #     CalibrationFactor, CalibrationOffset = doCalibration(data_True=data['E_True'][Multi], data_Recon=data[E_List_str][Multi])
            #         print E_List_str, Multi, CalibrationFactor, CalibrationOffset
            # data[E_List_str]['calib_' + Multi] = (data[E_List_str][Multi]-CalibrationOffset) / CalibrationFactor
            # print E_List_str, Multi, CalibrationFactor
            data[E_List_str]['calib_' + Multi] = data[E_List_str][Multi] / CalibrationFactor
        data[E_List_str]['calib_SSMS'] = np.concatenate((data[E_List_str]['calib_SS'], data[E_List_str]['calib_MS']))

    obs = {}
    obs['peak_pos'], obs['peak_sig'] = plot_spectrum(data_CNN=data['E_CNN']['SSMS'], data_EXO=data['E_EXO']['SSMS'],
                                                     data_True=data['E_True']['SSMS'], fit=('th' in sources), peakpos=peakpos, isMC=True,
                                                     fOUT=(folderOUT + '2prediction-spectrum/spectrum_' + sources + '_' + position + '_' + epoch + '_SSMS.pdf'))
    obs['resid_pos'], obs['resid_sig'] = plot_residual_histo(dE=(data['E_True']['SSMS'] - data['E_CNN']['SSMS']), name_x=name_True, name_y=name_CNN,
                        fOUT=folderOUT + '4residual-histo/histogram_' + sources + '_' + position + '_ConvNN_SSMS_' + epoch + '.pdf')

    # for Multi in ['SS', 'MS', 'SSMS', 'calib_SS', 'calib_MS', 'calib_SSMS']:
    # for Multi in ['calib_SS', 'calib_MS', 'calib_SSMS']:
    for Multi in ['SSMS', 'calib_SS', 'calib_MS', 'calib_SSMS']:
    # for Multi in ['SSMS']:
        # print 'plotting', Multi
        plot_spectrum(data_CNN=data['E_CNN'][Multi], data_EXO=data['E_EXO'][Multi], data_True=data['E_True'][Multi], fit=('th' in sources), isMC=True,
                      peakpos=peakpos, fOUT=(folderOUT + '2prediction-spectrum/spectrum_' + sources + '_' + position + '_' + epoch + '_' + Multi + '.pdf'))
        plot_residual_histo(dE=(data['E_True'][Multi] - data['E_EXO'][Multi]), name_x=name_True, name_y=name_EXO,
                            fOUT=folderOUT + '4residual-histo/histogram_' + sources + '_' + position + '_Standard_' + Multi + '_' + epoch + '.pdf')
        plot_residual_histo(dE=(data['E_True'][Multi] - data['E_CNN'][Multi]), name_x=name_True, name_y=name_CNN,
                            fOUT=folderOUT + '4residual-histo/histogram_' + sources + '_' + position + '_ConvNN_' + Multi + '_' + epoch + '.pdf')
        plot_scatter_hist2d(E_x=data['E_True'][Multi], E_y=data['E_EXO'][Multi], name_x=name_True, name_y=name_EXO,
                            fOUT=folderOUT + '3prediction-scatter/prediction_' + sources + '_' + position + '_Standard_' + Multi + '_' + epoch + '.pdf')
        plot_scatter_hist2d(E_x=data['E_True'][Multi], E_y=data['E_CNN'][Multi], name_x=name_True, name_y=name_CNN,
                            fOUT=folderOUT + '3prediction-scatter/prediction_' + sources + '_' + position + '_ConvNN_' + Multi + '_' + epoch + '.pdf')
        plot_residual_hist2d(E_x=data['E_True'][Multi], E_y=data['E_EXO'][Multi], name_x=name_True, name_y=name_EXO,
                             fOUT=folderOUT + '5residual-scatter/residual_' + sources + '_' + position + '_Standard_' + Multi + '_' + epoch + '.pdf')
        plot_residual_hist2d(E_x=data['E_True'][Multi], E_y=data['E_CNN'][Multi], name_x=name_True, name_y=name_CNN,
                             fOUT=folderOUT + '5residual-scatter/residual_' + sources + '_' + position + '_ConvNN_' + Multi + '_' + epoch + '.pdf')
        # plot_residual_hist2d(E_x=data['E_EXO'][Multi], E_y=data['E_CNN'][Multi], name_x=name_EXO, name_y=name_CNN,
        #                      fOUT=folderOUT + '5residual-scatter/residual_' + sources + '_' + position + '_Both_Standard_' + Multi + '_' + epoch + '.pdf')
        # plot_residual_hist2d(E_x=data['E_CNN'][Multi], E_y=data['E_EXO'][Multi], name_x=name_CNN, name_y=name_EXO,
        #                      fOUT=folderOUT + '5residual-scatter/residual_' + sources + '_' + position + '_Both_ConvNN_' + Multi + '_' + epoch + '.pdf')
        # plot_residual_scatter_mean(E_x=data['E_True'][Multi], E_y=data['E_EXO'][Multi], name_x=name_True, name_y=name_EXO,
        #                            fOUT=folderOUT + '6residual-mean/residual_mean_' + sources + '_' + position + '_Standard_' + Multi + '_' + epoch + '.pdf')
        # plot_residual_scatter_mean(E_x=data['E_True'][Multi], E_y=data['E_CNN'][Multi], name_x=name_True, name_y=name_CNN,
        #                            fOUT=folderOUT + '6residual-mean/residual_mean_' + sources + '_' + position + '_ConvNN_' + Multi + '_' + epoch + '.pdf')
        # plot_residual_scatter_sigma(E_x=data['E_True'][Multi], E_CNN=data['E_CNN'][Multi], E_EXO=data['E_EXO'][Multi],
        #                             name_x=name_True, name_CNN=name_CNN, name_EXO=name_EXO,
        #                             fOUT=folderOUT + '7residual-sigma/residual_sigma_' + sources + '_' + position + '_' + Multi + '_' + epoch + '.pdf')
        # plot_residual_violin(E_x=data['E_True'][Multi], E_CNN=data['E_CNN'][Multi], E_EXO=data['E_EXO'][Multi],
        #                      name_x=name_True, name_CNN=name_CNN, name_EXO=name_EXO,
        #                      fOUT=folderOUT + '8residual-violin/residual_' + sources + '_' + position + '_' + Multi + '_' + epoch + '.pdf')
    return obs

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def doCalibration(data_True, data_Recon):
    m, b = np.polyfit(data_True, data_Recon, 1)
    return m, b

def fit_spectrum(data, peakpos, fit, name, color, isMC, peakfinder='max', zorder=3):
    hist, bin_edges = np.histogram(data, bins=1200, range=(0, 12000), density=False)
    norm_factor = float(len(data))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    if name != 'MC':
        if fit:
            from scipy.optimize import curve_fit
            peak = find_peak(hist=hist, bin_centres=bin_centres, peakpos=peakpos, peakfinder=peakfinder)
            coeff = [hist[peak], bin_centres[peak], 50., -0.005]
            for i in range(5):
                try:
                    if isMC==True: #fit range for MC spectra
                        low = np.digitize(coeff[1] - (5.5 * abs(coeff[2])), bin_centres)
                        up = np.digitize(coeff[1] + (3.0 * abs(coeff[2])), bin_centres)
                    else: #fit range from RotationAngle script #original was 3. for lower bound
                        low = np.digitize(coeff[1] - (3.5 * abs(coeff[2])), bin_centres)
                        up = np.digitize(coeff[1] + (2.5 * abs(coeff[2])), bin_centres)
                    coeff, var_matrix = curve_fit(gaussErf, bin_centres[low:up], hist[low:up], p0=coeff)
                    coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
                except:
                    print name, 'fit did not work\t', i
                    coeff, coeff_err = [hist[peak], bin_centres[peak], 50.0*(i+1), -0.005], [0.0] * len(coeff)
            delE = abs(coeff[2]) / coeff[1] * 100.0
            delE_err = delE * np.sqrt((coeff_err[1] / coeff[1]) ** 2 + (coeff_err[2] / coeff[2]) ** 2)

            # plt.plot(bin_centres[low:up], gauss_zero(bin_centres[low:up], *coeff[:3]) / norm_factor, lw=1, ls='--', color=color)
            # plt.plot(bin_centres[low:up], erf(bin_centres[low:up], *coeff[1:]) / norm_factor, lw=1, ls='--', color=color)
            # plt.plot(bin_centres[low:up], gaussErf(bin_centres[low:up], *coeff) / norm_factor, lw=1 , ls='--', color=color)

            plt.plot(bin_centres, gauss_zero(bin_centres, *coeff[:3]) / norm_factor, lw=1, ls='--', color=color, zorder=zorder)
            # plt.step(bin_centres, hist / norm_factor, where='mid', color=color, label='%s: $%.4f \pm %.4f$ %% $(\sigma)$' % (name, delE, delE_err), zorder=zorder+1)
            plt.step(bin_centres, hist / norm_factor, where='mid', color=color, label='%s' % (name), zorder=zorder+1)
            # plt.axvline(x=2614, c='k', lw=2)
            return (coeff[1], coeff_err[1]), (abs(coeff[2]), coeff_err[2])
        else:
            plt.step(bin_centres, hist / norm_factor, where='mid', color=color, label='%s' % (name), zorder=zorder)
            return (-1000., 1000.), (-1000., -1000.)
    else:
        plt.plot(bin_centres, hist / norm_factor, label=name, color=color, lw=0.5, zorder=2)
        plt.fill_between(bin_centres, 0.0, hist / norm_factor, facecolor='black', alpha=0.3, interpolate=True, zorder=1)
        return

def find_peak(hist, bin_centres, peakpos, peakfinder='max'):
    peak = hist[hist.size/2]
    if peakfinder == 'max':
        peak = np.argmax(
            hist[np.digitize(peakpos - 300, bin_centres):np.digitize(peakpos + 300, bin_centres)]) + np.digitize(
            peakpos - 300, bin_centres)
    elif peakfinder == 'fromright':
        length = hist.size
        inter = 20
        # from math import  sqrt
        for i in range(len(hist) - inter, inter, -1):
            if hist[i] < 50: continue
            if np.argmax(hist[ np.max( [i - inter, 0] ) : np.min( [i + inter, length] ) ]) == inter: peak = i ; break
            # if hist[i + 1] <= 0: continue
            # sigma = sqrt(hist[i] + hist[i + 1])
            # if abs((hist[i + 1] - hist[i]) / sigma) >= 5.0:
            #     peak = i + 1
            #     break
    return peak

def calibrate_spectrum(data, name, peakpos, fOUT, isMC, peakfinder):
    import matplotlib.backends.backend_pdf
    from matplotlib.backends.backend_pdf import PdfPages
    if fOUT is not None: pp = PdfPages(fOUT)
    mean_recon = (peakpos, 0.0)
    CalibrationFactor = 1.
    data_new = data
    # print '==========='
    # print 'calibrating\t isMC:\t', isMC
    # print '==========='
    for i in range(7):
        data_new = data_new / (mean_recon[0] / peakpos)
        plot = plt.figure()
        mean_recon, sig_recon = fit_spectrum(data=data_new, peakpos=peakpos, fit=True, name=(name + '_' + str(i)), color='k', isMC=isMC, peakfinder=peakfinder)
        plt.xlabel('Energy [keV]')
        plt.ylabel('Probability')
        plt.legend(loc="lower left")
        plt.axvline(x=2614.5, lw=2, color='k')
        plt.xlim(xmin=500, xmax=3500)
        plt.ylim(ymin=(1.0 / float(len(data))), ymax=0.1)
        plt.grid(True)
        plt.gca().set_yscale('log')
        if fOUT is not None: pp.savefig(plot)
        plt.clf()
        plt.close()
        CalibrationFactor *= mean_recon[0] / peakpos
    if fOUT is not None: pp.close()
    return CalibrationFactor

def plot_spectrum(data_CNN, data_EXO, peakpos, fit, fOUT, isMC, data_True=None):
    # print '==========='
    # print 'plot spectrum\t isMC:\t', isMC
    # print '==========='
    if data_True is not None: fit_spectrum(data=data_True, peakpos=peakpos, fit=False, name="MC", color='k', isMC=True)
    peak_pos, peak_sig = fit_spectrum(data=data_CNN, peakpos=peakpos, fit=fit, name="DNN", color='blue', isMC=isMC, zorder=5)
    fit_spectrum(data=data_EXO, peakpos=peakpos, fit=fit , name="EXO Recon", color='firebrick', isMC=isMC, zorder=3)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    leg = plt.legend(loc="lower left")
    leg.set_zorder(20)
    # plt.xlim(xmin=600, xmax=2600) #for Co60 and Ra226
    plt.xlim(xmin=600, xmax=3300) #for Th228
    plt.ylim(ymin=5.e-5, ymax=5.e-2)
    plt.grid(True)
    plt.gca().set_yscale('log')
    plt.savefig(fOUT, bbox_inches='tight')
    # plt.savefig(fOUT[:-4] + "_res" + fOUT[-4:], bbox_inches='tight')
    plt.clf()
    plt.close()
    if peakpos is not None: return peak_pos, peak_sig
    else: return

# training curves
def plot_losses(folderOUT, history):
    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['loss'],     label='training')
    ax.plot(history['epoch'], history['val_loss'], label='validation')
    ax.set(xlabel='epoch', ylabel='loss')
    ax.grid(True)
    ax.semilogy()
    plt.legend(loc="best")
    fig.savefig(folderOUT+'loss-test.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['mean_absolute_error'],     label='training')
    ax.plot(history['epoch'], history['val_mean_absolute_error'], label='validation')
    ax.legend()
    ax.grid(True)
    plt.legend(loc="best")
    ax.set(xlabel='epoch', ylabel='mean absolute error')
    fig.savefig(folderOUT+'mean_absolute_error-test.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# histogram of the data
def plot_residual_histo(dE, name_x, name_y, fOUT):
    from scipy.optimize import curve_fit
    hist_dE, bin_edges, dummy = plt.hist(dE, bins=np.arange(-300,304,4), normed=True, facecolor='green', alpha=0.75)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak = np.argmax(hist_dE[np.digitize(-100, bin_centres):np.digitize(100, bin_centres)]) + np.digitize(-100,bin_centres)
    coeff = [hist_dE[peak], bin_centres[peak], 50.0]
    for i in range(5):
        try:
            low = np.digitize(coeff[1] - (2 * abs(coeff[2])), bin_centres)
            up = np.digitize(coeff[1] + (2 * abs(coeff[2])), bin_centres)
            coeff, var_matrix = curve_fit(gauss_zero, bin_centres[low:up], hist_dE[low:up], p0=coeff)
            coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
        except:
            print fOUT, '\tfit did not work\t', i
            coeff, coeff_err = [hist_dE[peak], bin_centres[peak], 50.0*(i+1)], [0.0, 0.0, 0.0]
    plt.plot(range(-200,200), gauss_zero(range(-200,200), *coeff), lw=2, color='red',
             label='%s\n$\mu=%.1f \pm %.1f$\n$\sigma=%.1f \pm %.1f$'%(name_y, coeff[1], coeff_err[1], abs(coeff[2]), coeff_err[2]))
    plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.xlabel('Probability')
    plt.legend(loc="best")
    plt.xlim(xmin=-250, xmax=250)
    plt.ylim(ymin=0.0, ymax=0.025)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return ((coeff[1], coeff_err[1]), (abs(coeff[2]), coeff_err[2]))

# scatter
def plot_scatter(E_x, E_y, name_x, name_y, fOUT):
    dE = E_x - E_y
    plt.scatter(E_x, E_y, label='%s\n$\mu=%.1f, \sigma=%.1f$'%('training set', np.mean(dE), np.std(dE)))
    plt.plot((500, 3000), (500, 3000), 'k--')
    plt.legend(loc="best")
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('%s Energy [keV]' % (name_y))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=600, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# scatter (Hist2D)
def plot_scatter_hist2d(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    hist, xbins, ybins = np.histogram2d(E_x, E_y, bins=200, normed=True)
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    diag = np.asarray([600,3300])
    plt.plot(diag, diag, 'k--')
    plt.plot(diag, diag + 200, 'k--')
    plt.plot(diag, diag - 200, 'k--')
    plt.plot(diag, diag + 400, 'k--')
    plt.plot(diag, diag - 400, 'k--')
    im = plt.imshow(hist.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'),
                    origin='lower', norm=mpl.colors.LogNorm(),
                    label='%s\n$\mu=%.1f, \sigma=%.1f$' % ('MC data', np.mean(dE), np.std(dE)))
    cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    cbar.set_label('Probability')
    plt.legend(loc="best")
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('%s Energy [keV]' % (name_y))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=600, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# scatter (Density)
def plot_scatter_density(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    # Calculate the point density
    from scipy.stats import gaussian_kde
    xy = np.vstack([E_x, E_y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    E_x, E_y, z = E_x[idx], E_y[idx], z[idx]

    plt.scatter(E_x, E_y, c=z, s=2, edgecolor='', cmap=plt.get_cmap('viridis'),
               label='%s\n$\mu=%.1f, \sigma=%.1f$' % ('physics data', np.mean(dE), np.std(dE)))
    plt.plot((600, 3300), (600, 3300), 'k--')
    plt.colorbar()
    plt.legend(loc="best")
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('%s Energy [keV]' % (name_y))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=600, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT)
    plt.clf()
    plt.close()
    return

# residual
def plot_residual_scatter(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    plt.scatter(E_x, dE)
    plt.plot((600, 3300), (0,0), color='black')
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=-300, ymax=300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# residual (Hist2D)
def plot_residual_hist2d(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    hist, xbins, ybins = np.histogram2d(E_x, dE, range=[[600,3300],[-250,250]], bins=180, normed=True )
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = (2700)/(500)
    im = plt.imshow(hist.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    plt.plot((600, 3300), (0, 0), color='black')
    cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    cbar.set_label('Probability')
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=-250, ymax=250)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# residual (Density)
def plot_residual_density(E_x, E_y, name_x, name_y, fOUT):
    dE = np.array(E_y - E_x)
    E_x = np.array(E_x)
    # Calculate the point density
    from scipy.stats import gaussian_kde
    xy = np.vstack([E_x, dE])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    E_x, E_y, z = E_x[idx], dE[idx], z[idx]

    plt.scatter(E_x, dE, c=z, s=5, edgecolor='', cmap=plt.get_cmap('viridis'))
    plt.plot((600, 3300), (0,0), color='black')
    plt.colorbar()
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=-300, ymax=300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# mean-residual
def plot_residual_scatter_mean(E_x, E_y, name_x, name_y, fOUT):
    import warnings
    dE = E_y - E_x
    bin_edges = [ x for x in range(0,4000,150) ]
    bins = [ [] for x in range(0,3850,150) ]
    for i in range(len(dE)):
        bin = np.digitize(E_x[i], bin_edges) - 1
        bins[bin].append(dE[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bins = [ np.array(bin) for bin in bins]
        mean = [ np.mean(bin)  for bin in bins]
        stda = [ np.std(bin)/np.sqrt(len(bin))  for bin in bins]
    bin_width=((bin_edges[1]-bin_edges[0])/2.0)
    plt.errorbar((np.array(bin_edges[:-1])+bin_width), mean, xerr=bin_width, yerr=stda, fmt="none")
    plt.axhline(y=0.0, lw=2, color='k')
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=-50, ymax=50)
    plt.grid(True)
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Mean Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# mean-residual violin
def plot_residual_violin(E_x, E_CNN, E_EXO, name_x, name_CNN, name_EXO, fOUT):
    import seaborn as sns
    import pandas as pd
    sns.set_style("whitegrid")
    dE_CNN = E_CNN - E_x
    dE_EXO = E_EXO - E_x
    bin_edges = [x for x in range(0, 4250, 250)]
    bin_width = int((bin_edges[1] - bin_edges[0]) / 2.0)
    data_dic = {'energy': [], 'residual': [], 'type': []}
    for i in range(len(E_x)):
        bin_CNN = np.digitize(E_x[i], bin_edges) - 1
        data_dic['energy'].append(bin_edges[bin_CNN]+bin_width)
        data_dic['residual'].append(dE_CNN[i])
        data_dic['type'].append(name_CNN)
        data_dic['energy'].append(bin_edges[bin_CNN]+bin_width)
        data_dic['residual'].append(dE_EXO[i])
        data_dic['type'].append(name_EXO)
    data = pd.DataFrame.from_dict(data_dic)
    fig, ax = plt.subplots()
    ax.axhline(y=0.0, lw=2, color='k')
    sns.violinplot(x='energy', y='residual', hue='type', data=data, inner="quartile", palette='Set2', split=True, cut=0, scale='area', scale_hue=True, bw=0.4)
    ax.set_ylim(-150, 150)
    ax.set_xlabel('%s Energy [keV]' % (name_x))
    ax.set_ylabel('Residual ( xxx - %s ) [keV]' % (name_x))
    fig.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    sns.reset_orig()
    return

# mean-residual
def plot_residual_scatter_sigma(E_x, E_CNN, E_EXO, name_x, name_CNN, name_EXO, fOUT):
    import warnings
    dE_CNN = E_CNN - E_x
    dE_EXO = E_EXO - E_x
    bin_edges = [ x for x in range(0,4000,150) ]
    bins_CNN = [ [] for x in range(0,3850,150) ]
    bins_EXO = [[] for x in range(0, 3850, 150)]
    for i in range(len(dE_CNN)):
        bin_CNN = np.digitize(E_x[i], bin_edges) - 1
        bins_CNN[bin_CNN].append(dE_CNN[i])
        bin_EXO = np.digitize(E_x[i], bin_edges) - 1
        bins_EXO[bin_EXO].append(dE_EXO[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bins_EXO = [ np.asarray(bin) for bin in bins_EXO]
        bins_CNN = [np.asarray(bin) for bin in bins_CNN]
        stda_EXO = [ np.std(bin)  for bin in bins_EXO]
        stda_CNN = [np.std(bin) for bin in bins_CNN]
    bin_width=((bin_edges[1]-bin_edges[0])/2.0)
    bin_centers = np.asarray(bin_edges[:-1])+bin_width
    plt.errorbar(bin_centers, 100.*np.asarray(stda_EXO)/bin_centers , xerr=bin_width, fmt="o", label=name_EXO)
    plt.errorbar(bin_centers, 100.*np.asarray(stda_CNN) / bin_centers, xerr=bin_width, fmt="o", label=name_CNN)
    plt.axhline(y=0.0, lw=2, color='k')
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=0, ymax=15.)
    plt.grid(True)
    plt.legend(loc="best")
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Energy Resolution')
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# anticorrelation
def plot_anticorrelation_hist2d(E_x, E_y, name_x, name_y, name_title, fOUT):
    hist, xbins, ybins = np.histogram2d(E_x, E_y, range=[[0,3300],[0,3300]], bins=250, normed=True )
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = (3300.0) / (3300.0)
    # plt.plot((500, 3200), (500, 3200), 'k--')
    im = plt.imshow(hist.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    #cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    #cbar.set_label('Probability')
    plt.title('%s' % (name_title))
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('%s Energy [keV]' % (name_y))
    plt.xlim(xmin=500, xmax=3300)
    plt.ylim(ymin=500, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# rotation vs resolution
def plot_rotationAngle_resolution(fOUT, data):
    for E_List_str in ['E_EXO', 'E_CNN']:
        if E_List_str == 'E_EXO':
            col = 'firebrick'
            label = 'EXO Recon'
        if E_List_str == 'E_CNN':
            col = 'blue'
            label = 'Neural Network'
        TestResolution_ss = np.array(data[E_List_str]['TestResolution_ss']) * 100.
        TestResolution_ms = np.array(data[E_List_str]['TestResolution_ms']) * 100.
        par0ss = data[E_List_str]['BestRes_ss'][0]
        par1ss = data[E_List_str]['Par1_ss'][0]
        par2ss = data[E_List_str]['Theta_ss'][0]
        par0ms = data[E_List_str]['BestRes_ms'][0]
        par1ms = data[E_List_str]['Par1_ms'][0]
        par2ms = data[E_List_str]['Theta_ms'][0]
        print E_List_str, '\tSS\t', par0ss * 100., '\t', par2ss
        print E_List_str, '\tMS\t', par0ms * 100., '\t', par2ms
        limit = 0.07
        x = np.arange(par2ms-limit, par2ms+limit, 0.005)
        plt.errorbar(data[E_List_str]['TestTheta'], TestResolution_ss[:, 0], yerr=TestResolution_ss[:, 1], color=col, fmt="o", label='%s-SS (%.3f%%)' % (label, par0ss * 100.))
        plt.errorbar(data[E_List_str]['TestTheta'], TestResolution_ms[:, 0], yerr=TestResolution_ms[:, 1], color=col, fmt="o", mec=col, mfc='None', label='%s-MS (%.3f%%)' % (label, par0ms * 100.))
        plt.plot(x, parabola(x, par0ss, par1ss, par2ss) * 100., color='k', lw=2)
        plt.plot(x, parabola(x, par0ms, par1ms, par2ms) * 100., color='k', lw=2)
    plt.grid(True)
    plt.legend(loc="best")
    plt.xlabel('Theta [rad]')
    plt.ylabel('Resolution @ Tl208 peak [%]')
    plt.xlim(xmin=(par2ms-0.15), xmax=(par2ms+0.15))
    plt.ylim(ymin=1.0, ymax=2.0)
    plt.savefig(fOUT[:-4] + "_zoom" + fOUT[-4:])
    plt.xlim(xmin=0.0, xmax=1.5)
    plt.ylim(ymin=1.2, ymax=5.5)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# deprecated spectrum
def get_energy_spectrum(args, files):
    import h5py
    entry = []
    for filename in files:
        f = h5py.File(str(filename), 'r')
        temp=np.array(f.get('trueEnergy'))
        for i in range(len(temp)):
            entry.append(temp[i])
        f.close()
    hist, bin_edges = np.histogram(entry, bins=210, range=(500,3000), density=True)
    plt.plot(bin_edges[:-1], hist)
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=500, xmax=3000)
    plt.savefig(args.folderOUT + 'spectrum.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    hist_inv=np.zeros(hist.shape)
    for i in range(len(hist)):
        try:
            hist_inv[i]=1.0/float(hist[i])
        except:
            pass
    hist_inv = hist_inv / hist_inv.sum(axis=0, keepdims=True)
    plt.plot(bin_edges[:-1], hist_inv)
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Weight')
    plt.xlim(xmin=500, xmax=3000)
    plt.savefig(args.folderOUT + 'spectrum_inverse.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    return (hist_inv, bin_edges[:-1])

# input energy spectrum
def get_energy_spectrum_mixed(args, files, add):
    import h5py
    entry, hist, entry_mixed = {}, {}, []
    for source in args.sources:
        entry[source] = []
        for filename in files[source]:
            f = h5py.File(str(filename), 'r')
            temp = np.array(f.get('trueEnergy')).tolist()
            f.close()
            entry[source].extend(temp)
        entry_mixed.extend(entry[source])
    num_counts =  float(len(entry_mixed))
    hist_mixed, bin_edges = np.histogram(entry_mixed, bins=500, range=(0, 5000), density=False)
    bin_width = ((bin_edges[1] - bin_edges[0]) / 2.0)
    plt.plot(bin_edges[:-1] + bin_width, np.array(hist_mixed)/num_counts, label="combined", lw = 2, color='k')

    for source in args.sources:
        if len(entry[source])==0: continue
        label = args.label[source]
        hist[source], bin_edges = np.histogram(entry[source], bins=500, range=(0,5000), density=False)
        plt.plot(bin_edges[:-1] + bin_width, np.array(hist[source])/num_counts, label=label)
        # print "%s\t%s\t%i" % (add, source , len(entry[source]))
    plt.axvline(x=2614.5, lw=2, color='k')
    plt.gca().set_yscale('log')
    plt.gcf().set_size_inches(10,5)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=500, xmax=3500)
    plt.ylim(ymin=(1.0/1000000), ymax=1.0)
    plt.savefig(args.folderOUT + 'spectrum_mixed_' + add + '.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    return

# ----------------------------------------------------------
# Final Plots
# ----------------------------------------------------------
def final_plots(folderOUT, obs):
    if obs == {} :
        print 'final plots \t save.p empty'
        return
    obs_sort, epoch = {}, []
    key_list = list(set( [ key for key_epoch in obs.keys() for key in obs[key_epoch].keys() if key not in ['E_true', 'E_pred']] ))
    for key in key_list:
        obs_sort[key]=[]

    for key_epoch in obs.keys():
        epoch.append(int(key_epoch))
        for key in key_list:
            try:
                obs_sort[key].append(obs[key_epoch][key])
            except KeyError:
                obs_sort[key].append(0.0)

    order = np.argsort(epoch)
    epoch = np.array(epoch)[order]

    for key in key_list:
        obs_sort[key] = np.array(obs_sort[key])[order]
        if key not in ['loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error']:
            obs_sort[key] = np.array([x if type(x) in [np.ndarray,tuple] and len(x)==2 else (x,0.0) for x in obs_sort[key]])

    try:
        plt.plot(epoch, obs_sort['loss'], label='Training set')
        plt.plot(epoch, obs_sort['val_loss'], label='Validation set')
        plt.xlabel('Training time [epoch]')
        plt.ylabel('Loss [keV$^2$]')
        plt.grid(True, which='both')
        # plt.ylim(ymin=7.e2, ymax=2.e4)
        plt.gca().set_yscale('log')
        plt.legend(loc="best")
        plt.savefig(folderOUT + 'loss.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.plot(epoch, obs_sort['mean_absolute_error'], label='Training set')
        plt.plot(epoch, obs_sort['val_mean_absolute_error'], label='Validation set')
        plt.grid(True)
        # plt.ylim(ymin=0.0, ymax=100.0)
        plt.legend(loc="best")
        plt.xlabel('Training time [epoch]')
        plt.ylabel('Mean absolute error [keV]')
        plt.savefig(folderOUT + 'mean_absolute_error.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
    except:
        print 'no loss / mean_err plot possible'

    plt.errorbar(epoch, obs_sort['peak_pos'][:,0], xerr=0.5, yerr=obs_sort['peak_pos'][:,1], fmt="none", lw=2)
    plt.axhline(y=2614.5, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Energy [keV]')
    plt.savefig(folderOUT + '2prediction-spectrum/ZZZ_Peak.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['peak_sig'][:,0], xerr=0.5, yerr=obs_sort['peak_sig'][:,1], fmt="none", lw=2)
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Width [keV]')
    plt.savefig(folderOUT + '2prediction-spectrum/ZZZ_Width.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['resid_pos'][:, 0], xerr=0.5, yerr=obs_sort['resid_pos'][:, 1], fmt="none", lw=2)
    plt.axhline(y=0, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Offset [keV]')
    plt.savefig(folderOUT + '4residual-histo/ZZZ_Offset.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['resid_sig'][:, 0], xerr=0.5, yerr=obs_sort['peak_sig'][:, 1], fmt="none", lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Width [keV]')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.savefig(folderOUT + '4residual-histo/ZZZ_Width.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    return

# ----------------------------------------------------------
# Math Functions
# ----------------------------------------------------------
def gauss(x, A, mu, sigma, off):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + off

def gauss_zero(x, A, mu, sigma):
    return gauss(x, A, mu, sigma, 0.0)

def erf(x, mu, sigma, B):
    import scipy.special
    return B * scipy.special.erf((x - mu) / (np.sqrt(2) * sigma)) + abs(B)

def shift(a, b, mu, sigma):
    return np.sqrt(2./np.pi)*float(b)/a*sigma

def gaussErf(x, A, mu, sigma, B):
    return gauss_zero(x, mu=mu, sigma=sigma, A=A) + erf(x, B=B, mu=mu, sigma=sigma)

def get_weight(Y, hist, bin_edges):
    return hist[np.digitize(Y, bin_edges) - 1]

def round_down(num, divisor):
    return num - (num%divisor)

def parabola(x, par0, par1, par2):
    return par0 + par1 * ((x - par2) ** 2)