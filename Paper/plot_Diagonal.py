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
    # DEFINE MODEL W/ OR W/O CALIBRATION
    Model = "/180308-1100/180309-1055/180310-1553/180311-2206/180312-1917/180313-2220/"
    Epoch = 99
    Source = 'th'
    Position = 'S5'
    Calibration = True

    # SET DATA PATHS
    Epoch = str(Epoch).zfill(3)
    folderOUT = "/home/vault/capm/sn0515/PhD/Th_U-Wire/Paper/"
    folderIN_MC = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model + "/1validation-data/" + Source + "ms-" + Position + "/"
    folderIN_Data = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/" + Model + "/0physics-data/" + Epoch + "/" + Source + "ms-" + Position + "/"
    fileIN = "spectrum_events_" + Epoch + "_" + Source + "ms-" + Position + ".p"

    print 'folderIN_MC\t', folderIN_MC
    print 'folderIN_Data\t', folderIN_Data
    print 'fileIN\t', fileIN

    # LOAD EVENTS
    E_MC = get_events(folderIN_MC + fileIN)
    E_Data = get_events(folderIN_Data + fileIN)

    print E_Data["E_CNN"].shape, E_Data["E_EXO"].shape
    print E_MC["E_True"].shape, E_MC["E_CNN"].shape, E_MC["E_CNN"].shape

    # APPLY CALIBRATION ON TL208 PEAK
    if Calibration:
        # CALIBRATE CNN AND EXO SEPARATELY FOR SS AND MS
        for E_List_str in ['E_CNN', 'E_EXO', 'E_True']:
            E_MC[E_List_str] = {'SS': E_MC[E_List_str][E_MC['isSS'] == True],
                                'MS': E_MC[E_List_str][E_MC['isSS'] == False]}
            if E_List_str != 'E_True':
                E_Data[E_List_str] = {'SS': E_Data[E_List_str][E_Data['isSS'] == True],
                                      'MS': E_Data[E_List_str][E_Data['isSS'] == False]}
                for Multi in ['SS', 'MS']:
                    CalibrationFactorData = plot.calibrate_spectrum(data=E_Data[E_List_str][Multi], name='', isMC=False,
                                                                peakpos=2614.5, fOUT=None, peakfinder='max')
                    E_Data[E_List_str][Multi] = E_Data[E_List_str][Multi] / CalibrationFactorData
                    CalibrationFactorMC = plot.calibrate_spectrum(data=E_MC[E_List_str][Multi], name='', isMC=True,
                                                                peakpos=2614.5, fOUT=None, peakfinder='max')
                    E_MC[E_List_str][Multi] = E_MC[E_List_str][Multi] / CalibrationFactorMC
                E_Data[E_List_str]['SSMS'] = np.concatenate((E_Data[E_List_str]['SS'], E_Data[E_List_str]['MS']))
            E_MC[E_List_str]['SSMS'] = np.concatenate((E_MC[E_List_str]['SS'], E_MC[E_List_str]['MS']))

        # PRODUCE PLOTS FOR MC AND DATA AND FOR SS AND MS AND SS+MS
        for Multi in ['SS', 'MS', 'SSMS']:
            plot_predict(x=E_Data["E_EXO"][Multi], y=E_Data["E_CNN"][Multi], xlabel='EXO Recon', ylabel='DNN',
                         fOUT=(folderOUT + Epoch + "_" + Source + "ms_scatter_hist2d_calib_" + Multi + ".pdf"))
            plot_predict(x=E_MC["E_True"][Multi], y=E_MC["E_CNN"][Multi], xlabel='True', ylabel='DNN',
                         fOUT=(folderOUT + "prediction_" + Source + "ms_ConvNN_" + Epoch + "_calib_" + Multi + ".pdf"))
            plot_predict(x=E_MC["E_True"][Multi], y=E_MC["E_EXO"][Multi], xlabel='True', ylabel='EXO Recon',
                         fOUT=(folderOUT + "prediction_" + Source + "ms_Standard_" + Epoch + "_calib_" + Multi + ".pdf"))
    # DONT APPLY CALIBRATION
    else:
        fileOUT_MC_CNN = "prediction_" + Source + Multi + "_ConvNN_" + Epoch + ".pdf"
        fileOUT_MC_Std = "prediction_" + Source + Multi + "_Standard_" + Epoch + ".pdf"
        fileOUT_Data = Epoch + "_" + Source + Multi + "_scatter_hist2d" + ".pdf"

        plot_predict(x=E_Data["E_EXO"], y=E_Data["E_CNN"], xlabel='EXO Recon', ylabel='DNN', fOUT=(folderOUT + fileOUT_Data))
        plot_predict(x=E_MC["E_True"], y=E_MC["E_CNN"], xlabel='True', ylabel='DNN', fOUT=(folderOUT + fileOUT_MC_CNN))
        plot_predict(x=E_MC["E_True"], y=E_MC["E_EXO"], xlabel='True', ylabel='EXO Recon', fOUT=(folderOUT + fileOUT_MC_Std))

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

def plot_predict(x,y, xlabel, ylabel, fOUT):
    # Create figure
    dE = y - x
    lowE = 600
    upE = 3300
    resE = 300
    gridsize = 200
    diag = np.asarray([lowE, upE])
    extent1 = [lowE, upE, lowE, upE]
    extent2 = [lowE, upE, -resE, resE]
    # plt.ion()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(8.5,11.5)) #, sharex=True) #, gridspec_kw = {'height_ratios':[3, 1]})
    # plt.subplots_adjust(bottom=0.1, right=0.95, top=0.95, left=0.1, wspace=0.0)
    fig.subplots_adjust(wspace=0, hspace=0.05)
    ax1.set(aspect='equal', adjustable='box-forced')
    ax1.set(aspect='auto')

    ax1.plot(diag, diag, 'k--', lw=2)
    for idx,shift in enumerate([200,400,600,800]):
        ax1.plot(diag, diag+shift, 'k--', alpha=(0.8-0.2*idx), lw=2, label=str(shift))
        ax1.plot(diag, diag-shift, 'k--', alpha=(0.8-0.2*idx), lw=2, label=str(shift))

    xvals = [2700., 3100., 2500., 3100., 2300., 3100]
    labelLines(ax1.get_lines()[3:], xvals=xvals, align=True,color='k')

    ax2.axhline(y=0.0, ls='--', lw=2, color='black')
    for idx,shift in enumerate([100,200,300,400]):
        ax2.axhline(y=-shift, ls='--', lw=2, alpha=(0.7-0.3*idx), color='black')
        ax2.axhline(y=shift , ls='--', lw=2, alpha=(0.7-0.3*idx), color='black')
    # ax2.axhline(y=-200.0, ls='--', lw=2, color='black')
    # ax2.axhline(y= 200.0, ls='--', lw=2, color='black')
    ax1.set(ylabel=ylabel + ' Energy [keV]')
    # ax2.set(xlabel=xlabel + ' Energy [keV]', ylabel='Residual [keV]')
    ax2.set(xlabel=xlabel + ' Energy [keV]', ylabel='(%s - %s) [keV]' % (ylabel, xlabel))
    ax1.set_xlim([lowE, upE])
    ax1.set_ylim([lowE, upE])
    ax2.set_ylim([-resE, resE])
    # ax1.xaxis.grid(True)
    # ax1.yaxis.grid(True)
    # ax2.xaxis.grid(True)
    # ax2.yaxis.grid(True)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp(ax2, yticks=[-200, -100, 0, 100, 200])
    ax1.hexbin(x, y, bins='log', extent=extent1, gridsize=gridsize, mincnt=1, cmap=plt.get_cmap('viridis'), linewidths=0.1)
    ax2.hexbin(x, dE, bins='log', extent=extent2, gridsize=(gridsize,gridsize/((upE-lowE)/(2*resE))), mincnt=1, cmap=plt.get_cmap('viridis'), linewidths=0.1)
    # plt.show()
    # raw_input("")
    plt.savefig(fOUT)
    return


# scatter (Hist2D)
def plot_scatter_hist2d(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    hist, xbins, ybins = np.histogram2d(E_x, E_y, bins=200, normed=True)
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    diag = np.asarray([700, 3500])
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
    plt.xlim(xmin=700, xmax=3500)
    plt.ylim(ymin=700, ymax=3500)
    plt.grid(True)
    plt.savefig(fOUT)
    plt.clf()
    plt.close()
    return

# residual (Hist2D)
def plot_residual_hist2d(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    hist, xbins, ybins = np.histogram2d(E_x, dE, range=[[700,3500],[-250,250]], bins=180, normed=True )
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = (2800)/(600)
    im = plt.imshow(hist.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    plt.plot((700, 3500), (0, 0), color='black')
    cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    cbar.set_label('Probability')
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.xlim(xmin=700, xmax=3500)
    plt.ylim(ymin=-250, ymax=250)
    plt.grid(True)
    plt.savefig(fOUT)
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
    plt.xlim(xmin=500, xmax=3500)
    plt.ylim(ymin=-50, ymax=50)
    plt.grid(True)
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Mean Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.savefig(fOUT)
    plt.clf()
    plt.close()
    return

#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        # ang = degrees(atan2(dy,dx))
        # ang = degrees(dy/dx)
        ang = 60.

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].get_axes()
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()