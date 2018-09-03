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

    # NETWORK W/ OVERTRAINING
    # Model = "/180308-1102/180309-1055/180310-1553/180311-2201/180312-1917/"
    # Epoch = 149
    # Source = 'ga'

    Multi = 'ms'
    Position = 'S5'
    Calibration = True
    # Calibration = False


    # PREPROCESSING
    Epoch = str(Epoch).zfill(3)
    folderIN = "/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/"+Model+"/1validation-data/"+Source+Multi+"-"+Position+"/"
    fileIN = "spectrum_events_"+Epoch+"_"+Source+Multi+"-"+Position+".p"
    fileOUT = "prediction_ellipse_"+Epoch+"_"+Source+Multi+"_"+Position

    data = get_events(folderIN+fileIN)

    if Calibration:
        fOUT_cal = "_calibrated"
        for E_List_str in ['E_True', 'E_CNN', 'E_EXO']:
            data[E_List_str] = {'SS': data[E_List_str][data['isSS'] == True],
                                'MS': data[E_List_str][data['isSS'] == False],
                                'SSMS': data[E_List_str]}
            if E_List_str != 'E_True':
                for Multi in ['SS', 'MS']:
                    if len(data[E_List_str][Multi]) != 0:
                        CalibrationFactor = plot.calibrate_spectrum(data=data[E_List_str][Multi], name='', peakpos=2614.5, isMC=True, fOUT=None, peakfinder='max')
                        data[E_List_str][Multi] = data[E_List_str][Multi] / CalibrationFactor
                        # CalibrationFactor, CalibrationOffset = plot.doCalibration(data_True=data['E_True'][Multi], data_Recon=data[E_List_str][Multi])
                        # print E_List_str, Multi, CalibrationFactor, CalibrationOffset
                        # data[E_List_str][Multi] = (data[E_List_str][Multi] - CalibrationOffset) / CalibrationFactor
            data[E_List_str]['SSMS'] = np.concatenate((data[E_List_str]['SS'],data[E_List_str]['MS']))
        for Multi in ['SS', 'MS', 'SSMS']:
            doPlot(trEXO=data["E_True"][Multi], prEXO=data["E_EXO"][Multi],
                   trXv=data["E_True"][Multi], prXv=data["E_CNN"][Multi],
                   fileOUT=fileOUT+fOUT_cal+"_"+Multi+".pdf")
            # for Elow in range(500,3000,500):
            #     print Multi, Elow
            #     filter = np.where((data["E_True"][Multi] > Elow) & (data["E_True"][Multi] < Elow+500))
            #     doPlot(trEXO=data["E_True"][Multi][filter], prEXO=data["E_EXO"][Multi][filter],
            #            trXv=data["E_True"][Multi][filter], prXv=data["E_CNN"][Multi][filter],
            #            fileOUT=fileOUT+fOUT_cal+"_"+Multi+"_"+str(Elow)+".pdf")
    else:
        doPlot(trEXO=data["E_True"], prEXO=data["E_EXO"], trXv=data["E_True"], prXv=data["E_CNN"], fileOUT=fileOUT+".pdf")

    print '===================================== Program finished =============================='


def doPlot(trEXO, prEXO, trXv, prXv, fileOUT):
    # Plot Range
    limit = 250

    n = (prEXO - trEXO).size
    print n

    if n == 476000:
        std = 0.00102490088454579823177649356950594768872742154553209283079481032110475742809435628537056046718576824687908469772
    elif n == 233308:
        std = 0.00146393193504558881460694807724313983900815301926964134878146897133674275814333498354469816367175626732147840865
    elif n == 242692:
        std = 0.00143535042719889728365847124103744647895517934928155671154341885600985852221409602041218185480114163940637255786
    else:
        std = 0.; print 'strange length. Press key'; raw_input('')

    # make Figure
    fig = plt.figure()

    # set size of Figure
    fig.set_size_inches(w=12*0.8, h=7*0.8)

    # add Axes
    ax1 = fig.add_axes([0, 0., .47, 0.75])
    #ax11 = fig.add_axes([0.02, 0.45, 0.03, 0.27])

    ax2 = fig.add_axes([0, 0.8, .47, 0.2])

    ax3 = fig.add_axes([0.5, 0., .12, 0.75])

    ax4 = fig.add_axes([0.68, 0., .47 /3 *2, 0.75])



    # ax1.hexbin(prEXO - trEXO, prXv - trXv, gridsize=(10*7,3*8), mincnt = 1, alpha = 0.7, vmin=0, vmax=100) norm=mpl.colors.LogNorm()
    plt1 = ax1.hexbin(prEXO - trEXO, prXv - trXv, gridsize=400, mincnt = 1, norm=mpl.colors.LogNorm(), cmap=plt.get_cmap('viridis'), linewidths=0.1)
    # fig.colorbar(plt1, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))


    # ax1.axis([-120, 120, -120, 120])
    # ax1.set_yticklabels([-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120])
    # ax1.set_xticklabels([-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120])
    ax1.set_xlim([-limit, limit])
    ax1.set_ylim([-limit, limit])
    ax1.grid()
    ax1.set_xlabel(r'$\mathregular{E_{Recon} - E_{True}}$ [keV]', fontsize = 14)
    ax1.set_ylabel(r'$\mathregular{E_{DNN} - E_{True}}$ [keV]', fontsize = 14)

    # fig.colorbar(plt1, cax=ax11, ticks=[0,25,50,75,100])
    #cbar = fig.colorbar(plt1, cax=ax11, ticks=mpl.ticker.LogLocator(subs=range(10)), format='')
    # cbar.set_label('counts', fontsize=14)
    # cbar = fig.colorbar(plt1, cax=ax11, ticks=mpl.ticker.LogLocator(subs=range(10)), norm=colors.LogNorm(vmin=1, vmax=100))



    # ax2.axis([-limit, limit, 1.e-6, 1.e-1])
    ax2.axis([-limit, limit, 0, 0.02])
    ax2.hist(prEXO - trEXO, bins=100, color = "red", range = (-limit,limit), normed = True, alpha = 0.4)
    # ax2.set_yscale("log", nonposy='clip')

    bins = range(-limit,limit)
    mu = np.mean(prEXO - trEXO)
    sigma2 = np.std(prEXO - trEXO - mu)
    # fit2 = mlab.normpdf( bins, mu, sigma)
    # ax2.plot(bins, fit2, 'b--', linewidth=3)
    print '%.4f $\pm$ %.4f' % (sigma2, sigma2 * std)

    ax2.text(-230, 0.015, '$\sigma$ =  %.2f(%d)' %(sigma2,np.ceil(sigma2*std*100)), color = "red", transform=ax2.transData, verticalalignment='center', horizontalalignment='left', fontsize=14)


    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid()






    ax3.axis([0, 0.02, -limit, limit])
    # ax3.axis([1.e-6, 1.e-1, -limit, limit])
    ax3.hist(prXv - trXv, bins=100, color ="blue", orientation="horizontal", range = (-limit,limit), normed = True, alpha = 0.4)
    # ax3.set_xscale("log", nonposx='clip')

    mu = np.mean(prXv - trXv)
    sigma3 = np.std(prXv - trXv - mu)
    # fit3 = mlab.normpdf( bins, mu, sigma)
    # ax3.plot(fit3, bins, 'g--', linewidth=3)

    print '%.4f $\pm$ %.4f' % (sigma3, sigma3 * std)
    ax3.text(0.013, 150, '$\sigma$ =  %.2f(%d)' %(sigma3,np.ceil(sigma3*std*100)), color = "blue", transform=ax3.transData, rotation=270, verticalalignment='center', horizontalalignment='left', fontsize=14)

    ax3.set_xticks([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.grid()




    # ax4.axis([-limit, limit, 1.e-6, 1.e-1])
    ax4.axis([-100, 100, 0, 0.02])
    ax4.hist(prXv - prEXO, bins=50, color ="green", range = (-100,100), normed = True, alpha = 0.4)
    # ax4.set_yscale("log", nonposy='clip')

    mu = np.mean(prXv - prEXO)
    sigma4 = np.std(prXv - prEXO - mu)
    # fit4 = mlab.normpdf( bins, mu, sigma)
    # ax4.plot(bins, fit4, 'r--', linewidth=3)

    print '%.4f $\pm$ %.4f' %(sigma4,sigma4*std)

    ax4.text(-90, 0.0185, '$\sigma$ =  %.2f(%d)' %(sigma4,np.ceil(sigma4*std*100)), color = "green", transform=ax4.transData, verticalalignment='center', horizontalalignment='left', fontsize=14)


    ax4.set_xlabel(r'$\mathregular{E_{DNN} - E_{Recon}}$ [keV]', fontsize = 14)
    ax4.set_yticks([])
    ax4.set_xticks([-100,-50,0,50,100])
    # ax4.set_xticks([-200, -100, 0, 100, 200])
    ax4.grid()



    green_patch = mpatches.Patch(color='blue', label='DNN - True')
    blue_patch = mpatches.Patch(color='red', label='EXO Recon - True')
    red_patch = mpatches.Patch(color='green', label='DNN - EXO Recon')

    ax1.legend(bbox_to_anchor=(1.8, 1.33), handles=[blue_patch,green_patch,red_patch], fontsize = 14)
    # ax2.legend(handles=[blue_patchsi], loc=2)
    # ax3.legend(handles=[green_patchsi], loc=2)
    # ax4.legend(handles=[red_patchsi], loc=2)

    fig.savefig(fileOUT, bbox_inches='tight')
    # plt.show()
    # raw_input("")

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()