#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import h5py
import time
import random
import os
import argparse
import warnings
import cPickle as pickle

def main():
    args = make_organize()
    data = open_data(args=args)
    get_reconstructed_spectrum(args=args, data=data)
    # final_plots(args=args, data=data)
    print '===================================== Program finished =============================='

def make_organize():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-out', dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderOUT Path')
    parser.add_argument('-in', dest='foldersIN', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', nargs="*", help='folderIN Paths')
    parser.add_argument('-epochs', dest='epochs', default='final', nargs="*", help='Load weights from Epoch')
    parser.add_argument('-sources', dest='sources', default=['th', 'ra', 'co'], nargs="*", choices=["th", "co", "ra"], help='sources for training (ra,co,th)')
    parser.add_argument('--reconstruct', dest='reconstruct', action='store_true', help='Compare to EXO200 Reconstruction')
    parser.add_argument('--predict', dest='predict_data', action='store_true', help='Predict REAL Data')
    args, unknown = parser.parse_known_args()

    args.folderOUT = os.path.join(args.folderOUT,'')
    for idx, folderIN in enumerate(args.foldersIN):
        print 'Input Folder:\t\t', '/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/', folderIN, '/0physics-data/', args.epochs[idx]

    print 'Output Folder:\t\t'  , args.folderOUT
    return args

def open_data(args):
    data = {}
    for idx,input in enumerate(args.foldersIN):
        input_full = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join('/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/', input),''),'0physics-data/'),args.epochs[idx]),'')
        print input_full
        print input_full + "spectrum_events_" + args.epochs[idx] + "_th.p"
        try:
            data[str(input)] = pickle.load(open(input_full + "spectrum_events_" + args.epochs[idx] + "_th.p", "rb"))
        except IOError:
            print input_full, 'save.p\t\tnot found!'
    print data.keys()
    for key in data.keys():
        print data[key].keys()
    return data

def get_reconstructed_spectrum(args, data):
    from scipy.optimize import curve_fit
    for idx, run in enumerate(data.keys()):
        print data[run].keys()
        for energy in data[run].keys():
            print energy, type(energy)
            if energy == 'E_Light':
                continue
            if energy == 'E_DCNN':
                colors = ['r', 'b', 'g', 'k', 'y']
                name = str(run)
                color = 'limegreen'
                color = colors[idx]
            if energy == 'E_EXO':
                if idx != 2:
                    continue
                name = 'EXO200'
                color = 'blue'
                colors = ['r', 'b', 'g', 'k', 'y']
                color = colors[idx+1]

            hist, bin_edges = np.histogram(data[run][energy], bins=140, range=(650, 3500), density=False)
            norm_factor = float(len(data[run][energy]))
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            peak = np.argmax(hist[np.digitize(2400, bin_centres):np.digitize(2800, bin_centres)]) + np.digitize(2400, bin_centres)
            coeff = [hist[peak], bin_centres[peak], 100., -0.005, 0.0]
            for i in range(5):
                try:
                    low = np.digitize(coeff[1] - (4 * abs(coeff[2])), bin_centres)
                    up = np.digitize(coeff[1] + (4 * abs(coeff[2])), bin_centres)
                    coeff, var_matrix = curve_fit(gaussErf, bin_centres[low:up], hist[low:up], p0=coeff)
                    # lambda x, sigma, A, off: gauss(x, mean, sigma, A, off)
                    coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
                except:
                    print 'Gauss fit did not work\t', i
                    coeff, coeff_err = [0.001, 2614, 50., -0.005, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]
            delE = abs(coeff[2])/coeff[1]*100.0
            delE_err = delE * np.sqrt((coeff_err[1]/coeff[1])**2+(coeff_err[2]/coeff[2])**2)
            plt.step(bin_centres, hist/norm_factor, where='mid', color=color,
                     label='%s\t$\mu=%.1f \pm %.1f$' % (name, coeff[1], coeff_err[1]))
            plt.plot(bin_centres[low:up], gaussErf(bin_centres, *coeff)[low:up]/norm_factor, lw=2, color=color,
                     label='Resolution $(\sigma)$: $%.2f \pm %.2f$ %%  ' % (delE, delE_err))
            print 'Fitted %s Peak (@2614):\t%.2f +- %.2f \t\t Resolution: %.2f%% +- %.2f%%' % (name, coeff[1], abs(coeff[2]), delE, delE_err)
    plt.legend(loc="best")
    plt.axvline(x=2614, lw=2, color='black')
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=650, xmax=3000)
    plt.ylim(ymin=5.e-5, ymax=1.e-1)
    plt.savefig(args.folderOUT + 'spectrum_combined.pdf')
    plt.close()
    plt.clf()
    return

def final_plots(args, data):
    print 'final plots \t start'

    key_list = list(set([ key for key_data in
                         data.keys() for key_epoch in
                         data[key_data].keys() for key in
                         data[key_data][key_epoch].keys() ]))
    print key_list

    data_sort, epoch = {}, {}
    for key_data in data.keys():
        print key_data
        data_sort[key_data] = {}
        epoch[key_data] = []
        for key in key_list:
            data_sort[key_data][key] = []
        for key_epoch in data[key_data].keys():
            epoch[key_data].append(int(key_epoch))
            for key in key_list:
                try:
                    data_sort[key_data][key].append(data[key_data][key_epoch][key])
                except KeyError:
                    data_sort[key_data][key].append(0.0)

        order = {}
        order[key_data] = np.argsort(epoch[key_data])
        epoch[key_data] = np.array(epoch[key_data])[order[key_data]]

        for key in key_list:
            data_sort[key_data][key] = np.array(data_sort[key_data][key])[order[key_data]]

    colors = ['r','b','g','k','y']

    try:
        for idx, key_data in enumerate(data_sort.keys()):
            plt.plot(epoch[key_data], data_sort[key_data]['loss'], color=colors[idx], label='%s-%s' % (key_data, 'train'), ls='--')
            plt.plot(epoch[key_data], data_sort[key_data]['val_loss'], color=colors[idx], label='%s-%s' % (key_data, 'val'))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.gca().set_yscale('log')
        plt.legend(loc="best", ncol=2)
        plt.savefig(args.folderOUT + 'loss.pdf')
        plt.clf()
        plt.close()

        for idx, key_data in enumerate(data_sort.keys()):
            plt.plot(epoch[key_data], data_sort[key_data]['mean_absolute_error'], color=colors[idx], label='%s-%s' % (key_data, 'train'), ls='--')
            plt.plot(epoch[key_data], data_sort[key_data]['val_mean_absolute_error'], color=colors[idx], label='%s-%s' % (key_data, 'val'))
        plt.grid(True)
        plt.legend(loc="best", ncol=2)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(args.folderOUT + 'mean_absolute_error.pdf')
        plt.clf()
        plt.close()
    except:
        print 'no loss / mean_err plot possible'

    exit()

    plt.errorbar(epoch, obs_sort['peak_pos'][:,0], xerr=0.5, yerr=obs_sort['peak_pos'][:,1], fmt="none", lw=2)
    plt.axhline(y=2614, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Energy [keV]')
    plt.savefig(args.folderOUT + 'ZZZ_Peak.pdf')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['peak_sig'][:,0], xerr=0.5, yerr=obs_sort['peak_sig'][:,1], fmt="none", lw=2)
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Width [keV]')
    plt.savefig(args.folderOUT + 'ZZZ_Width.pdf')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['resid_pos'][:, 0], xerr=0.5, yerr=obs_sort['resid_pos'][:, 1], fmt="none", lw=2)
    plt.axhline(y=0, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Offset [keV]')
    plt.savefig(args.folderOUT + 'ZZZ_Offset.pdf')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['resid_sig'][:, 0], xerr=0.5, yerr=obs_sort['peak_sig'][:, 1], fmt="none", lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Width [keV]')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.savefig(args.folderOUT + 'ZZZ_Width2.pdf')
    plt.clf()
    plt.close()

    print 'final plots \t end'
    return

def gauss(x, A, mu, sigma, off):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + off

def gauss_zero(x, A, mu, sigma):
    return gauss(x, A, mu, sigma, 0.0)

def erf(x, B, mu, sigma):
    import scipy.special
    return B * scipy.special.erf((x - mu) / (np.sqrt(2) * sigma)) + abs(B)

def shift(a, b, mu, sigma):
    return np.sqrt(2./np.pi)*float(b)/a*sigma

def gaussErf(x, A, mu, sigma, B, off):
    return gauss(x, mu=mu, sigma=sigma, A=A, off=off) + erf(x, B=B, mu=mu, sigma=sigma)

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
