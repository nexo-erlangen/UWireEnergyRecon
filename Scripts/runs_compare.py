#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import h5py
import math
import time
import random
import os
import argparse
import warnings
import cPickle as pickle
from collections import OrderedDict
import RotationAngle_custom as rot

def main():
    args = make_organize()
    if not args.data and not args.rotate:
        data_MC = open_data_MC(args=args)
        data_MC_sort, epoch_sort = sort_data_MC(args=args, data=data_MC)
        final_plots_MC(args=args, data=data_MC_sort, epoch=epoch_sort)
    else:
        data = open_data(args=args)
        if not args.rotate:
            spectrum_data(args=args, data=data)
        else:
            rotate_energy(args=args, data=data)
    print '===================================== Program finished =============================='

def make_organize():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-out', dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderOUT Path')
    parser.add_argument('-in', dest='foldersIN', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', nargs="*", help='folderIN Paths')
    parser.add_argument('-epochs', dest='epochs', nargs="*", help='Load weights from Epoch')
    parser.add_argument('-label', dest='name', nargs="*", help='Plot Label')
    parser.add_argument('--data', dest='data', action='store_true', help='Produce Data Plots')
    parser.add_argument('--rotate', dest='rotate', action='store_true', help='Rotate Energy')
    parser.add_argument('-source', dest='source', default='th', choices=["th", "co", "ra"], help='Source (ra,co,th)')
    args, unknown = parser.parse_known_args()

    args.epochs = [str(i).zfill(3) for i in args.epochs]
    args.folderOUT = os.path.join(args.folderOUT,'')
    for folderIN in args.foldersIN:
        print 'Input Folder:\t\t', '/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/', folderIN, '/'

    print 'Output Folder:\t\t'  , args.folderOUT
    return args

def open_data_MC(args):
    data, name = {}, {}
    for idx, input in enumerate(args.foldersIN):
        input_full = os.path.join(os.path.join('/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/', input),'')
        try:
            data[str(input)] = pickle.load(open(input_full + "save.p", "rb"))
            try: name[str(input)] = args.name[idx]
            except: name[str(input)] = str(input)
        except IOError:
            print input_full, 'save.p\t\tnot found!'
    print data.keys()
    args.name = name
    return data

def open_data(args):
    data, name, epochs = OrderedDict(), {}, {}
    for idx,input in enumerate(args.foldersIN):
        input_full = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join('/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/', input),''),'0physics-data/'),args.epochs[idx]),'')
        print 'open file:\t\t' + input_full + "spectrum_events_" + args.epochs[idx] + "_" + args.source + ".p"
        try:
            data[str(input)+"_"+str(args.epochs[idx])] = pickle.load(open(input_full + "thms-S5/spectrum_events_" + args.epochs[idx] + "_thms-S5.p", "rb"))
            try: name[str(input) + "_" + str(args.epochs[idx])] = args.name[idx]
            except: name[str(input) + "_" + str(args.epochs[idx])] = str(input) + "_" + str(args.epochs[idx])
            try: epochs[str(input) + "_" + str(args.epochs[idx])] = args.epochs[idx]
            except: epochs[str(input) + "_" + str(args.epochs[idx])] = str(input) + "_" + str(args.epochs[idx])
        except IOError:
            print input_full, '\t\tnot found!'
    args.name = name
    args.epochs = epochs
    return data

def sort_data_MC(args, data):
    key_list = list(set([key for key_data in
                         data.keys() for key_epoch in
                         data[key_data].keys() for key in
                         data[key_data][key_epoch].keys()]))
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
    return data_sort, epoch

def final_plots_MC(args, data, epoch):
    print 'final plots \t start'

    # colors = ['r','b','g','k','y']

    for idx, key_data in enumerate(args.foldersIN):
        label = args.name[key_data]
        # plt.plot(epoch[key_data], data[key_data]['loss'], color=colors[idx], label='%s-%s' % (label, 'train'), ls='--')
        # plt.plot(epoch[key_data], data[key_data]['val_loss'], color=colors[idx], label='%s-%s' % (label, 'val'))
        plt.plot(epoch[key_data], data[key_data]['loss'], color=color(args.name,idx), ls='--')
        plt.plot(epoch[key_data], data[key_data]['val_loss'], color=color(args.name,idx), label='%s' % (label))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.gca().set_yscale('log')
    plt.grid(True, axis='x')
    plt.grid(True, which="major", ls="-", axis='y')
    plt.grid(True, which="minor", ls=":", axis='y')
    plt.ylim(ymin=1.e3, ymax=5.e4)
    plt.xlim(xmax=35)
    plt.legend(loc="best", ncol=1)
    plt.savefig(args.folderOUT + 'loss.pdf')
    plt.clf()
    plt.close()

    for idx, key_data in enumerate(args.foldersIN):
        label = args.name[key_data]
        # plt.plot(epoch[key_data], data[key_data]['mean_absolute_error'], color=colors[idx], label='%s-%s' % (label, 'train'), ls='--')
        # plt.plot(epoch[key_data], data[key_data]['val_mean_absolute_error'], color=colors[idx], label='%s-%s' % (label, 'val'))
        plt.plot(epoch[key_data], data[key_data]['mean_absolute_error'], color=color(args.name,idx), ls='--')
        plt.plot(epoch[key_data], data[key_data]['val_mean_absolute_error'], color=color(args.name,idx), label='%s' % (label))
    plt.grid(True)
    plt.legend(loc="best", ncol=1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc="best", ncol=1)
    plt.ylim(ymin=25., ymax=75.)
    plt.xlim(xmax=35)
    plt.savefig(args.folderOUT + 'mean_absolute_error.pdf')
    plt.clf()
    plt.close()

    for idx, key_data in enumerate(args.foldersIN):
        label = args.name[key_data]
        plt.errorbar(epoch[key_data], data[key_data]['peak_pos'][:,0], xerr=0.5, yerr=data[key_data]['peak_pos'][:,1], label='%s' % (label), color=color(args.name,idx), fmt=".", lw=2)
    plt.axhline(y=2614, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0, xmax=35)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Energy [keV]')
    plt.legend(loc="best", ncol=1)
    plt.savefig(args.folderOUT + 'ZZZ_Peak.pdf')
    plt.clf()
    plt.close()

    for idx, key_data in enumerate(args.foldersIN):
        label = args.name[key_data]
        plt.errorbar(epoch[key_data], data[key_data]['peak_sig'][:,0], xerr=0.5, yerr=data[key_data]['peak_sig'][:,1], label='%s' % (label),fmt=".", lw=2, color=color(args.name,idx))
    plt.grid(True)
    plt.xlim(xmin=0, xmax=35)
    plt.ylim(ymin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Width [keV]')
    plt.legend(loc="best", ncol=1)
    plt.savefig(args.folderOUT + 'ZZZ_Width.pdf')
    plt.clf()
    plt.close()

    for idx, key_data in enumerate(args.foldersIN):
        label = args.name[key_data]
        plt.errorbar(epoch[key_data], data[key_data]['resid_pos'][:, 0], xerr=0.5, yerr=data[key_data]['resid_pos'][:, 1], label='%s' % (label), fmt=".", lw=2, color=color(args.name,idx))
    plt.axhline(y=0, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0, xmax=35)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Offset [keV]')
    plt.legend(loc="best", ncol=1)
    plt.savefig(args.folderOUT + 'ZZZ_Offset.pdf')
    plt.clf()
    plt.close()

    for idx, key_data in enumerate(args.foldersIN):
        label = args.name[key_data]
        plt.errorbar(epoch[key_data], data[key_data]['resid_sig'][:, 0], xerr=0.5, yerr=data[key_data]['peak_sig'][:, 1], label='%s' % (label), fmt=".", lw=2, color=color(args.name,idx))
    plt.xlabel('Epoch')
    plt.ylabel('Residual Width [keV]')
    plt.grid(True)
    plt.xlim(xmin=0, xmax=35)
    plt.ylim(ymin=0)
    plt.legend(loc="best", ncol=1)
    plt.savefig(args.folderOUT + 'ZZZ_Width2.pdf')
    plt.clf()
    plt.close()

    print 'final plots \t end'
    return

def spectrum_data(args, data):
    E_EXO = np.array([])
    for idx, run in enumerate(data.keys()):
        print idx, run, args.epochs[run], args.name[run]
        for energy in data[run].keys():
            if energy == 'E_Light':
                continue
            if energy == 'E_EXO':
                E_EXO = np.append(E_EXO, data[run][energy])
            if energy == 'E_CNN':
                name = args.name[run]
                fit_spectrum(energy=data[run][energy], name=name, color=color(args.name, idx))
    fit_spectrum(energy=E_EXO, name='EXO-200', color='k')

    plt.legend(loc="lower left")
    plt.axvline(x=2614, lw=2, color='black')
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=700, xmax=3500)
    plt.ylim(ymin=1.e-4, ymax=2.e-2)
    plt.savefig(args.folderOUT + 'spectrum_combined.pdf')
    plt.close()
    plt.clf()
    return

def rotate_energy(args, data):
    print 'rotate'
    for idx, run in enumerate(data.keys()):
        print idx, run, args.epochs[run], args.name[run]
        E_Light = data[run]['E_Light']
        try:
            spec = pickle.load(open(args.folderOUT + 'RotationAngle_' + args.epochs[run] + '.p', "rb"))
        except IOError:
            spec = {}
            for E_List_str in ['E_DCNN', 'E_EXO']:
                E_List = data[run][E_List_str]
                spec[E_List_str]  = rot.Run(EnergyList2D_ss=zip(E_List , E_Light), prefix_local=args.folderOUT+E_List_str)
                print E_List_str, ':\t\t', spec[E_List_str], '\n'
            pickle.dump(spec, open(args.folderOUT + 'RotationAngle_' + args.epochs[run] + '.p', "wb"))

        for E_List_str in ['E_DCNN', 'E_EXO']:
            E_List = data[run][E_List_str]
            E_List_Rot = E_List * math.cos(spec[E_List_str]['Theta_ss'][0]) + E_Light * math.sin(spec[E_List_str]['Theta_ss'][0])
            # if E_List_str == 'E_EXO':
            #     spec[E_List_str]['E_rot_cal'] = E_List_Rot/3.59573580522/0.441851568477/0.938475113847/0.999947756503/0.999999434294/1.0000006072/0.999999434294
            # if E_List_str == 'E_DCNN':
            #     spec[E_List_str]['E_rot_cal'] = E_List_Rot/1.52218370556/0.999987990297/0.999998065571/0.999999941019/0.999999941019/0.999999941019/0.999999393048
            spec[E_List_str]['E_rot_cal'] = calibrate_spectrum(args=args, data=E_List_Rot, name=E_List_str)

        for E_List_str in ['E_DCNN', 'E_EXO']:
            if E_List_str == 'E_EXO':
                col = 'firebrick'
                label = 'Standard'
            if E_List_str == 'E_DCNN':
                col = 'blue'
                label = 'Conv. NN'
            fit_spectrum(energy=spec[E_List_str]['E_rot_cal'], name=label, color=col)
        plt.axvline(x=2614, lw=2, color='black')
        plt.grid(True)
        plt.xlim(xmin=2400, xmax=2800)
        plt.ylim(ymin=1.e-4, ymax=5.e-2)
        plt.savefig(args.folderOUT + args.source + '_spectrum_combined_zoom_lin.pdf')
        plt.gca().set_yscale('log')
        plt.savefig(args.folderOUT + args.source + '_spectrum_combined_zoom_log.pdf')
        plt.legend(loc="upper left")
        plt.xlabel('Energy [keV]')
        plt.ylabel('Counts')
        plt.xlim(xmin=1000, xmax=3000)
        plt.ylim(ymin=1.e-4, ymax=5.e-2)
        plt.savefig(args.folderOUT + args.source + '_spectrum_combined_log.pdf')
        plt.legend(loc="upper left")
        plt.gca().set_yscale('linear')
        plt.ylim(ymin=0.0, ymax=0.013)
        plt.savefig(args.folderOUT + args.source + '_spectrum_combined_lin.pdf')
        plt.clf()
        plt.close()

        for E_List_str in ['E_DCNN', 'E_EXO']:
            if E_List_str == 'E_EXO':
                col = 'firebrick'
                label = 'Standard'
            if E_List_str == 'E_DCNN':
                col = 'blue'
                label = 'Conv. NN'
            TestResolution_ss = np.array(spec[E_List_str]['TestResolution_ss'])*100.
            par0 = spec[E_List_str]['Par0'][0]
            par1 = spec[E_List_str]['Par1'][0]
            par2 = spec[E_List_str]['Theta_ss'][0]
            print E_List_str, '\t', par0*100., '\t', par2
            limit = 0.07
            x = np.arange(par2-limit, par2+limit, 0.005)
            plt.errorbar(spec[E_List_str]['TestTheta_ss'], TestResolution_ss[:, 0], yerr=TestResolution_ss[:, 1], color=col, fmt="o", label='%s (%.3f%%)'%(label, par0*100.))
            plt.plot(x, parabola(x, par0, par1, par2)*100., color='k', lw=2)
        plt.grid(True)
        plt.xlim(xmin=0.1, xmax=0.2)
        plt.ylim(ymin=1.6, ymax=2.0)
        plt.savefig(args.folderOUT + args.source + '_resolution_vs_angle_zoom.pdf')
        plt.legend(loc="best")
        plt.xlabel('Theta [rad]')
        plt.ylabel('Resolution @ Th228 peak [%]')
        plt.xlim(xmin=0.0, xmax=0.8)
        plt.ylim(ymin=1.5, ymax=4.5)
        plt.savefig(args.folderOUT + args.source + '_resolution_vs_angle.pdf')
        plt.clf()
        plt.close()
    return

def plot_fit_spectrum(args, data, name, color='k'):
    hist, bin_edges = np.histogram(data, bins=1000, range=(0, 10000), density=False)
    norm_factor = float(len(data))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i in range(len(hist)-2, 0, -1):
        if hist[i + 1] <= 0: continue
        sigma = math.sqrt(hist[i] + hist[i + 1])
        if abs((hist[i + 1] - hist[i]) / sigma) >= 1.5:
            peak = i + 1
            break
    # peak = np.argmax(hist[np.digitize(2400, bin_centres):np.digitize(2800, bin_centres)]) + np.digitize(2400, bin_centres)
    coeff = [hist[peak], bin_centres[peak], 100., -0.005, 0.0]
    from scipy.optimize import curve_fit
    for i in range(5):
        try:
            low = np.digitize(coeff[1] - (4 * abs(coeff[2])), bin_centres)
            up = np.digitize(coeff[1] + (4 * abs(coeff[2])), bin_centres)
            coeff, var_matrix = curve_fit(gaussErf, bin_centres[low:up], hist[low:up], p0=coeff)
            coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
        except:
            print 'fit did not work\t', i
            coeff, coeff_err = [hist[peak], bin_centres[peak], 50.0*(i+1), -0.005, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]
        delE = abs(coeff[2]) / coeff[1] * 100.0
        delE_err = delE * np.sqrt((coeff_err[1] / coeff[1]) ** 2 + (coeff_err[2] / coeff[2]) ** 2)
    plt.step(bin_centres, hist / norm_factor, where='mid', color=color,
             label='%s\t$\mu=%.1f \pm %.1f$' % (name, coeff[1], coeff_err[1]))
    plt.plot(bin_centres[low:up], gaussErf(bin_centres, *coeff)[low:up] / norm_factor, lw=2, color=color,
             label='Resolution $(\sigma)$: $%.2f \pm %.2f$ %%  ' % (delE, delE_err))
    plt.axvline(x=2614, lw=2, color='black')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.legend(loc="best")
    plt.xlim(xmin=min(data), xmax=max(data))
    plt.ylim(ymin=1.e-4, ymax=1.e-1)
    plt.grid(True)
    plt.gca().set_yscale('log')
    plt.savefig(args.folderOUT + 'spectrum_' + name + '.pdf')
    plt.clf()
    plt.close()
    print 'Fitted %s Peak (@2614):\t\t%.2f +- %.2f' % (name, coeff[1], abs(coeff[2]))
    print 'Fitted %s Reso (@2614):\t\t%.2f +- %.2f' % (name, delE, delE_err)
    return (coeff[1], coeff_err[1]), (abs(coeff[2]), coeff_err[2])

def fit_spectrum(energy, name, color):
    from scipy.optimize import curve_fit
    hist, bin_edges = np.histogram(energy, bins=1000, range=(0, 10000), density=False)
    norm_factor = float(len(energy))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak = np.argmax(hist[np.digitize(2400, bin_centres):np.digitize(2800, bin_centres)]) + np.digitize(2400,bin_centres)
    coeff = [hist[peak], bin_centres[peak], 100., -0.005]
    for i in range(5):
        try:
            low = np.digitize(coeff[1] - (4 * abs(coeff[2])), bin_centres)
            up = np.digitize(coeff[1] + (4 * abs(coeff[2])), bin_centres)
            coeff, var_matrix = curve_fit(gaussErf, bin_centres[low:up], hist[low:up], p0=coeff)
            # lambda x, sigma, A, off: gauss(x, mean, sigma, A, off)
            coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
        except:
            print 'Gauss fit did not work\t', i
            coeff, coeff_err = [0.001, 2614, 50., -0.005], [0.0, 0.0, 0.0, 0.0]
    delE = abs(coeff[2]) / coeff[1] * 100.0
    delE_err = delE * np.sqrt((coeff_err[1] / coeff[1]) ** 2 + (coeff_err[2] / coeff[2]) ** 2)
    plt.step(bin_centres, hist / norm_factor, where='mid', color=color,label='%s: $%.2f$ %% $(\sigma)$\t$\mu=%.1f$' % (name, delE, coeff[1]))
    plt.plot(bin_centres[low:up], gaussErf(bin_centres, *coeff)[low:up] / norm_factor, lw=2,color=color)
    print 'Fitted %s Peak (@2614):\t%.2f +- %.2f \t\t Resolution: %.2f%% +- %.2f%%' % (name, coeff[1], abs(coeff[2]), delE, delE_err)
    return

def calibrate_spectrum(args, data, name):
    mean_recon = (2614.0, 0.0)
    for i in range(7):
        data = data/(mean_recon[0]/2614.0)
        mean_recon, sig_recon = plot_fit_spectrum(args=args, data=data, name=(name+ '_' + str(i)))
        print 'mean value\t', mean_recon[0]/2614.0
    return data

def color(all, idx):
    cmap = matplotlib.cm.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=len(all)-1)
    return cmap(norm(idx))

def gauss(x, A, mu, sigma, off):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + off

def gauss_zero(x, A, mu, sigma):
    return gauss(x, A, mu, sigma, 0.0)

def parabola(x, par0, par1, par2):
    return par0 + par1 * ((x - par2) ** 2)

def erf(x, B, mu, sigma):
    import scipy.special
    return B * scipy.special.erf((x - mu) / (np.sqrt(2) * sigma)) + abs(B)

def shift(a, b, mu, sigma):
    return np.sqrt(2./np.pi)*float(b)/a*sigma

def gaussErf(x, A, mu, sigma, B):
    return gauss_zero(x, mu=mu, sigma=sigma, A=A) + erf(x, B=B, mu=mu, sigma=sigma)

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
