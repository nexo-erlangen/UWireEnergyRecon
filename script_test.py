#!/usr/bin/env python

import numpy as np
import h5py
import time
import os
import cPickle as pickle
from sys import path
path.append('/home/vault/capm/sn0515/PhD/Th_U-Wire/Scripts')
import script_plot as plot

def main():
    args  = make_organize()
    files = split_data(args=args)

    try:
        from keras.models import load_model
    except ImportError:
        print 'tensorflow cannot be loaded'
    for epoch in args.epochs:
        print epoch
        try:
            try:
                model = load_model(args.folderOUT + "models/model-initial.hdf5")
                model.load_weights(args.folderOUT + "models/weights-" + epoch + ".hdf5")
            except:
                model = load_model(args.folderOUT + "models/model-" + epoch + ".hdf5")
            print model.summary()
        except: print 'model not found' ; model = None #; continue

        args.source = "".join(sorted([k for k, v in files.items() if v]))
        if args.predict_mc:     folderOUT = args.folderOUT + "1validation-data/" + args.source + "-" + args.position + "/"
        if args.predict_data:   folderOUT = args.folderOUT + "0physics-data/" + epoch + "/" + args.source + "-" + args.position + "/"
        os.system("mkdir -p %s " % (folderOUT))
        data = get_events(args=args, files=files, model=model, fOUT=(folderOUT + "spectrum_events_" + epoch + "_" + args.source + "-" + args.position + ".p"))

        if args.predict_mc:
            print 'predict MC events'
            plot.make_plots(folderOUT=folderOUT, dataIn=data, epoch=epoch, sources=args.source, position=args.position, mode='eval')
        elif args.predict_data:
            print 'predict Data events'
            if not args.rotate:
                make_plots_data(folderOUT=folderOUT, dataIn=data, epoch=epoch, sources=args.source, position=args.position)
            else:
                print 'rotate energy axis'
                if 'th' in args.source and args.position=='S5' :
                    print 'self:\t True'
                    data_rot = rotate_energy(folder=folderOUT, data=data, epoch=epoch, new=args.new, self=True)
                    plot.plot_rotationAngle_resolution(data=data_rot, fOUT=(folderOUT + 'Rotation_' + args.source + '_' + args.position + '_resolution_vs_angle.pdf'))
                else:
                    print 'self:\t False'
                    folderIN = args.folderOUT + "0physics-data/" + epoch + "/th" + args.multi.lower() + "-S5/"
                    print folderIN
                    data_rot = rotate_energy(folder=folderIN, data=data, epoch=epoch, new=args.new, self=False)

                for E_List_str in ['E_CNNPur', 'E_EXOPur']:
                    print 'Peak Position', E_List_str, 'SS', data_rot[E_List_str]['PeakPos_ss'][0]/2614.5
                    print 'Peak Position', E_List_str, 'MS', data_rot[E_List_str]['PeakPos_ms'][0]/2614.5
                    data_rot[E_List_str]['E_Rot_SS_cal'] = data_rot[E_List_str]['E_Rot_ss'] * (2614.5 / data_rot[E_List_str]['PeakPos_ss'][0])
                    data_rot[E_List_str]['E_Rot_MS_cal'] = data_rot[E_List_str]['E_Rot_ms'] * (2614.5 / data_rot[E_List_str]['PeakPos_ms'][0])
                    data_rot[E_List_str]['E_Rot_SSMS_cal'] = np.concatenate((data_rot[E_List_str]['E_Rot_SS_cal'], data_rot[E_List_str]['E_Rot_MS_cal']))

                for Multi in ['SS', 'MS', 'SSMS']:
                    plot.plot_spectrum(data_CNN=data_rot['E_CNNPur']['E_Rot_' + Multi + '_cal'],
                                       data_EXO=data_rot['E_EXOPur']['E_Rot_' + Multi + '_cal'], peakpos=2614.5, fit=('th' in args.source), isMC=False,
                                       fOUT=(folderOUT + 'Rotation_spectrum_' + args.source + '_' + args.position + '_' + Multi + '_final.pdf'))
                    plot.plot_scatter_hist2d(E_x=data_rot['E_EXOPur']['E_Rot_' + Multi + '_cal'], E_y=data_rot['E_CNNPur']['E_Rot_' + Multi + '_cal'],
                                        name_x='EXO Recon', name_y='Neural Network',
                                        fOUT=(folderOUT + 'Rotation_scatter_' + args.source + '_' + args.position + '_' + Multi + '.pdf'))
                    plot.plot_residual_hist2d(E_x=data_rot['E_EXOPur']['E_Rot_' + Multi + '_cal'], E_y=data_rot['E_CNNPur']['E_Rot_' + Multi + '_cal'],
                                         name_x='EXO Recon', name_y='Neural Network',
                                         fOUT=(folderOUT + 'Rotation_residual_' + args.source + '_' + args.position + '_' + Multi + '.pdf'))


    print '===================================== Program finished =============================='

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def make_organize():
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-model' , dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='Model Path')
    parser.add_argument('-in'    , dest='folderIN' , default='/home/vault/capm/sn0515/PhD/Th_U-Wire/Data_MC', help='MC/Data Path')
    parser.add_argument('-epochs', dest='epochs'   , default=['final'], nargs="*", help='Load weights from Epoch')
    parser.add_argument('-source', dest='sources'  , default=['thms'], nargs="*", choices=["thss", "coss", "rass", "gass", "gams", "unss", 'thms', 'unms', 'rams', 'coms', 'thms2000', 'thms4500'], help='sources for training (ra,co,th,ga,un)')
    parser.add_argument('-position', dest='position', default='S5', choices=['S2', 'S5', 'S8'], help='sources position')
    parser.add_argument('-multi' , dest='multi', default='MS', choices=['SS', 'MS'], help='event multiplicity')
    parser.add_argument('-events', dest='events'   , default=10000, type=int, help='number of events to reconstruct')

    parser.add_argument('--mc'    , dest='predict_mc'   , action='store_true', help='Predict MC Data')
    parser.add_argument('--data'  , dest='predict_data' , action='store_true', help='Predict REAL Data')
    parser.add_argument('--gamma' , dest='predict_gamma', action='store_true', help='Predict MC Gamma Data')
    parser.add_argument('--rotate', dest='rotate'       , action='store_true', help='Rotate Energy')
    parser.add_argument('--new'   , dest='new'          , action='store_true', help='Process new events')
    parser.add_argument('--range' , dest='epoch_range'  , action='store_true', help='range of epochs')
    args, unknown = parser.parse_known_args()

    args.label = {'thss': "Th228-SS", 'rass': "Ra226-SS", 'coss': "Co60-SS", 'gass': "Gamma-SS", 'unss': "Uniform-SS",
                  'thms': "Th228-MS", 'rams': "Ra226-MS", 'coms': "Co60-MS", 'thms2000': "Th228-MS", 'thms4500': "Th228-MS",}
    list_mc   = ['thss','rass','coss','gass', 'gams', 'unss', 'gams', 'thms', 'unms']
    list_data = ['thss','rass','coss', 'thms','rams','coms', 'thms2000', 'thms4500']

    if args.epoch_range:
        if (len(args.epochs) > 1):
            epoch_min = int(min(args.epochs))
            epoch_max = int(max(args.epochs))+1
            args.epochs = [ str(i).zfill(3) for i in range(epoch_min,epoch_max) ]
        else: print 'specify at least 2 epochs for --range' ; exit()
    else: args.epochs = [ str(i).zfill(3) for i in args.epochs ]

    if (args.predict_mc and args.sources == ['gass']) or args.predict_gamma:
        args.predict_gamma = True
        args.predict_mc = True
        if args.multi == 'SS': args.sources = ['gass']
        if args.multi == 'MS': args.sources = ['gams']
    if args.rotate:
        args.predict_data = True
        # if args.multi == 'SS' and 'thss' not in args.sources: args.sources.append('thss')
        # if args.multi == 'MS' and 'thms' not in args.sources: args.sources.append('thms')
    if args.predict_mc and args.rotate: print 'mc and rotate active!' ; exit()
    if args.predict_data and args.predict_mc: print 'data and mc active!' ; exit()
    if args.predict_mc and 'S5' not in args.position: print 'mc is only possible at S5!'; exit()

    if args.predict_mc:
        args.folderIN = '/home/vault/capm/sn0515/PhD/Th_U-Wire/Data_MC'
        if not set(args.sources).issubset(set(list_mc)): print "wrong sources!"; exit()
    elif args.predict_data:
        args.folderIN = '/home/vault/capm/sn0515/PhD/Th_U-Wire/Data'
        if not set(args.sources).issubset(set(list_data)): print "wrong sources!"; exit()
    else: print 'no option mc/data chosen' ; exit()

    args.folderOUT  = os.path.join(args.folderOUT,'')

    print 'Predict MC:\t\t'    , args.predict_mc
    print 'Predict Gamma:\t\t' , args.predict_gamma
    print 'Predict Data:\t\t'  , args.predict_data
    print 'Rotate Data:\t\t'   , args.rotate
    print 'Sources:\t\t'       , args.sources
    print 'Position:\t\t'      , args.position
    print 'Epoch(s):\t\t'      , args.epochs
    print 'Output Folder:\t\t' , args.folderOUT

    return args

def split_data(args):
    files = {}
    if args.predict_mc:
        endings, folderIN = {'thss': "Th228_Wfs_SS_" + args.position + "_MC/",
                             'rass': "Ra226_Wfs_SS_" + args.position + "_MC/",
                             'coss': "Co60_Wfs_SS_" + args.position + "_MC/",
                             'unss': "Uniform_Wfs_SS_" + args.position + "_MC/",
                             'gass': "Gamma_Wfs_SS_" + args.position + "_MC/",
                             'thms': "Th228_Wfs_SS+MS_" + args.position + "_MC/",
                             'rams': "Ra226_Wfs_SS+MS_" + args.position + "_MC/",
                             'coms': "Co60_Wfs_SS+MS_" + args.position + "_MC/",
                             'unms': "Uniform_Wfs_SS+MS_" + args.position + "_MC/",
                             'gams': "Gamma_Wfs_SS+MS_" + args.position + "_MC/"}, {}
        files_training = pickle.load(open(args.folderOUT + "splitted_files.p", "rb"))
        for source in args.sources:
            try:
                #files[source] = files_training['val'][source] + files_training['test'][source]
                raise KeyError
            except KeyError:
                folderIN[source] = os.path.join(os.path.join(args.folderIN, ''), endings[source])
                files[source] = [os.path.join(folderIN[source], f) for f in os.listdir(folderIN[source]) if os.path.isfile(os.path.join(folderIN[source], f))]
                print 'Input  Folder: (', source, ')\t', folderIN[source]
        print 'Input  File:\t\t', (args.folderOUT + "splitted_files.p")
        return files
    if args.predict_data:
        endings, folderIN = {'thss': "Th228_Wfs_SS_" + args.position+"_Data/",
                             'rass': "Ra226_Wfs_SS_" + args.position+"_Data/",
                             'coss':  "Co60_Wfs_SS_" + args.position+"_Data/",
                             'thms': "Th228_Wfs_SS+MS_" + args.position + "_Data/",
                             'rams': "Ra226_Wfs_SS+MS_" + args.position + "_Data/",
                             'coms':  "Co60_Wfs_SS+MS_" + args.position + "_Data/",
                             'thms2000': "Th228_Wfs_SS+MS_" + args.position + "_Data_2000/",
                             'thms4500': "Th228_Wfs_SS+MS_" + args.position + "_Data_4500/"}, {}
        for source in args.sources:
            folderIN[source] = os.path.join(os.path.join(args.folderIN, ''), endings[source])
            files[source] = [os.path.join(folderIN[source], f) for f in os.listdir(folderIN[source]) if os.path.isfile(os.path.join(folderIN[source], f))]
            print 'Input  Folder: (', source, ')\t', folderIN[source]

        return files

def get_events(args, files, model, fOUT):
    try:
        if args.new: raise IOError
        spec = pickle.load(open(fOUT, "rb"))
        if args.events > len(spec['E_CNN']): raise IOError
    except IOError:
        if model == None: print 'model not found and not events file found' ; exit()
        E_CNN, E_EXO, E_EXOPur, E_True, E_Light, isSS = [], [], [], [], [], []
        posX, posY, posZ = [], [], []
        misClu, misEne, recEneInd, numCC = [], [], [], []
        files_list = np.concatenate(files.values()).tolist()
        events_per_batch = 2000
        if args.events % events_per_batch != 0: print 'choose event number in orders of 2000 events'; exit()
        iterations = plot.round_down(args.events, events_per_batch)/events_per_batch
        if args.predict_mc:     gen = generate_batch_reconstruction(generate_event_reconstruction(files_list), events_per_batch)
        if args.predict_data:   gen = generate_batch_data(generate_event_data(files_list), events_per_batch)
        for i in range(iterations):
            print args.source, i
            if args.predict_mc:
                E_CNN_temp, E_True_temp, E_EXO_temp, isSS_temp, posX_temp, posY_temp, posZ_temp, \
                misClu_temp, misEne_temp, recEneInd_temp, numCC_temp = predict_energy_reconstruction(model, gen)
                E_True.extend(E_True_temp)
                misClu.extend(misClu_temp)
                misEne.extend(misEne_temp)
                recEneInd.extend(recEneInd_temp)
                numCC.extend(numCC_temp)
            if args.predict_data:
                E_CNN_temp, E_EXO_temp, E_EXOPur_temp, E_Light_temp, isSS_temp, posX_temp, posY_temp, posZ_temp = predict_energy_data(model, gen)
                E_EXOPur.extend(E_EXOPur_temp)
                E_Light.extend(E_Light_temp)
            E_CNN.extend(E_CNN_temp)
            E_EXO.extend(E_EXO_temp)
            isSS.extend(isSS_temp)
            posX.extend(posX_temp)
            posY.extend(posY_temp)
            posZ.extend(posZ_temp)
        if args.predict_mc:
            spec = {'E_CNN': np.asarray(E_CNN), 'E_EXO': np.asarray(E_EXO), 'E_True': np.asarray(E_True),
                    'isSS': np.asarray(isSS), 'posX': np.asarray(posX), 'posY': np.asarray(posY), 'posZ': np.asarray(posZ),
                    'missedCluster': np.asarray(misClu), 'missedEnergy': np.asarray(misEne),
                    'reconEnergyInd': np.asarray(recEneInd), 'numCC': np.asarray(numCC)}
        if args.predict_data:
            spec = {'E_CNN': np.asarray(E_CNN), 'E_EXO': np.asarray(E_EXO), 'E_EXOPur': np.asarray(E_EXOPur), 'E_Light': np.asarray(E_Light),
                    'isSS': np.asarray(isSS), 'posX': np.asarray(posX), 'posY': np.asarray(posY), 'posZ': np.asarray(posZ)}
            spec['E_CNNPur'] = get_purity_corretion(data=spec)
        pickle.dump(spec, open(fOUT, "wb"))
        print 'posX', len(posX), min(posX), max(posX)
        print 'posY', len(posY), min(posY), max(posY)
        print 'posZ', len(posZ), min(posZ), max(posZ)
        if args.predict_mc:
            print 'misClu', len(misClu), min(misClu), max(misClu)
            print 'misEne', len(misEne), min(misEne), max(misEne)
            print 'recEneInd', len(recEneInd), min(recEneInd), max(recEneInd)
            print 'numCC', len(numCC), min(numCC), max(numCC)
    return spec

def get_purity_corretion(data):
    # Combine lifetime correction factor from EXO CorrectedEnergy and EXO PurityCorrectedEnergy
    # TODO check if this approach is valid for MS events
    return data['E_CNN']*data['E_EXOPur']/data['E_EXO']

def make_plots_data(folderOUT, dataIn, epoch, sources, position):
    fileOUT = epoch + '_' + sources + '_' + position + '_'
    name_CNN = 'DNN'
    name_EXO = 'EXO Recon'
    peakpos = 2614.5

    if 'th' in sources and position=='S5':
        CalibrateSelf = True
    else:
        CalibrateSelf = False
        dataIn_Ref = pickle.load(open(folderOUT + '../thms-S5/spectrum_events_' + epoch + '_thms-S5.p', "rb"))

    data = {}
    data_Ref = {}
    for E_List_str in ['E_CNN', 'E_EXO', 'E_CNNPur', 'E_EXOPur', 'E_Light']:
        data[E_List_str] = {'SS': dataIn[E_List_str][dataIn['isSS'] == True],
                            'MS': dataIn[E_List_str][dataIn['isSS'] == False],
                            'SSMS': dataIn[E_List_str]}
        if not CalibrateSelf:
            data_Ref[E_List_str] = {'SS': dataIn_Ref[E_List_str][dataIn_Ref['isSS'] == True],
                                    'MS': dataIn_Ref[E_List_str][dataIn_Ref['isSS'] == False],
                                    'SSMS': dataIn_Ref[E_List_str]}

        for Multi in ['SS', 'MS']:
            if CalibrateSelf:
                CalibrationFactor = plot.calibrate_spectrum(data=data[E_List_str][Multi], name=E_List_str, peakpos=peakpos, isMC=False,
                                                            fOUT=(folderOUT + fileOUT + 'calibration_' + E_List_str + '_' + Multi+ '.pdf'), peakfinder='max')
            else:
                CalibrationFactor = plot.calibrate_spectrum(data=data_Ref[E_List_str][Multi], name="", peakpos=peakpos, isMC=False, fOUT=None, peakfinder='max')
            data[E_List_str]['calib_' + Multi] = data[E_List_str][Multi] / CalibrationFactor
        data[E_List_str]['calib_SSMS'] = np.concatenate((data[E_List_str]['calib_SS'],data[E_List_str]['calib_MS']))

    for Multi in ['SS', 'MS', 'SSMS', 'calib_SS', 'calib_MS', 'calib_SSMS']:
        plot.plot_spectrum(data_CNN=data['E_CNN'][Multi], data_EXO=data['E_EXO'][Multi], fit=('th' in sources), isMC=False,
                           peakpos=peakpos, fOUT=(folderOUT + fileOUT + 'spectrum_' + Multi + '.pdf'))
        plot.plot_spectrum(data_CNN=data['E_CNNPur'][Multi], data_EXO=data['E_EXOPur'][Multi], fit=('th' in sources), isMC=False,
                           peakpos=peakpos, fOUT=(folderOUT + fileOUT + 'spectrumPurity_' + Multi + '.pdf'))
        plot.plot_scatter_hist2d(E_x=data['E_EXO'][Multi], E_y=data['E_CNN'][Multi],
                                 name_x=name_EXO, name_y=name_CNN, fOUT=(folderOUT + fileOUT + 'scatter_hist2d_' + Multi + '.pdf'))
        plot.plot_residual_hist2d(E_x=data['E_EXO'][Multi], E_y=data['E_CNN'][Multi],
                                  name_x=name_EXO, name_y=name_CNN, fOUT=(folderOUT + fileOUT + 'residual_hist2d_' + Multi + '.pdf'))
        plot.plot_anticorrelation_hist2d(E_x=data['E_EXOPur'][Multi], E_y=data['E_Light'][Multi],
                                         name_x='PurityCorrectedCharge', name_y='Light', name_title=name_EXO, fOUT=(folderOUT + fileOUT + 'anticorrelation_Standard_' + Multi + '.pdf'))
        plot.plot_anticorrelation_hist2d(E_x=data['E_CNNPur'][Multi], E_y=data['E_Light'][Multi],
                                         name_x='PurityCorrectedCharge', name_y='Light', name_title=name_CNN, fOUT=(folderOUT + fileOUT + 'anticorrelation_ConvNN_' + Multi + '.pdf'))

    return

def rotate_energy(folder, data, epoch, new, self):
    import RotationAngle_custom as rot
    import math
    file = folder + 'RotationAngle_' + epoch + '.p'
    if self:
        try:
            if new: raise IOError
            # raise IOError
            spec = pickle.load(open(file, "rb"))
        except IOError:
            E_Light = data['E_Light']
            spec = {}
            ThetaToTry_fine, ThetaToTry_med, ThetaToTry_raw = [], [], []
            ThetaToTry_raw  = [ round(0.00 + 0.05 * i,3) for i in range(30) ]
            ThetaToTry_fine = [ round(0.45 + 0.01 * i,3) for i in range(19) ]
            ThetaToTry_med  = [ round(0.40 + 0.02 * i,3) for i in range(18) ]
            ThetaToTry = sorted(set(ThetaToTry_fine + ThetaToTry_med + ThetaToTry_raw))
            for E_List_str in ['E_CNNPur', 'E_EXOPur']:
                E_List = data[E_List_str]
                E_List2D_ss = zip(E_List[data['isSS']==True] , E_Light[data['isSS']==True] )
                E_List2D_ms = zip(E_List[data['isSS']==False], E_Light[data['isSS']==False])
                spec[E_List_str] = rot.Run(EnergyList2D_ss=E_List2D_ss, EnergyList2D_ms=E_List2D_ms, ThetaToTry=ThetaToTry, prefix_local=folder+'Rotation_spectrum_'+E_List_str)
                spec[E_List_str]['E_Rot_ss'] = E_List[data['isSS']==True]  * math.cos(spec[E_List_str]['Theta_ss'][0]) \
                                               + E_Light[data['isSS']==True]  * math.sin(spec[E_List_str]['Theta_ss'][0])
                spec[E_List_str]['E_Rot_ms'] = E_List[data['isSS']==False] * math.cos(spec[E_List_str]['Theta_ms'][0]) \
                                               + E_Light[data['isSS']==False] * math.sin(spec[E_List_str]['Theta_ms'][0])
            pickle.dump(spec, open(file, "wb"))
    else:
        spec = pickle.load(open(file, "rb"))
        E_Light = data['E_Light']
        for E_List_str in ['E_CNNPur', 'E_EXOPur']:
            E_List = data[E_List_str]
            spec[E_List_str]['E_Rot_ss'] = E_List[data['isSS'] == True] * math.cos(spec[E_List_str]['Theta_ss'][0]) \
                                           + E_Light[data['isSS'] == True] * math.sin(spec[E_List_str]['Theta_ss'][0])
            spec[E_List_str]['E_Rot_ms'] = E_List[data['isSS'] == False] * math.cos(spec[E_List_str]['Theta_ms'][0]) \
                                           + E_Light[data['isSS'] == False] * math.sin(spec[E_List_str]['Theta_ms'][0])
    return spec

def generate_event_data(files):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            Y_recon_array = np.asarray(f.get('reconEnergy'))
            Y_reconPur_array = np.asarray(f.get('reconEnergyPurity'))
            Y_light_array = np.asarray(f.get('lightEnergy'))
            # X_init_array = np.asarray(f.get('wfs'))
            isSS_array = np.asarray(f.get('isSS'))
            posX_array = np.asarray(f.get('posX'))
            posY_array = np.asarray(f.get('posY'))
            posZ_array = np.asarray(f.get('posZ'))
            lst = range(len(Y_recon_array))
            random.shuffle(lst)
            for i in lst:
                isSS = isSS_array[i]
                Y_recon = Y_recon_array[i]
                Y_reconPur = Y_reconPur_array[i]
                Y_light = Y_light_array[i]
                xs_i = f['wfs'][i]
                posX = posX_array[i]
                posY = posY_array[i]
                posZ = posZ_array[i]
                xs_i = np.asarray(np.split(xs_i, 2, axis=1))
                yield (xs_i, Y_recon, Y_reconPur, Y_light, isSS, posX, posY, posZ)
            f.close()
        print 'all files used. Re-iterating files'

def generate_batch_data(generator, batchSize):
    while 1:
        X, Y, YPur, Z, SS, posX, posY, posZ = [], [], [], [], [], [], [], []
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            YPur.append(temp[2])
            Z.append(temp[3])
            SS.append(temp[4])
            posX.append(temp[5])
            posY.append(temp[6])
            posZ.append(temp[7])
        X = np.swapaxes(np.asarray(X), 0, 1)
        yield (list(X), np.asarray(Y), np.asarray(YPur), np.asarray(Z), np.asarray(SS), np.asarray(posX), np.asarray(posY), np.asarray(posZ))
        #yield (np.asarray(X), np.asarray(Y), np.asarray(Z), np.asarray(SS), np.asarray(posX), np.asarray(posY), np.asarray(posZ))

def predict_energy_data(model, generator):
    E_pred_wfs, E_recon, E_reconPur, E_light, isSS, posX, posY, posZ = generator.next()
    E_pred = np.asarray(model.predict(E_pred_wfs, 100)[:,0])
    return (E_pred, E_recon, E_reconPur, E_light, isSS, posX, posY, posZ)

def generate_event_reconstruction(files):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            X_True_i = np.asarray(f.get('trueEnergy'))
            X_EXO_i = np.asarray(f.get('reconEnergy'))
            # wfs_i = np.asarray(f.get('wfs'))
            try:    isSS_i = ~np.asarray(f.get('isSS'))  # inverted because of logic error in file production
            except: print 'check input data. Abort.' ; exit()
            posX_i = np.asarray(f.get('posX'))
            posY_i = np.asarray(f.get('posY'))
            posZ_i = np.asarray(f.get('posZ'))
            missedCluster_i = np.asarray(f.get('missedCluster'))
            missedEnergy_i = np.asarray(f.get('missedEnergy'))
            reconEnergyInd_i = np.asarray(f.get('reconEnergyInd'))
            numCC_i = np.asarray(f.get('numCC'))
            lst = range(len(X_True_i))
            random.shuffle(lst)
            for i in lst:
                xs_i = f['wfs'][i]
                isSS = isSS_i[i]
                X_True = X_True_i[i]
                X_EXO = X_EXO_i[i]
                posX = posX_i[i]
                posY = posY_i[i]
                posZ = posZ_i[i]
                missedCluster = missedCluster_i[i]
                missedEnergy = missedEnergy_i[i]
                reconEnergyInd = reconEnergyInd_i[i]
                numCC = numCC_i[i]
                xs_i = np.asarray(np.split(xs_i, 2, axis=1))
                yield (xs_i, X_True, X_EXO, isSS, posX, posY, posZ, missedCluster, missedEnergy, reconEnergyInd, numCC)
            f.close()
        print 'all files used. Re-iterating files'

def generate_batch_reconstruction(generator, batchSize):
    while 1:
        X, Y, Z, SS, posX, posY, posZ, mCl, mEn, rEInd, numCC = [], [], [], [], [], [], [], [], [], [], []
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            Z.append(temp[2])
            SS.append(temp[3])
            posX.append(temp[4])
            posY.append(temp[5])
            posZ.append(temp[6])
            mCl.append(temp[7])
            mEn.append(temp[8])
            rEInd.append(temp[9])
            numCC.append(temp[10])
        X = np.swapaxes(np.asarray(X), 0, 1)
        yield (list(X), np.asarray(Y), np.asarray(Z), np.asarray(SS),
               np.asarray(posX), np.asarray(posY), np.asarray(posZ), np.asarray(mCl),
               np.asarray(mEn), np.asarray(rEInd), np.asarray(numCC))
        #yield (np.asarray(X), np.asarray(Y), np.asarray(Z), np.asarray(SS),
         #      np.asarray(posX), np.asarray(posY), np.asarray(posZ), np.asarray(mCl),
           #    np.asarray(mEn), np.asarray(rEInd), np.asarray(numCC))

def predict_energy_reconstruction(model, generator):
    E_CNN_wfs, E_True, E_EXO, isSS, posX, posY, posZ, misClu, misEne, recEneInd, numCC = generator.next()
    E_CNN = np.asarray(model.predict(E_CNN_wfs, 100)[:,0])
    return (E_CNN, E_True, E_EXO, isSS, posX, posY, posZ, misClu, misEne, recEneInd, numCC)

def num_events(files):
    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += len(np.asarray(f.get('trueEnergy')))
        f.close()
    return counter

def get_energy_spectrum(args, files):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    entry = []
    for filename in files:
        f = h5py.File(str(filename), 'r')
        temp=np.array(f.get('trueEnergy'))
        for i in range(len(temp)):
            entry.append(temp[i])
        f.close()
    hist, bin_edges = np.histogram(entry, bins=125, range=(0,4000), density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.

    def lin(x,m,t):
        return x*m + t

    def funcExpTest(x,x0,A,B):
        return A/np.exp(-1.*x/x0) + B

    from scipy.optimize import curve_fit

    # coeff = [2.e-7,8.e-4]
    # for i in range(5):
    #     coeff, var_matrix = curve_fit(lin, bin_centres, hist, p0=coeff)
    #     coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))

    # coeff2 = [7.e3, 1.e-3, 8.e-4]
    # for i in range(5):
    #     coeff2, var_matrix = curve_fit(funcExpTest, bin_centres[:-5], hist[:-5], p0=coeff2)
    #     print coeff2
    plt.step(bin_centres, hist, where='mid')
    # plt.plot(bin_centres, lin(bin_centres, *coeff), lw =2)
    # plt.plot(bin_centres, funcExpTest(bin_centres, *coeff2), lw=2)


    # plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=200, xmax=3700)
    plt.ylim(ymin=0, ymax=7.e-4)
    plt.savefig(args.folderOUT + 'spectrum.pdf')
    plt.close()
    plt.clf()

    def funcExp(x,x0,A):
        return A*np.exp(-1.*x/x0)

    hist_inv=np.zeros(hist.shape)
    for i in range(len(hist)):
        try:
            hist_inv[i]=1.0/float(hist[i])
        except:
            pass
    hist_inv = hist_inv / hist_inv.sum(axis=0, keepdims=True)

    coeff1 = [-1.6e3, 4.e-3]
    for i in range(5):
        low = np.argmax(bin_centres>750)
        up = np.argmax(bin_centres > 2950)
        coeff1, var_matrix = curve_fit(funcExp, bin_centres[low:up], hist_inv[low:up], p0=coeff1)
        print coeff1

    plt.step(bin_centres, hist_inv, where='mid')
    plt.plot(bin_centres, funcExp(bin_centres, *coeff1), lw=2)
    # plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Inverse')
    plt.xlim(xmin=0, xmax=5000)
    plt.savefig(args.folderOUT + 'spectrum_inverse.pdf')
    plt.close()
    plt.clf()


    plt.step(bin_centres, hist*hist_inv, where='mid')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.xlim(xmin=500, xmax=3200)
    plt.savefig(args.folderOUT + 'spectrum_combined.pdf')
    plt.close()
    plt.clf()



    return (hist_inv, bin_edges[:-1])

def get_energy_spectrum_mixed(args, files, add):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    entry, hist, entry_mixed = {}, {}, []
    for source in args.sources:
        entry[source] = []
        for filename in files[source]:
            f = h5py.File(str(filename), 'r')
            temp = np.array(f.get('trueEnergy')).tolist()
            # temp = np.array(f.get('reconEnergy')).tolist()
            f.close()
            entry[source].extend(temp)
        entry_mixed.extend(entry[source])
    num_counts =  float(len(entry_mixed))
    hist_mixed, bin_edges = np.histogram(entry_mixed, bins=500, range=(0, 5000), density=False)
    bin_width = ((bin_edges[1] - bin_edges[0]) / 2.0)
    plt.plot(bin_edges[:-1] + bin_width, np.array(hist_mixed)/num_counts, label="combined", lw = 2, color='k')

    for source in args.sources:
        label = args.label[source]
        hist[source], bin_edges = np.histogram(entry[source], bins=500, range=(0,5000), density=False)
        plt.plot(bin_edges[:-1] + bin_width, np.array(hist[source])/num_counts, label=label)
        # print "%s\t%s\t%i" % (add, source , len(entry[source]))
    # plt.axvline(x=2614, lw=3, color='blue')
    plt.gca().set_yscale('log')
    plt.gcf().set_size_inches(10,5)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=500, xmax=3500)
    plt.ylim(ymin=(1.0/1000000), ymax=1.0)
    plt.savefig(args.folderOUT + 'spectrum_mixed_' + add + '.pdf')
    plt.close()
    plt.clf()
    return

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()