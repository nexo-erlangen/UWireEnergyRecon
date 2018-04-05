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

    files = {}
    #endings, folderIN = {'thms': "/home/vault/capm/sn0515/PhD/Th_U-Wire/Th228-FullWfs-S8/"}, {}
    endings, folderIN = {'thms': "/home/woody/capm/sn0515/PhD/Th_U-Wire/Th228_Wfs_SS+MS_S5_Data_FullWfs/"}, {}
    for source in endings.keys():
        folderIN[source] = endings[source]
        files[source] = [os.path.join(folderIN[source], f) for f in os.listdir(folderIN[source]) if os.path.isfile(os.path.join(folderIN[source], f))]
    print files

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
        except: print 'model not found' ; model = None #; continue

        args.source = "".join(sorted([k for k, v in files.items() if v]))
        print args
        print files
        print model
        get_events(args=args, files=files, model=model)

    print '===================================== Program finished =============================='

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def make_organize():
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-model' , dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy/', help='Model Path')
    parser.add_argument('-in'    , dest='folderIN' , default='/home/vault/capm/sn0515/PhD/Th_U-Wire/Data_MC/', help='MC/Data Path')
    parser.add_argument('-epochs', dest='epochs'   , default=['final'], nargs="*", help='Load weights from Epoch')
    parser.add_argument('-source', dest='sources'  , default=['thms'], nargs="*", choices=["thss", "coss", "rass", "gass", "unss", 'thms', 'unms', 'rams', 'coms'], help='sources for training (ra,co,th,ga,un)')
    parser.add_argument('-position', dest='position', default='S5', choices=['S2', 'S5', 'S8'], help='sources position')
    parser.add_argument('-multi' , dest='multi', default='SS', choices=['SS', 'MS'], help='event multiplicity')
    parser.add_argument('-events', dest='events'   , default=10000, type=int, help='number of events to reconstruct')

    parser.add_argument('--mc'    , dest='predict_mc'   , action='store_true', help='Predict MC Data')
    parser.add_argument('--data'  , dest='predict_data' , action='store_true', help='Predict REAL Data')
    parser.add_argument('--gamma' , dest='predict_gamma', action='store_true', help='Predict MC Gamma Data')
    parser.add_argument('--rotate', dest='rotate'       , action='store_true', help='Rotate Energy')
    parser.add_argument('--new'   , dest='new'          , action='store_true', help='Process new events')
    parser.add_argument('--range' , dest='epoch_range'  , action='store_true', help='range of epochs')
    args, unknown = parser.parse_known_args()

    args.label = {'thss': "Th228-SS", 'rass': "Ra226-SS", 'coss': "Co60-SS", 'gass': "Gamma-SS", 'unss': "Uniform-SS",
                  'thms': "Th228-MS", 'rams': "Ra226-MS", 'coms': "Co60-MS"}
    list_mc   = ['thss','rass','coss','gass','unss', 'gams', 'thms', 'unms']
    list_data = ['thss','rass','coss', 'thms','rams','coms']

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
        if args.multi == 'SS': args.sources = ['thss']
        if args.multi == 'MS': args.sources = ['thms']
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
    print 'Events:\t\t\t'      , args.events
    print 'Output Folder:\t\t' , args.folderOUT

    return args

def get_events(args, files, model):
    if model == None: print 'model not found and not events file found' ; exit()
    E_CNN, E_EXO, E_True, E_Light = [], [], [], []
    files_list = np.concatenate(files.values()).tolist()
    print files_list
    events_per_batch = 1000
    iterations = plot.round_down(args.events, events_per_batch)/events_per_batch
    if args.predict_mc:     gen = generate_batch_reconstruction(generate_event_reconstruction(files_list), events_per_batch)
    if args.predict_data:   gen = generate_batch_data(generate_event_data(files_list), events_per_batch)
    for i in range(iterations):
        print args.source, i
        if args.predict_mc:
            E_CNN_temp, E_True_temp, E_EXO_temp = predict_energy_reconstruction(model, gen, i)
            E_True.extend(E_True_temp)
        if args.predict_data:
            E_CNN_temp, E_EXO_temp, E_Light_temp = predict_energy_data(model, gen, i)
            E_Light.extend(E_Light_temp)
        E_CNN.extend(E_CNN_temp)
        E_EXO.extend(E_EXO_temp)
    return

def generate_event_reconstruction(files):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            print filename
            print 'open and read'
            f = h5py.File(str(filename), 'r')
            X_True_i = np.asarray(f.get('trueEnergy'))
            X_EXO_i = np.asarray(f.get('reconEnergy'))
            wfs_i = np.asarray(f.get('wfs'))
            gains = np.asarray(f.get('gains'))
            wfs_i = np.asarray(wfs_i / gains[:, None])
            f.close()
            print 'stop reading'
            lst = range(len(X_True_i))
            random.shuffle(lst)
            for i in lst:
                X_True = X_True_i[i]
                X_EXO = X_EXO_i[i]
                wfs = wfs_i[i]
                yield (wfs, X_True, X_EXO)

def generate_batch_reconstruction(generator, batchSize):
    while 1:
        X, Y, Z = [], [], []
        print 'start batch reading'
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            Z.append(temp[2])
        print 'yield batch'
        yield (np.asarray(X), np.asarray(Y), np.asarray(Z))

def predict_energy_reconstruction(model, generator, iter):
    print 'start predicting'
    E_CNN_wfs, E_True, E_recon = generator.next()
    E_CNN_wfs = np.swapaxes(E_CNN_wfs, 1, 2)
    E_CNN_wfs = E_CNN_wfs[..., np.newaxis]
    E_CNN_wfs_cnn = E_CNN_wfs[:, 512:1536, :, :]
    E_CNN = np.asarray(model.predict(E_CNN_wfs_cnn, 100)[:,0])
    counterDiff, counterELow, counterGood = 0, 0, 0
    # for i in xrange(len(E_recon)):
        # if E_True[i] < 2500:
        #     if abs(E_True[i] - E_CNN[i]) > 200. and counterELow < 20:
        #         print '\t Difference is good\t', counterDiff, '\tMC:', E_True[i], '\tCNN:', E_CNN[i]
        #         plot_waveforms(E_CNN_wfs[i], 'bad_'+str(iter)+'_'+str(counterDiff), E_True[i], E_CNN[i])
        #         counterDiff += 1
        # if abs(E_True[i]  - E_CNN[i]) < 1. and counterGood < 10:
        #     print '\t Reconstruction is good\t', counterGood, '\tMC:', E_True[i], '\tCNN:', E_CNN[i]
        #     plot_waveforms(E_CNN_wfs[i], 'good_'+str(iter)+'_'+str(counterGood), E_True[i], E_CNN[i])
        #     counterGood += 1
        # if E_recon[i] > 1500 and counterDiff < 10:
        #     if E_recon[i] - E_CNN[i] > 400.:
        #         print '\t Big Difference\t', counterDiff, '\tMC:', E_True[i], '\tCNN:', E_CNN[i], '\tDiff:', E_recon[i] - E_CNN[i]
        #         plot_waveforms(E_CNN_wfs[i], 'Diff_'+str(iter)+'_'+str(counterDiff), E_recon[i], E_CNN[i])
        #         counterDiff += 1
    return (E_CNN, E_True, E_recon)

def generate_event_data(files):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            print filename
            print 'open and read'
            f = h5py.File(str(filename), 'r')
            Y_recon_array = np.asarray(f.get('reconEnergy'))
            Y_light_array = np.asarray(f.get('lightEnergy'))
            isSS_array = ~np.asarray(f.get('isSS'))
            X_init_array = np.asarray(f.get('wfs'))
            gains = np.asarray(f.get('gains'))
            X_init_array = np.asarray(X_init_array / gains[:, None])
            f.close()
            print X_init_array.shape
            print 'stop reading'

            lst = range(len(Y_recon_array))
            random.shuffle(lst)
            for i in lst:
                # if not GoodEvent(wfs=X_init_array[i],ch=75,idx=i) or GoodEvent(wfs=X_init_array[i],ch=74,idx=i-1): continue
                # if not GoodEvent(wfs=X_init_array[i], ch=38, idx=i): continue #0, 37, 38, 75
                isSS = isSS_array[i]
                Y_recon = Y_recon_array[i]
                Y_light = Y_light_array[i]
                X_norm = X_init_array[i]
                yield (X_norm, Y_recon, Y_light, isSS)

def GoodEvent(wfs,ch,idx):
    # print wfs.shape, ch, wfs[ch].shape
    wfs = np.asarray(wfs[ch])
    wfs_base = wfs[abs(wfs) < abs(wfs.mean() + 1. * wfs.std())]
    thresh = 12. * wfs_base.std() + wfs_base.mean()
    # import matplotlib.pyplot as plt

    if any(wfs[wfs>thresh]):
        print 'true'
        return True
    return False

def generate_batch_data(generator, batchSize):
    while 1:
        X, Y, Z, SS = [], [], [], []
        print 'start batch reading'
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            Z.append(temp[2])
            SS.append(temp[3])
        print 'yield batch'
        yield (np.asarray(X), np.asarray(Y), np.asarray(Z), np.asarray(SS))

def predict_energy_data(model, generator, iter):
    E_pred_wfs, E_recon, E_light, isSS = generator.next()
    E_pred_wfs = np.swapaxes(E_pred_wfs, 1, 2)
    E_pred_wfs = E_pred_wfs[..., np.newaxis]
    E_pred_wfs_cnn = E_pred_wfs[:,512:1536,:,:]
    print 'start predicting'
    E_pred = np.asarray(model.predict(E_pred_wfs_cnn, 100)[:,0])
    print 'end predicting'

    counterDiff, counterELow, counterGood = 0, 0, 0
    counterSS, counterMS = 0, 0
    for i in xrange(len(E_recon)):
        # print i, E_recon[i], E_pred[i], E_recon[i]-E_pred[i], isSS[i]
        # if E_recon[i] > 700 and E_recon[i] < 3000:
        #     if E_recon[i] - E_pred[i] > 400. and counterDiff < 20:
        #         print '\t Difference is good\t', counterDiff, '\tEXO:', E_recon[i], '\tCNN:', E_pred[i]
        #         plot_waveforms(E_pred_wfs[i], 'bad_'+str(iter)+'_'+str(counterDiff), E_recon[i], E_pred[i])
        #         counterDiff += 1
        # if abs(E_recon[i]  - E_pred[i]) < 5. and counterGood < 5:
        #     if E_recon[i] > 2500 and E_recon[i] < 2700:
        #         print '\t Reconstruction is good\t', counterGood, '\tEXO:', E_recon[i], '\tCNN:', E_pred[i]
        #         plot_waveforms(E_pred_wfs[i], 'good_'+str(iter)+'_'+str(counterGood), E_recon[i], E_pred[i])
        #         counterGood += 1
        if E_recon[i] > 1500 and counterDiff < 10:
            if E_recon[i] - E_pred[i] > 400.:
                print '\t Big Difference\t', counterDiff, '\tEXO:', E_recon[i], '\tCNN:', E_pred[i], '\tDiff:', E_recon[i] - E_pred[i]
                plot_waveforms(E_pred_wfs[i], 'Diff_'+str(iter)+'_'+str(counterDiff), E_recon[i], E_pred[i])
                counterDiff += 1
        # if isSS[i] == True and counterSS < 10:
        #     if E_recon[i] > 1500. and E_pred[i] > 1500.:
        #         print '\t', i, '\t SS event\t', counterSS, '\tEXO:', E_recon[i], '\tCNN:', E_pred[i]
        #         plot_waveforms(E_pred_wfs[i], 'SS_'+str(iter)+'_'+str(counterSS), E_recon[i], E_pred[i])
        #         counterSS += 1
        # if isSS[i] == False and counterMS < 40:
        #     if E_recon[i] > 1500. and E_pred[i] > 1500.:
        #         print '\t', i, '\t MS event\t', counterMS, '\tEXO:', E_recon[i], '\tCNN:', E_pred[i]
        #         plot_waveforms(E_pred_wfs[i], 'MS_'+str(iter)+'_'+str(counterSS), E_recon[i], E_pred[i])
        #         counterMS += 1
    return (E_pred, E_recon, E_light)

def plot_waveforms(wf, idx, E_EXO, E_CNN):
    import matplotlib.pyplot as plt
    cut_1 = 512
    length = 1024
    cut_2 = cut_1 + length
    time = range(0, 2048)
    # plt.ion()
    for j in range(76):
        offset = 20. * j + (j//38)*20
        plt.plot(time[ : cut_1], wf[ : cut_1 , j] + offset, color='gray')
        plt.plot(time[ cut_1 : cut_2], wf[ cut_1 : cut_2, j] + offset, color='k')
        plt.plot(time[ cut_2 : ], wf[ cut_2 :, j] + offset, color='gray')
    plt.axvline(x=cut_1, lw=2, color='black')
    plt.axvline(x=cut_2, lw=2, color='black')
    plt.xlabel('Time [$\mu$s]')
    plt.ylabel('U-Wire')
    plt.title('EXO: %s       CNN: %s' % (E_EXO, E_CNN) )
    plt.xlim(xmin=0, xmax=2048)
    plt.ylim(ymin=-180, ymax=1700)
    # plt.ylim(ymin=-30, ymax=1550)
    plt.axes().set_aspect(0.7)
    plt.yticks([])
    # plt.draw()
    # raw_input('')
    plt.savefig('/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy/wavesData/' + str(idx) + '.png')
    plt.close()
    plt.clf()

def num_events(files):
    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += len(np.asarray(f.get('trueEnergy')))
        f.close()
    return counter

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()