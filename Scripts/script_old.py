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
    args, files = make_organize()
    if not args.predict_data and not args.reconstruct_gamma:
        splitted_files = split_data(args, files, frac_val={'th': 0.5, 'ra': 0.0, 'co': 0.0, 'un': 0.0},frac_test={'th': 0.0, 'ra': 0.0, 'co': 0.0, 'un': 0.0})
        # get_energy_spectrum_mixed(args, splitted_files['train'], add='train')
        # get_energy_spectrum_mixed(args, splitted_files['val'], add='val')
        # get_energy_spectrum_mixed(args, files, add='all')

        if not args.plotsonly and not args.reconstruct:
            model = train_model(args, splitted_files, get_model(args), args.nb_batch * args.nb_GPU)

    try:
        model
    except NameError:
        from keras.models import load_model
        try:
            model = load_model(args.folderOUT + "models/model-initial.hdf5")
            model.load_weights(args.folderOUT + "models/weights-" + args.nb_weights + ".hdf5")
        except:
            model = load_model(args.folderOUT + "models/model-" + args.nb_weights + ".hdf5")

    if not args.predict_data:
        if args.reconstruct_gamma:
            files_gamma = split_data(args, files, frac_val={'ga': 1.0, 'th': 0.0, 'ra': 0.0, 'co': 0.0}, frac_test={'th': 0.0, 'ra': 0.0, 'co': 0.0, 'ga': 0.0})
            print len(files_gamma['val']['ga'])
            reconstruct_MC(args=args, splitted_files=files_gamma, model=model)
            exit()
        if args.reconstruct:
            reconstruct_MC(args=args, splitted_files=splitted_files, model=model)
        else:
            E_pred, E_true = predict_energy(model, generate_batch(generate_event(np.concatenate(splitted_files['val'].values()).tolist()), 50000))
            make_plots(args=args, E_pred=E_pred, E_true=E_true, E_recon=[], epoch="9999")
        final_plots(args, pickle.load(open(args.folderOUT + "save.p", "rb")))
    else:
        # weight = int(np.genfromtxt(args.folderOUT + 'history.csv', delimiter=',', names=True)['epoch'][-1]+1)
        # weight = int(args.nb_weights)
        # for i in range(weight-4,weight+1):
        #     args.nb_weights = str(i).zfill(3)
        #     try: model.load_weights(args.folderOUT + "models/weights-" + args.nb_weights + ".hdf5")
        #     except: continue
        #     reconstruct_data(args=args, files=files, model=model)
        model.load_weights(args.folderOUT + "models/weights-" + args.nb_weights + ".hdf5")
        reconstruct_data(args=args, files=files, model=model)

    print '===================================== Program finished =============================='

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def make_organize():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-out', dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderOUT Path')
    parser.add_argument('-in', dest='folderIN', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/Data_MC', help='folderIN Path')
    parser.add_argument('-model', dest='folderMODEL', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderMODEL Path')
    parser.add_argument('-gpu', type=int, dest='nb_GPU', default=1, choices=[1, 2, 3, 4], help='nb of GPU')
    parser.add_argument('-epoch', type=int, dest='nb_epoch', default=1, help='nb Epochs')
    parser.add_argument('-batch', type=int, dest='nb_batch', default=16, help='Batch Size')
    parser.add_argument('-weights', dest='nb_weights', default='final', help='Load weights from Epoch')
    parser.add_argument('-source', dest='sources', default=['th', 'ra', 'co'], nargs="*", choices=["th", "co", "ra", "ga", "un"], help='sources for training (ra,co,th,ga,un)')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume Training')
    parser.add_argument('--plot', dest='plotsonly', action='store_true', help='Only produce plots')
    parser.add_argument('--test', dest='test', action='store_true', help='Only reduced data')
    parser.add_argument('--reconstruct', dest='reconstruct', action='store_true', help='Compare to EXO200 Reconstruction')
    parser.add_argument('--predict', dest='predict_data', action='store_true', help='Predict REAL Data')
    parser.add_argument('--gamma', dest='reconstruct_gamma', action='store_true', help='Reconstruct MC Gamma Data')
    args, unknown = parser.parse_known_args()

    folderIN, files, endings = {}, {}, {}
    if not args.predict_data:
        endings['th'] = "Th228_Wfs_SS_S5_MC/"
        endings['ra'] = "Ra226_Wfs_SS_S5_MC/"
        endings['co'] = "Co60_Wfs_SS_S5_MC/"
        endings['ga'] = "Gamma_WFs_S5_MC/"
        endings['un'] = "Uniform_WFs_S5_MC/"
    else:
        endings['th'] = "Th228_Wfs_SS_S5_Data/"
        endings['ra'] = "Ra226_Wfs_SS_S5_Data/"
        endings['co'] = "Co60_Wfs_SS_S5_Data/"
    for source in args.sources:
        folderIN[source] = os.path.join(args.folderIN, endings[source])
        files[source] = [os.path.join(folderIN[source], f) for f in os.listdir(folderIN[source]) if os.path.isfile(os.path.join(folderIN[source], f))]
        print 'Input  Folder: (', source, ')\t', folderIN[source]
    args.folderOUT = os.path.join(args.folderOUT,'')
    args.folderMODEL = os.path.join(args.folderMODEL,'')
    args.folderIN = folderIN

    if not os.path.exists(args.folderOUT+'models'):
        os.makedirs(args.folderOUT+'models')

    print 'Output Folder:\t\t'  , args.folderOUT
    if args.resume:
        print 'Model Folder:\t\t', args.folderMODEL
    print 'Number of GPU:\t\t', args.nb_GPU
    print 'Number of Epoch:\t', args.nb_epoch
    print 'BatchSize:\t\t', args.nb_batch,"\n"
    return args, files

def reconstruct_data(args, files, model):
    print 'predict Data events'
    sources = "".join(sorted([k for k, v in files.items() if v]))
    print sources
    os.system("mkdir -p %s " % (args.folderOUT + "0physics-data/" + args.nb_weights + "/"))
    try:
        spec = pickle.load(open(args.folderOUT + "0physics-data/" + args.nb_weights + "/spectrum_events_" + args.nb_weights + "_" + sources + ".p", "rb"))
    except IOError:
        E_DCNN, E_EXO, E_Light = [], [], []
        gen = generate_batch_data(generate_event_data(np.concatenate(files.values()).tolist()), 5000)
        for i in range(15):
            print args.nb_weights, i
            E_DCNN_temp, E_EXO_temp, E_Light_temp = predict_energy_data(model, gen)
            E_DCNN.extend(E_DCNN_temp)
            E_EXO.extend(E_EXO_temp)
            E_Light.extend(E_Light_temp)
        spec = {'E_DCNN': np.array(E_DCNN), 'E_EXO': np.array(E_EXO), 'E_Light': np.array(E_Light)}
        pickle.dump(spec, open(args.folderOUT + "0physics-data/" + args.nb_weights + "/spectrum_events_" + args.nb_weights + "_" + sources + ".p", "wb"))
    make_plots_data(args=args, E_DCNN=spec['E_DCNN'], E_EXO=spec['E_EXO'], E_Light=spec['E_Light'], sources=sources)
    return

def reconstruct_MC(args, splitted_files, model):
    print 'predict MC events'
    sources = "".join(sorted([k for k, v in splitted_files['val'].items() if v]))
    try:
        spec = pickle.load(open(args.folderOUT + "spectrum_events_" + args.nb_weights + "_" + sources + ".p", "rb"))
    except IOError:
        # splitted_files = split_data(args, files, frac_val={'th': 0.5, 'ra': 0.5, 'co': 0.4},frac_test={'th': 0.0, 'ra': 0.0, 'co': 0.0})
        # E_pred_th, E_true_th, E_recon_th = predict_energy_reconstruction(model, generate_batch_reconstruction(generate_event_reconstruction(splitted_files['val']['th']), 50000))
        # E_pred_ra, E_true_ra, E_recon_ra = predict_energy_reconstruction(model, generate_batch_reconstruction(generate_event_reconstruction(splitted_files['val']['ra']), 40000))
        # E_pred_co, E_true_co, E_recon_co = predict_energy_reconstruction(model, generate_batch_reconstruction(generate_event_reconstruction(splitted_files['val']['co']), 15000))
        E_pred, E_true, E_recon = [], [], []
        gen = generate_batch_reconstruction(generate_event_reconstruction(np.concatenate(splitted_files['val'].values()).tolist()), 5000)
        for i in range(8):
            print sources, i
            E_pred_temp, E_true_temp, E_recon_temp = predict_energy_reconstruction(model, gen)
            E_pred.extend(E_pred_temp)
            E_true.extend(E_true_temp)
            E_recon.extend(E_recon_temp)
        spec = {'E_pred': E_pred, 'E_true': E_true, 'E_recon': E_recon}
        pickle.dump(spec, open(args.folderOUT + "spectrum_events_" + args.nb_weights + "_" + sources + ".p", "wb"))
    make_plots(args=args, E_pred=np.array(spec['E_pred']), E_true=np.array(spec['E_true']),
               E_recon=np.array(spec['E_recon']), epoch=args.nb_weights)
    # exit()
    # make_plots(args=args, E_pred=np.array(spec['E_pred']), E_true=np.array(spec['E_true']), E_recon=np.array(spec['E_recon']),epoch=args.nb_weights)
    return

def split_data(args, files, frac_val, frac_test):
    if args.resume:
        os.system("cp %s %s" % (args.folderMODEL + "splitted_files.p", args.folderOUT + "splitted_files.p"))
        print 'load splitted files from %s' % (args.folderMODEL + "splitted_files.p")
        return pickle.load(open(args.folderOUT + "splitted_files.p", "rb"))
    else:
        splitted_files= {'train': {}, 'val': {}, 'test': {}}
        print "Source\tTotal\tTrain\tValid\tTest"
        for source in args.sources:
            num_val   = int(round(len(files[source]) * frac_val[source]))
            num_test  = int(round(len(files[source]) * frac_test[source]))
            random.shuffle(files[source])
            if not args.test:
                splitted_files['val'][source] = files[source][0:num_val]
                splitted_files['test'][source] = files[source][num_val:num_val + num_test]
                splitted_files['train'][source] = files[source][num_val + num_test : ]
            else:
                splitted_files['val'][source] = files[source][0:1]
                splitted_files['test'][source] = files[source][1:2]
                splitted_files['train'][source] = files[source][2:3]
            print "%s\t%i\t%i\t%i\t%i" % (source, len(files[source]), len(splitted_files['train'][source]), len(splitted_files['val'][source]), len(splitted_files['test'][source]))
        pickle.dump(splitted_files, open(args.folderOUT + "splitted_files.p", "wb"))
        return splitted_files

def num_events(files):
    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += len(np.array(f.get('trueEnergy')))
        f.close()
    return counter

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

def get_weight(Y, hist, bin_edges):
    return hist[np.digitize(Y, bin_edges) - 1]

def round_down(num, divisor):
    return num - (num%divisor)

def get_energy_spectrum(args, files):
    entry = []
    for filename in files:
        f = h5py.File(str(filename), 'r')
        temp=np.array(f.get('trueEnergy'))
        for i in range(len(temp)):
            entry.append(temp[i])
        f.close()
    hist, bin_edges = np.histogram(entry, bins=210, range=(700,3000), density=True)
    plt.plot(bin_edges[:-1], hist)
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=700, xmax=3000)
    plt.savefig(args.folderOUT + 'spectrum.pdf')
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
    plt.xlim(xmin=700, xmax=3000)
    plt.savefig(args.folderOUT + 'spectrum_inverse.pdf')
    plt.close()
    plt.clf()
    return (hist_inv, bin_edges[:-1])

def get_energy_spectrum_mixed(args, files, add=""):
    entry, hist, entry_mixed = {}, {}, []
    for source in args.sources:
        entry[source] = []
        for filename in files[source]:
            f = h5py.File(str(filename), 'r')
            temp = np.array(f.get('trueEnergy'))
            for i in range(len(temp)):
                entry[source].append(temp[i])
                entry_mixed.append(temp[i])
            f.close()
    num_counts =  float(len(entry_mixed))
    hist_mixed, bin_edges = np.histogram(entry_mixed, bins=280, range=(700, 3500), density=False)
    bin_width = ((bin_edges[1] - bin_edges[0]) / 2.0)
    plt.plot(bin_edges[:-1] + bin_width, np.array(hist_mixed)/num_counts, label="combined", lw = 2, color='k')

    for source in args.sources:
        if source is 'th': label = 'Th228'
        if source is 'ra': label = 'Ra226'
        if source is 'co': label = 'Co60'
        # if source is 'ga': label = 'Gamma'
        hist[source], bin_edges = np.histogram(entry[source], bins=280, range=(700,3500), density=False)
        plt.plot(bin_edges[:-1] + bin_width, np.array(hist[source])/num_counts, label=label)
        # print "%s\t%s\t%i" % (add, source , len(entry[source]))
    # plt.axvline(x=2614, lw=3, color='blue')
    plt.gca().set_yscale('log')
    plt.gcf().set_size_inches(10,5)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=700, xmax=3000)
    plt.ylim(ymin=(1.0/1000000), ymax=1.0)
    plt.savefig(args.folderOUT + 'spectrum_mixed_' + add + '.pdf')
    plt.close()
    plt.clf()
    return

def generate_event(files):
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            Y_true_array = np.array(f.get('trueEnergy'))
            X_init_array = np.array(f.get('wfs'))
            lst = range(len(Y_true_array))
            random.shuffle(lst)
            for i in lst:
                Y_true = Y_true_array[i]
                X_norm = X_init_array[i]
                yield (X_norm, Y_true)
            f.close()

def generate_batch(generator, batchSize):
    while 1:
        X, Y = [], []
        for i in range(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
        yield (np.array(X), np.array(Y))

def generate_batch_mixed(generator, batchSize, numEvents):
    order = np.array( ['ra']*numEvents['ra'] + ['co']*numEvents['co'] + ['th']*numEvents['th'] )
    X, Y = [], []
    while 1:
        random.shuffle(order)
        for source in order:
            temp = generator[source].next()
            X.append(temp[0])
            Y.append(temp[1])
            if len(Y)==batchSize:
                yield (np.array(X), np.array(Y))
                X, Y = [], []

def predict_energy(model, generator):
    E_pred_wfs, E_true = generator.next()
    E_pred = np.array(model.predict(E_pred_wfs, 100)[:,0])
    return (E_pred, E_true)

def generate_event_reconstruction(files):
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            Y_true_array = np.array(f.get('trueEnergy'))
            Y_recon_array = np.array(f.get('reconEnergy'))
            X_init_array = np.array(f.get('wfs'))
            lst = range(len(Y_true_array))
            random.shuffle(lst)
            for i in lst:
                Y_true = Y_true_array[i]
                Y_recon = Y_recon_array[i]
                X_norm = X_init_array[i]
                yield (X_norm, Y_true, Y_recon)
            f.close()

def generate_batch_reconstruction(generator, batchSize):
    while 1:
        X, Y, Z = [], [], []
        for i in range(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            Z.append(temp[2])
        yield (np.array(X), np.array(Y), np.array(Z))

def predict_energy_reconstruction(model, generator):
    E_pred_wfs, E_true, E_recon = generator.next()
    E_pred = np.array(model.predict(E_pred_wfs, 100)[:,0])
    return (E_pred, E_true, E_recon)

def generate_event_data(files):
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            Y_recon_array = np.array(f.get('reconEnergy'))
            Y_light_array = np.array(f.get('lightEnergy'))
            X_init_array = np.array(f.get('wfs'))
            lst = range(len(Y_recon_array))
            random.shuffle(lst)
            for i in lst:
                Y_recon = Y_recon_array[i]
                Y_light = Y_light_array[i]
                X_norm = X_init_array[i]
                yield (X_norm, Y_recon, Y_light)
            f.close()

def generate_batch_data(generator, batchSize):
    while 1:
        X, Y, Z = [], [], []
        for i in range(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            Z.append(temp[2])
        yield (np.array(X), np.array(Y), np.array(Z))

def predict_energy_data(model, generator):
    E_pred_wfs, E_recon, E_light = generator.next()
    E_pred = np.array(model.predict(E_pred_wfs, 100)[:,0])
    return (E_pred, E_recon, E_light)

# ----------------------------------------------------------
# Parallel GPU Computing
# ----------------------------------------------------------
def make_parallel(model, gpu_count):
    from keras.layers import merge
    from keras.layers.core import Lambda
    from keras.models import Model
    import tensorflow as tf

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [shape[:1] // parts, shape[1:]])
        stride = tf.concat(0, [shape[:1] // parts, shape[1:] * 0])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
        return Model(input=model.inputs, output=merged)

# ----------------------------------------------------------
# Callback Redefinition
# ----------------------------------------------------------
from keras import callbacks

class Histories(callbacks.Callback):
    def __init__(self, args, files):
        self.validation_data = None
        self.args = args
        self.files = files

    def on_train_begin(self, logs={}):
        self.losses = []
        if self.args.resume:
            os.system("cp %s %s" % (self.args.folderMODEL + "save.p", self.args.folderOUT + "save.p"))
        else:
            pickle.dump({}, open(self.args.folderOUT + "save.p", "wb"))
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        E_pred, E_true = predict_energy(self.model, generate_batch(generate_event(np.concatenate(self.files['val'].values()).tolist()), 10000))
        obs = make_plots(self.args, E_pred=E_pred, E_true=E_true, E_recon=[], epoch=str(epoch))
        self.dict_out = pickle.load(open(self.args.folderOUT + "save.p", "rb"))
        self.dict_out[str(epoch)] = {'E_pred': E_pred, 'E_true': E_true,
                                     'peak_pos': obs['peak_pos'],
                                     'peak_sig': obs['peak_sig'],
                                     'resid_pos': obs['resid_pos'],
                                     'resid_sig': obs['resid_sig'],
                                     'loss': logs['loss'], 'mean_absolute_error': logs['mean_absolute_error'],
                                     'val_loss': logs['val_loss'], 'val_mean_absolute_error': logs['val_mean_absolute_error']}
        pickle.dump(self.dict_out, open(self.args.folderOUT + "save.p", "wb"))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

# ----------------------------------------------------------
# Define model
# ----------------------------------------------------------
def get_model(args):
    def def_model():
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
        from keras.regularizers import l2, l1, l1l2

        init = "glorot_uniform"
        activation = "relu"
        padding = "same"
        regul = l2(1.e-2)
        model = Sequential()
        # convolution part
        model.add(Convolution2D(16, 5, 3, border_mode=padding, init=init, W_regularizer=regul, input_shape=(1024, 76, 1)))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), border_mode=padding))

        model.add(Convolution2D(32, 5, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), border_mode=padding))

        model.add(Convolution2D(64, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), border_mode=padding))

        # regression part
        model.add(Flatten())
        model.add(Dense(64, activation=activation, init=init, W_regularizer=regul))
        model.add(Dense(16, activation=activation, init=init, W_regularizer=regul))
        model.add(Dense(4,  activation=activation, init=init, W_regularizer=regul))
        model.add(Dense(1,  activation=activation, init=init))
        return model

    if not args.resume:
        from keras import optimizers
        print "===================================== new Model =====================================\n"
        model = def_model()
        epoch_start = 0
        optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error'])
    else:
        from keras.models import load_model
        print "===================================== load Model ==============================================="
        try:
            print "%smodels/(model/weights)-%s.hdf5" % (args.folderMODEL, args.nb_weights)
            print "================================================================================================\n"
            try:
                model = load_model(args.folderMODEL + "models/model-initial.hdf5")
                model.load_weights(args.folderMODEL + "models/weights-" + args.nb_weights + ".hdf5")
            except:
                model = load_model(args.folderMODEL + "models/model-" + args.nb_weights + ".hdf5")
            os.system("cp %s %s" % (args.folderMODEL + "history.csv", args.folderOUT + "history.csv"))
            if args.nb_weights=='final':
                epoch_start = 1+int(np.genfromtxt(args.folderOUT+'history.csv', delimiter=',', names=True)['epoch'][-1])
                print epoch_start
            else:
                epoch_start = 1+int(args.nb_weights)
        except:
            print "\t\tMODEL NOT FOUND!\n"
            exit()
    print "\nfirst epoch:\t", epoch_start
    print model.summary(), "\n"
    return model, epoch_start

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------
def train_model(args, files, (model, epoch_start), batchSize):
    from keras import callbacks
    start = time.clock()

    gen_train, gen_val, numEvents_train, numEvents_val = {}, {}, {}, {}
    for source in args.sources:
        gen_train[source] = generate_event(files['train'][source])
        gen_val[source] = generate_event(files['val'][source])
        numEvents_train[source] = num_events(files['train'][source])
        numEvents_val[source] = num_events(files['val'][source])

    if args.nb_GPU>1:
        model = make_parallel(model, args.nb_GPU)

    model.save(args.folderOUT + "models/model-initial.hdf5")
    model.save_weights(args.folderOUT + "models/weights-initial.hdf5")

    model.fit_generator(
        generate_batch_mixed(gen_train, batchSize, numEvents_train),
        samples_per_epoch=round_down(numEvents_train['th']+numEvents_train['ra']+numEvents_train['co'], batchSize),
        nb_epoch=args.nb_epoch+epoch_start,
        verbose=1,
        validation_data=generate_batch_mixed(gen_val, batchSize, numEvents_val),
        nb_val_samples=round_down(numEvents_val['th']+numEvents_val['ra']+numEvents_val['co'], batchSize),
        initial_epoch=epoch_start,
        callbacks=[
            callbacks.CSVLogger(args.folderOUT + 'history.csv', append=args.resume),
            callbacks.ModelCheckpoint(args.folderOUT + 'models/weights-{epoch:03d}.hdf5', save_weights_only=True, period=int(args.nb_epoch/100)),
            Histories(args, files)
        ])

    model.save(args.folderOUT+"models/model-final.hdf5")
    model.save_weights(args.folderOUT+"models/weights-final.hdf5")

    end = time.clock()
    print "\nElapsed time:\t%.2f minutes\tor rather\t%.2f hours\n" % (((end-start)/60.),((end-start)/60./60.))

    print 'Model performance\tloss\t\tmean_abs_err'
    print '\tTrain:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(np.concatenate(files['train'].values()).tolist()), batchSize), val_samples=128))
    print '\tValid:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(np.concatenate(files['val'].values()).tolist())  , batchSize), val_samples=128))
    return model

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def make_plots(args, E_pred, E_true, E_recon=[], epoch=""):
    # training curves
    def plot_losses(history):
        fig, ax = plt.subplots(1)
        ax.plot(history['epoch'], history['loss'],     label='training')
        ax.plot(history['epoch'], history['val_loss'], label='validation')
        ax.set(xlabel='epoch', ylabel='loss')
        ax.grid(True)
        ax.semilogy()
        plt.legend(loc="best")
        fig.savefig(args.folderOUT+'loss-test.pdf')
        plt.clf()
        plt.close()

        fig, ax = plt.subplots(1)
        ax.plot(history['epoch'], history['mean_absolute_error'],     label='training')
        ax.plot(history['epoch'], history['val_mean_absolute_error'], label='validation')
        ax.legend()
        ax.grid(True)
        plt.legend(loc="best")
        ax.set(xlabel='epoch', ylabel='mean absolute error')
        fig.savefig(args.folderOUT+'mean_absolute_error-test.pdf')
        plt.clf()
        plt.close()
        return

    # spectrum of predicted/true energy
    def plot_spectrum(E_pred, E_true, E_recon, epoch):
        def plot_fit_spectrum(data, fit, color="", name=""):
            hist, bin_edges = np.histogram(data, bins=280, range=(700, 3500), density=False)
            norm_factor = float(len(data))
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            peak = np.argmax(hist[np.digitize(2400, bin_centres):np.digitize(2800, bin_centres)]) + np.digitize(2400,
                                                                                                                bin_centres)
            coeff = [hist[peak], bin_centres[peak], 50., -0.005, 0.0]
            if fit:
                from scipy.optimize import curve_fit
                for i in range(5):
                    try:
                        low = np.digitize(coeff[1] - (5 * abs(coeff[2])), bin_centres)
                        up = np.digitize(coeff[1] + (5 * abs(coeff[2])), bin_centres)
                        coeff, var_matrix = curve_fit(gaussErf, bin_centres[low:up], hist[low:up], p0=coeff)
                        coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
                    except:
                        print epoch, '\tfit did not work\t', i
                        coeff, coeff_err = [hist[peak], bin_centres[peak], 50.0*(i+1), -0.005, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]
                delE = abs(coeff[2]) / coeff[1] * 100.0
                delE_err = delE * np.sqrt((coeff_err[1] / coeff[1]) ** 2 + (coeff_err[2] / coeff[2]) ** 2)
                if args.reconstruct_gamma:
                    plt.step(bin_centres, hist / norm_factor, where='mid', color=color, lw=2, label='%s' % (name))
                    # plt.plot(bin_centres[low:up], gaussErf(bin_centres[low:up], *coeff) / norm_factor, lw=2 , color=color)
                else:
                    plt.step(bin_centres, hist / norm_factor, where='mid', color=color,
                             label='%s\t$\mu=%.1f \pm %.1f$' % (name, coeff[1], coeff_err[1]))
                    plt.plot(bin_centres[low:up], gaussErf(bin_centres[low:up], *coeff) / norm_factor, lw=2 , color=color,
                         label='Resolution $(\sigma)$: $%.2f \pm %.2f$ %%  ' % (delE, delE_err))
                if epoch == "9999":
                    print 'Fitted %s Peak (@2615):\t\t%.2f +- %.2f' % (name, coeff[1], abs(coeff[2]))
                return (coeff[1], coeff_err[1]), (abs(coeff[2]), coeff_err[2])
            else:
                plt.plot(bin_centres, hist / norm_factor, label=name, color=color, lw=0.5)
                plt.fill_between(bin_centres, 0.0, hist / norm_factor, facecolor='black', alpha=0.3, interpolate=True)
                return

        mean_recon = (2614.0, 0.0)
        for i in ( range(1) if len(E_recon) is len([]) else range(5) ):
            plot_fit_spectrum(data=E_true, name="MC", color='black', fit=False)
            if len(E_recon) is not len([]):
                E_recon = E_recon/(mean_recon[0]/2614.0)
                mean_recon, sig_recon = plot_fit_spectrum(data=E_recon, name="Standard", color='firebrick', fit=True)
            mean_pred, sig_pred = plot_fit_spectrum(data=E_pred, name="Conv. NN", color='blue', fit=True)
            plt.xlabel('Energy [keV]')
            plt.ylabel('Probability')
            plt.legend(loc="lower left")
            plt.xlim(xmin=700, xmax=3000)
            plt.ylim(ymin=(1.0/float(len(E_true))), ymax=0.1)
            plt.grid(True)
            # if len(E_recon) is len([]):
            #     plt.savefig(args.folderOUT+'2prediction-spectrum/spectrum_'+ epoch +'.pdf')
            #     plt.xlim(xmin=2300, xmax=2865)
            #     plt.savefig(args.folderOUT+'2prediction-spectrum/spectrum_zoom_'+ epoch +'.pdf')
            # else:
            #     plt.savefig(args.folderOUT + '2prediction-spectrum/spectrum_' + epoch + '_' + str(i) + '.pdf')
            #     plt.xlim(xmin=2250, xmax=2865)
            #     plt.savefig(args.folderOUT + '2prediction-spectrum/spectrum_zoom_' + epoch + '_' + str(i) + '.pdf')
            plt.gca().set_yscale('log')
            plt.xlim(xmin=700, xmax=3000)
            if len(E_recon) is len([]):
                plt.savefig(args.folderOUT + '2prediction-spectrum/spectrum_' + epoch + '_log.pdf')
                plt.xlim(xmin=2000, xmax=3000)
                plt.savefig(args.folderOUT + '2prediction-spectrum/spectrum_zoom_' + epoch + '_log.pdf')
            else:
                plt.savefig(args.folderOUT + '2prediction-spectrum/spectrum_' + epoch + '_' + str(i) + '_log.pdf')
                plt.xlim(xmin=2000, xmax=3000)
                plt.savefig(args.folderOUT + '2prediction-spectrum/spectrum_zoom_' + epoch + '_' + str(i) + '_log.pdf')
            plt.clf()
        plt.close()
        if len(E_recon) is not len([]):
            return (mean_pred[0], mean_pred[1]), (abs(sig_pred[0]), sig_pred[1]), E_recon
        return (mean_pred[0], mean_pred[1]), (abs(sig_pred[0]), sig_pred[1])

    # histogram of the data
    def plot_residual_histo(E_x, E_y, epoch, name_x='', name_y=''):
        dE  = E_x - E_y
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
                print epoch, '\tfit did not work\t', i
                coeff, coeff_err = [hist_dE[peak], bin_centres[peak], 50.0*(i+1)], [0.0, 0.0, 0.0]
        if epoch == "9999":
            print 'Fitted Residual %s:\t%.2f +- %.2f'%(name_y, coeff[1], abs(coeff[2]))
        plt.plot(range(-250,250), gauss_zero(range(-250,250), *coeff), lw=2, color='red',
                 label='%s\n$\mu=%.1f \pm %.1f$\n$\sigma=%.1f \pm %.1f$'%(name_y, coeff[1], coeff_err[1], abs(coeff[2]), coeff_err[2]))
        plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
        plt.ylabel('Probability')
        plt.legend(loc="best")
        plt.xlim(xmin=-250, xmax=250)
        plt.ylim(ymin=0.0, ymax=0.025)
        plt.grid(True)
        plt.savefig(args.folderOUT+'4residual-histo/histogram_'+epoch+'_'+name_y+'.pdf')
        plt.clf()
        plt.close()
        return ((coeff[1], coeff_err[1]), (abs(coeff[2]), coeff_err[2]))

    # scatter
    def plot_scatter_hist2d(E_x, E_y, epoch, name_x='', name_y=''):
        dE = E_y - E_x
        if epoch == "9999":
            print 'Model performance %s: <dE>=%.2f, std(dE)=%.2f' % (name_y, np.mean(dE), np.std(dE))
        hist, xbins, ybins = np.histogram2d(E_x, E_y, bins=100, normed=True)
        extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            im = plt.imshow(np.log10(np.ma.masked_where(hist == 0, hist).T), cmap=plt.get_cmap('viridis'),
                   interpolation='nearest', origin='lower', extent=extent,
                   label='%s\n$\mu=%.1f, \sigma=%.1f$' % ('MC data', np.mean(dE), np.std(dE)))
        plt.plot((500, 3000), (500, 3000), 'k--')
        cbar = plt.colorbar(im, fraction=0.025, pad=0.04, format='$10^{%.1f}$')
        cbar.set_label('Probability')
        plt.legend(loc="best")
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('%s Energy [keV]' % (name_y))
        plt.xlim(xmin=500, xmax=3000)
        plt.ylim(ymin=500, ymax=3000)
        plt.grid(True)
        plt.savefig(args.folderOUT + '3prediction-scatter/prediction_'+epoch+'_'+name_y+'.pdf')
        plt.clf()
        plt.close()
        return

    # scatter-residual
    def plot_residual_hist2d(E_x, E_y, epoch, name_x='', name_y=''):
        dE = E_y - E_x
        hist, xbins, ybins = np.histogram2d(E_x, dE, range=[[700,3000],[-250,250]], bins=100, normed=True )
        extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
        aspect = (2300)/(600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            im = plt.imshow(np.log10(np.ma.masked_where(hist == 0, hist).T), cmap=plt.get_cmap('viridis'),
                   interpolation='nearest', origin='lower', extent=extent, aspect=aspect)
        plt.plot((700, 3000), (0,0), color='black')
        cbar = plt.colorbar(im, fraction=0.025, pad=0.04, format='$10^{%.1f}$')
        cbar.set_label('Probability')
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
        plt.xlim(xmin=700, xmax=3000)
        plt.ylim(ymin=-250, ymax=250)
        plt.grid(True)
        plt.savefig(args.folderOUT+'5residual-scatter/residual_'+epoch+'_'+name_y+'.pdf')
        plt.clf()
        plt.close()
        return

    # mean-residual
    def plot_residual_scatter_mean(E_x, E_y, epoch, name_x='', name_y=''):
        dE = E_y - E_x
        bin_edges = [ x for x in range(650,3350,50) ]
        bins = [ [] for x in range(650,3300,50) ]
        for i in range(len(dE)):
            bin = np.digitize(E_x[i], bin_edges) - 1
            bins[bin].append(dE[i])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bins = [ np.array(bin) for bin in bins]
            mean = [ np.mean(bin)  for bin in bins]
            stda = [ np.std(bin)   for bin in bins]
        bin_width=((bin_edges[1]-bin_edges[0])/2.0)
        plt.errorbar((np.array(bin_edges[:-1])+bin_width), mean, xerr=bin_width, yerr=stda, fmt="none")
        plt.plot((500, 3200), (0,0), color='black')
        plt.xlim(xmin=500, xmax=3200)
        plt.ylim(ymin=-100, ymax=100)
        plt.grid(True)
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
        plt.savefig(args.folderOUT+'6residual-mean/residual_mean_'+epoch+'_'+name_y+'.pdf')
        plt.clf()
        plt.close()
        return

    os.system("mkdir -p %s " % (args.folderOUT + "2prediction-spectrum/"))
    os.system("mkdir -p %s " % (args.folderOUT + "3prediction-scatter/" ))
    os.system("mkdir -p %s " % (args.folderOUT + "4residual-histo/"     ))
    os.system("mkdir -p %s " % (args.folderOUT + "5residual-scatter/"   ))
    os.system("mkdir -p %s " % (args.folderOUT + "6residual-mean/"      ))

    # plot_losses(np.genfromtxt(args.folderOUT + 'history.csv', delimiter=',', names=True))
    obs = {}
    if args.reconstruct or args.reconstruct_gamma:
        obs['peak_pos'], obs['peak_sig'], E_recon = plot_spectrum(E_pred, E_true, E_recon, epoch)
    else:
        obs['peak_pos'], obs['peak_sig'] = plot_spectrum(E_pred, E_true, E_recon, epoch)
    obs['resid_pos'], obs['resid_sig'] = plot_residual_histo(E_x=E_true, E_y=E_pred, epoch=epoch, name_x='MC', name_y='Conv. NN')
    plot_scatter_hist2d(E_x=E_true, E_y=E_pred, epoch=epoch, name_x='MC', name_y='Conv. NN')
    plot_residual_hist2d(E_x=E_true, E_y=E_pred, epoch=epoch, name_x='MC', name_y='Conv. NN')
    plot_residual_scatter_mean(E_x=E_true, E_y=E_pred, epoch=epoch, name_x='MC', name_y='Conv. NN')
    if args.reconstruct:
        plot_residual_histo(E_x=E_true, E_y=E_recon, epoch=epoch, name_x='MC', name_y='Standard')
        plot_scatter_hist2d(E_x=E_true, E_y=E_recon, epoch=epoch, name_x='MC', name_y='Standard')
        plot_residual_hist2d(E_x=E_true, E_y=E_recon, epoch=epoch, name_x='MC', name_y='Standard')
        plot_residual_scatter_mean(E_x=E_true, E_y=E_recon, epoch=epoch, name_x='MC', name_y='Standard')
    return obs

def final_plots(args, obs):
    if obs == {} :
        print 'final plots \t save.p empty'
        return
    print 'final plots \t start'
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
        plt.plot(epoch, obs_sort['loss'], label='training')
        plt.plot(epoch, obs_sort['val_loss'], label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.gca().set_yscale('log')
        plt.legend(loc="best")
        plt.savefig(args.folderOUT + 'loss.pdf')
        plt.clf()
        plt.close()

        plt.plot(epoch, obs_sort['mean_absolute_error'], label='training')
        plt.plot(epoch, obs_sort['val_mean_absolute_error'], label='validation')
        plt.grid(True)
        plt.legend(loc="best")
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(args.folderOUT + 'mean_absolute_error.pdf')
        plt.clf()
        plt.close()
    except:
        print 'no loss / mean_err plot possible'

    plt.errorbar(epoch, obs_sort['peak_pos'][:,0], xerr=0.5, yerr=obs_sort['peak_pos'][:,1], fmt="none", lw=2)
    plt.axhline(y=2614, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Energy [keV]')
    plt.savefig(args.folderOUT + '2prediction-spectrum/ZZZ_Peak.pdf')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['peak_sig'][:,0], xerr=0.5, yerr=obs_sort['peak_sig'][:,1], fmt="none", lw=2)
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Width [keV]')
    plt.savefig(args.folderOUT + '2prediction-spectrum/ZZZ_Width.pdf')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['resid_pos'][:, 0], xerr=0.5, yerr=obs_sort['resid_pos'][:, 1], fmt="none", lw=2)
    plt.axhline(y=0, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Offset [keV]')
    plt.savefig(args.folderOUT + '4residual-histo/ZZZ_Offset.pdf')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['resid_sig'][:, 0], xerr=0.5, yerr=obs_sort['peak_sig'][:, 1], fmt="none", lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Width [keV]')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.savefig(args.folderOUT + '4residual-histo/ZZZ_Width.pdf')
    plt.clf()
    plt.close()

    print 'final plots \t end'
    return

def make_plots_data(args, E_DCNN, E_EXO, E_Light, sources):
    def plot_spectrum(E_pred, E_recon, sources):
        def plot_fit_spectrum(data, color="", name=""):
            hist, bin_edges = np.histogram(data, bins=1200, range=(0, 12000), density=False)
            norm_factor = float(len(data))
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            peak = np.argmax(hist[np.digitize(2400, bin_centres):np.digitize(2800, bin_centres)]) + np.digitize(2400,
                                                                                                                bin_centres)
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
            print 'Fitted %s Peak (@2614):\t\t%.2f +- %.2f' % (name, coeff[1], abs(coeff[2]))
            return (coeff[1], coeff_err[1]), (abs(coeff[2]), coeff_err[2])

        mean_recon = (2614.0, 0.0)
        for i in range(5):
            mean_pred, sig_pred = plot_fit_spectrum(data=E_pred, name="Conv. NN", color='blue')
            E_recon = E_recon/(mean_recon[0]/2614.0)
            mean_recon, sig_recon = plot_fit_spectrum(data=E_recon, name="Standard", color='firebrick')
            plt.xlabel('Energy [keV]')
            plt.ylabel('Counts')
            plt.legend(loc="best")
            plt.xlim(xmin=650, xmax=3000)
            plt.ylim(ymin=1.e-4, ymax=1.e-1 )
            plt.grid(True)
            # plt.savefig(args.folderOUT + '0physics-data/'+args.nb_weights+'_spectrum_data_' + str(i) + '.pdf')
            plt.gca().set_yscale('log')
            plt.savefig(args.folderOUT + '0physics-data/' + args.nb_weights + '/' + args.nb_weights + '_' + sources + '_spectrum_data_'  + str(i) + '_log.pdf')
            plt.clf()
        plt.close()
        return E_recon

    def get_reconstructed_spectrum(E_pred, E_recon, sources):
        from scipy.optimize import curve_fit
        for data in [E_pred,E_recon]:
            if data is E_pred:
                name = 'Conv. NN'
                color = 'blue'
            if data is E_recon:
                name = 'Standard'
                color = 'firebrick'
            hist, bin_edges = np.histogram(data, bins=200, range=(650, 3500), density=False)
            norm_factor = float(len(data))
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            if 'th' in sources:
                peak = np.argmax(hist[np.digitize(2400, bin_centres):np.digitize(2800, bin_centres)]) + np.digitize(2400, bin_centres)
            else:
                peak = np.argmax(hist[np.digitize(1500, bin_centres):np.digitize(2000, bin_centres)]) + np.digitize(1500, bin_centres)
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
        if 'th' in sources:
            plt.axvline(x=2614, lw=2, color='black')
        plt.gca().set_yscale('log')
        plt.grid(True)
        plt.xlabel('uncalibrated Energy [keV]')
        plt.ylabel('Counts')
        plt.xlim(xmin=1000, xmax=3000)
        plt.ylim(ymin=5.e-4, ymax=5.e-2)
        plt.savefig(args.folderOUT + '0physics-data/' + args.nb_weights + '/'+args.nb_weights+ '_' + sources +'_spectrum_data.pdf')
        plt.close()
        plt.clf()
        return

    def plot_scatter(E_x, E_y, name_x, name_y, sources):
        dE = E_y - E_x
        plt.scatter(E_x, E_y, label='%s\n$\mu=%.1f, \sigma=%.1f$'%('training set', np.mean(dE), np.std(dE)))
        plt.plot((500, 3000), (500, 3000), 'k--')
        plt.legend(loc="best")
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('%s Energy [keV]' % (name_y))
        plt.xlim(xmin=500, xmax=3000)
        plt.ylim(ymin=500, ymax=3000)
        plt.grid(True)
        plt.savefig(args.folderOUT+'0physics-data/' + args.nb_weights + '/'+args.nb_weights+ '_' + sources + '_scatter_data.pdf')
        plt.clf()
        plt.close()
        return

    def plot_scatter_hist2d(E_x, E_y, name_x, name_y, sources):
        dE = E_y - E_x
        hist, xbins, ybins = np.histogram2d(E_x, E_y, bins=180, normed=True)
        extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            plt.imshow(np.log10(np.ma.masked_where(hist == 0, hist).T), cmap=plt.get_cmap('viridis'),
                   interpolation='nearest', origin='lower', extent=extent,
                   label='%s\n$\mu=%.1f, \sigma=%.1f$' % ('physics data', np.mean(dE), np.std(dE)))
        plt.plot((500, 3000), (500, 3000), 'k--')
        cbar = plt.colorbar(format='$10^{%.1f}$')
        cbar.set_label('Probability')
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('%s Energy [keV]' % (name_y))
        plt.xlim(xmin=500, xmax=3000)
        plt.ylim(ymin=500, ymax=3000)
        plt.grid(True)
        plt.savefig(args.folderOUT+'0physics-data/' + args.nb_weights + '/'+args.nb_weights+ '_' + sources +'_scatter_hist2d_data.pdf')
        plt.clf()
        plt.close()
        return

    def plot_scatter_density(E_x, E_y, name_x, name_y, sources):
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
        plt.plot((500, 3000), (500, 3000), 'k--')
        plt.colorbar()
        plt.legend(loc="best")
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('%s Energy [keV]' % (name_y))
        plt.xlim(xmin=500, xmax=3000)
        plt.ylim(ymin=500, ymax=3000)
        plt.grid(True)
        plt.savefig(args.folderOUT+'0physics-data/' + args.nb_weights + '/'+args.nb_weights+ '_' + sources +'_scatter_density_data.pdf')
        plt.clf()
        plt.close()
        return

    def plot_residual_scatter(E_x, E_y, name_x, name_y, sources):
        dE = E_y - E_x
        plt.scatter(E_x, dE)
        plt.plot((500, 3000), (0,0), color='black')
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
        plt.xlim(xmin=500, xmax=3000)
        plt.ylim(ymin=-300, ymax=300)
        plt.grid(True)
        plt.savefig(args.folderOUT+'0physics-data/' + args.nb_weights + '/'+args.nb_weights+ '_' + sources +'_residual_data.pdf')
        plt.clf()
        plt.close()
        return

    def plot_residual_hist2d(E_x, E_y, name_x, name_y, sources):
        from matplotlib.colors import LogNorm
        dE = E_y - E_x
        hist, xbins, ybins = np.histogram2d(E_x, dE, range=[[700,3000],[-250,250]], bins=180, normed=True )
        extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
        aspect = (2300)/(600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            plt.imshow(np.log10(np.ma.masked_where(hist == 0, hist).T), cmap=plt.get_cmap('viridis'),
                   interpolation='nearest', origin='lower', extent=extent, aspect=aspect)
        plt.plot((700, 3000), (0, 0), color='black')
        cbar = plt.colorbar(format='$10^{%.1f}$')
        cbar.set_label('Probability')
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
        plt.xlim(xmin=700, xmax=3000)
        plt.ylim(ymin=-250, ymax=250)
        plt.grid(True)
        plt.savefig(args.folderOUT+'0physics-data/' + args.nb_weights + '/'+args.nb_weights + '_' + sources + '_residual_hist2d_data.pdf')
        plt.clf()
        plt.close()
        return

    def plot_residual_density(E_x, E_y, name_x, name_y, sources):
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
        plt.plot((500, 3000), (0,0), color='black')
        plt.colorbar()
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
        plt.xlim(xmin=500, xmax=3000)
        plt.ylim(ymin=-300, ymax=300)
        plt.grid(True)
        plt.savefig(args.folderOUT+'0physics-data/' + args.nb_weights + '/'+args.nb_weights + '_' + sources + '_residual_density_data.pdf')
        plt.clf()
        plt.close()
        return

    def plot_anticorrelation_hist2d(E_x, E_y, name_x, name_y, sources):
        from matplotlib.colors import LogNorm
        hist, xbins, ybins = np.histogram2d(E_x, E_y, range=[[0,3500],[0,3500]], bins=250, normed=True )
        extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
        # aspect = (2300.0) / (12000.0)
        aspect = (3000.0) / (3000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # im = plt.imshow(np.log10(np.ma.masked_where(hist == 0, hist).T), cmap=plt.get_cmap('viridis'),
            #        interpolation='nearest', origin='lower', extent=extent, aspect=aspect)
            im = plt.imshow(np.log10(np.ma.masked_where(hist == 0, hist).T), cmap=plt.get_cmap('viridis'),
                            interpolation='nearest', origin='lower', extent=extent)
        # plt.plot((500, 3000), (500, 3000), 'k--')
        cbar = plt.colorbar(im, fraction=0.025, pad=0.04, format='$10^{%.1f}$')
        cbar.set_label('Probability')
        plt.xlabel('%s Energy [keV]' % (name_x))
        plt.ylabel('%s Energy [keV]' % (name_y))
        plt.xlim(xmin=700, xmax=3200)
        plt.ylim(ymin=700, ymax=3200)
        plt.grid(True)
        plt.savefig(args.folderOUT+'0physics-data/' + args.nb_weights + '/'+args.nb_weights + '_' + sources + '_anticorr_'+name_x+'_data.pdf')
        plt.clf()
        plt.close()
        return

    E_DCNN_new = plot_spectrum(E_pred=E_DCNN, E_recon=E_DCNN, sources=sources)
    E_EXO_new = plot_spectrum(E_pred=E_DCNN, E_recon=E_EXO, sources=sources)
    E_Light_new = plot_spectrum(E_pred=E_DCNN, E_recon=E_Light/3.0, sources=sources)
    plot_anticorrelation_hist2d(E_x=E_EXO_new, E_y=E_Light_new, name_x='Standard', name_y='Light', sources=sources)
    plot_anticorrelation_hist2d(E_x=E_DCNN_new, E_y=E_Light_new, name_x='Conv. NN', name_y='Light', sources=sources)

    get_reconstructed_spectrum(E_pred=E_DCNN, E_recon=E_EXO, sources=sources)
    # plot_scatter(E_x=E_EXO, E_y=E_DCNN, name_x='EXO-200', name_y='DCNN', sources=sources)
    plot_scatter_hist2d(E_x=E_EXO, E_y=E_DCNN, name_x='EXO-200', name_y='DCNN', sources=sources)
    # plot_residual_scatter(E_x=E_EXO, E_y=E_DCNN, name_x='EXO-200', name_y='DCNN', sources=sources)
    plot_residual_hist2d(E_x=E_EXO, E_y=E_DCNN, name_x='EXO-200', name_y='DCNN', sources=sources)
    # plot_anticorrelation_hist2d(E_x=E_EXO, E_y=E_Light, name_x='EXO-200', name_y='Light', sources=sources)
    # plot_anticorrelation_hist2d(E_x=E_DCNN, E_y=E_Light, name_x='DCNN', name_y='Light', sources=sources)
    return

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()