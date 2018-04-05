#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import time
import random
import os
import argparse
import warnings

def make_organize():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-out', dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderOUT Path')
    parser.add_argument('-in', dest='folderIN', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Data', help='folderIN Path')
    parser.add_argument('-model', dest='folderMODEL', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderMODEL Path')
    parser.add_argument('-gpu', type=int, dest='nb_GPU', default=1, choices=[1, 2, 3, 4], help='nb of GPU')
    parser.add_argument('-epoch', type=int, dest='nb_epoch', default=1, help='nb Epochs')
    parser.add_argument('-batch', type=int, dest='nb_batch', default=16, help='Batch Size')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume Training')
    parser.add_argument('--plot', dest='plotsonly', action='store_true', help='Only produce plots')
    parser.add_argument('-sources', dest='sources', default=['th', 'ra', 'co'], nargs="*", choices=["th", "co", "ra"], help='sources for training (ra,co,th)')
    args, unknown = parser.parse_known_args()

    if not 'th' in args.sources:
        print 'include Th in training data'
        exit()

    folderIN, files, endings = {}, {}, {}
    endings['th'] = "Th228_Wfs_SS_S5_MC/"
    endings['ra'] = "Ra226_Wfs_SS_S5_MC/"
    endings['co'] = "Co60_Wfs_SS_S5_MC/"
    for source in args.sources:
        folderIN[source] = os.path.join(args.folderIN, endings[source])
        files[source] = [os.path.join(folderIN[source], f) for f in os.listdir(folderIN[source]) if os.path.isfile(os.path.join(folderIN[source], f))]
        print 'Input  Folder:(', source, ')\t', folderIN[source]
    # args.folderIN = os.path.join(args.folderIN,'')
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

def split_data(args, files, frac_val, frac_test):
    files_train, files_val, files_test = {}, {}, {}
    print "Source\tTotal\tTrain\tValid\tTest"
    for source in args.sources:
        num_val = int(len(files[source]) * frac_val)
        num_test = int(len(files[source]) * frac_test)
        random.shuffle(files[source])
        # files_val[source]   = files[source][0:num_val]
        # files_test[source]  = files[source][num_val:num_val + num_test]
        # files_train[source] = files[source][num_val + num_test:]
        files_val[source]   = files[source][0:1]
        files_test[source]  = files[source][1:2]
        files_train[source] = files[source][2:3]
        print "%s\t%i\t%i\t%i\t%i" % (source, len(files[source]), len(files[source][num_val + num_test:]), len(files[source][0:num_val]),len(files[source][num_val:num_val + num_test]))
    return (files_train, files_val, files_test)

def num_events(files):
    counter = 0
    for filename in files:
        f = h5py.File(str(filename))
        counter += len(np.array(f.get('trueEnergy')))
        f.close()
    return counter

def get_weight(Y, hist, bin_edges):
    return hist[np.digitize(Y, bin_edges) - 1]

def round_down(num, divisor):
    return num - (num%divisor)

def get_energy_spectrum(files):
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
    plt.savefig(args.folderOUT + 'spectrum.png')
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
    plt.savefig(args.folderOUT + 'spectrum_inverse.png')
    plt.close()
    plt.clf()
    return (hist_inv, bin_edges[:-1])

def get_energy_spectrum_mixed(args, files):
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
    plt.plot(bin_edges[:-1] + bin_width, np.array(hist_mixed)/num_counts, label="combined")

    for source in args.sources:
        hist[source], bin_edges = np.histogram(entry[source], bins=280, range=(700,3500), density=False)
        plt.plot(bin_edges[:-1] + bin_width, np.array(hist[source])/num_counts, label=source)
        print "%s\t%i" % (source , len(entry[source]))
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.legend(loc="best")
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=700, xmax=3000)
    plt.savefig(args.folderOUT + 'spectrum_mixed.png')
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

def get_waveforms(number, generator):
    os.system("mkdir -p %s " % (args.folderOUT + "0waveforms/"))
    os.system("mkdir -p %s " % (args.folderOUT + "0images/"))
    for i in range(number):
        X, Y = generator.next()
        T = np.array(range(512, 512 + 1024))
        for j in range(76):
            plt.plot(T, X[:, j] + 15. * j, label=str(j), color='black')
        plt.xlabel('time [$\mu$s]')
        plt.ylabel('amplitude + offset')
        plt.savefig(args.folderOUT + '0waveforms/' + str(i) + '.png')
        plt.close()
        plt.clf()

        ax = plt.gca()
        plt.imshow((np.swapaxes(X, 0, 1))[:, :, 0])
        ax.set_aspect(1. / ax.get_data_ratio())
        plt.colorbar()
        plt.savefig(args.folderOUT + "0images/" + str(i) + '.png')
        plt.close()
        plt.clf()
        return

# ----------------------------------------------------------
# Parallel GPU Computing
# ----------------------------------------------------------
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf

def make_parallel(model, gpu_count):
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
    def __init__(self, args, splitted_files):
        self.validation_data = None
        self.args = args
        self.splitted_files = splitted_files

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # if self.args.nb_epoch<10 or not epoch % (self.args.nb_epoch/10):
        #     make_plots(self.args, self.splitted_files, self.model, str(epoch))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

# ----------------------------------------------------------
# Define model
# ----------------------------------------------------------
def get_model(args):
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D

    def def_model():
        init = "glorot_normal"
        activation = "relu"
        padding = "same"
        model = Sequential()
        # convolution part
        model.add(Convolution2D( 16, 5, 3, border_mode=padding, init=init, input_shape=(1024, 76, 1)))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), border_mode=padding))
        model.add(Convolution2D( 32, 5, 3, border_mode=padding, init=init))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), border_mode=padding))
        model.add(Convolution2D( 64, 3, 3, border_mode=padding, init=init))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), border_mode=padding))
        model.add(Convolution2D(128, 3, 3, border_mode=padding, init=init))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), border_mode=padding))
        # regression part
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(64, activation=activation, init=init))
        model.add(Dropout(0.1))
        model.add(Dense( 8, activation=activation, init=init))
        model.add(Dense( 1, activation=activation, init=init))
        return model

    if not args.resume:
        print "==================new Model==================\n"
        model = def_model()
        epoch_start = 0
    else:
        print "==================load Model==================\n"
        os.system("cp %s %s" % (args.folderMODEL+"models/model.h5", args.folderOUT+"models/model.h5"))
        os.system("cp %s %s" % (args.folderMODEL + "history.csv", args.folderOUT+"history.csv"))
        model = load_model(args.folderOUT + "models/model.h5")
        model.load_weights(args.folderOUT + "models/weights.h5", by_name=True)
        epoch_start = 1+int(np.genfromtxt(args.folderOUT+'history.csv', delimiter=',', names=True)['epoch'][-1])
    print "first epoch:\t", epoch_start
    print model.summary(), "\n"
    return model, epoch_start

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------
def train_model(args, (files_train, files_val, files_test), (model, epoch_start), batchSize):
    from keras import callbacks
    start = time.clock()

    gen_train, numEvents = {}, {}
    for source in args.sources:
        gen_train[source] = generate_event(files_train[source])
        numEvents[source] = num_events(files_train[source])

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_absolute_error'])
    model.save(args.folderOUT + "models/model.h5")
    print model.get_config()

    if args.nb_GPU>1:
        model = make_parallel(model, args.nb_GPU)

    print model.get_config()

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_absolute_error'])

    model.fit_generator(
        generate_batch_mixed(gen_train, batchSize, numEvents),
        # generate_batch(generate_event(files_train_th), batchSize),
        samples_per_epoch=11500,
        nb_epoch=args.nb_epoch+epoch_start,
        verbose=1,
        validation_data=generate_batch(generate_event(files_val['th']), batchSize),
        nb_val_samples=round_down(num_events(files_val['th']), batchSize),
        initial_epoch=epoch_start,
        callbacks=[
            callbacks.CSVLogger(args.folderOUT + 'history.csv', append=args.resume),
            callbacks.ModelCheckpoint(args.folderOUT + 'models/model-{epoch:02d}.hdf5', save_weights_only=True, period=int(args.nb_epoch/10)),
            Histories(args, (files_train, files_val, files_test))
        ])

    # model.save(args.folderOUT+"models/model_final.h5")
    model.save_weights(args.folderOUT+"models/weights.h5")
    # model.save(args.folderOUT + "models/model.h5")

    end = time.clock()
    print "\nElapsed time:\t%.2f minutes\tor rather\t%.2f hours\n" % (((end-start)/60.),((end-start)/60./60.))

    print 'Model performance (Th)\tloss\t\tmean_abs_err'
    print '\tTrain:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(files_train['th']), batchSize), val_samples=128))
    print '\tValid:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(files_val['th'])  , batchSize), val_samples=128))
    print '\tTest: \t\t%.4f\t%.4f\n'  % tuple(model.evaluate_generator(generate_batch(generate_event(files_test['th']) , batchSize), val_samples=128))
    return model

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def make_plots(args, (files_train, files_val, files_test), model, epoch=""):
    os.system("mkdir -p %s " % (args.folderOUT + "2prediction-spectrum/"))
    os.system("mkdir -p %s " % (args.folderOUT + "3prediction-scatter/" ))
    os.system("mkdir -p %s " % (args.folderOUT + "4residual-histo/"     ))
    os.system("mkdir -p %s " % (args.folderOUT + "5residual-scatter/"   ))
    os.system("mkdir -p %s " % (args.folderOUT + "6residual-mean/"      ))

    # training curves
    history = np.genfromtxt(args.folderOUT+'history.csv', delimiter=',', names=True)

    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['loss'],     label='training')
    ax.plot(history['epoch'], history['val_loss'], label='validation')
    ax.set(xlabel='epoch', ylabel='loss')
    ax.grid(True)
    ax.semilogy()
    fig.savefig(args.folderOUT+'loss.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['mean_absolute_error'],     label='training')
    ax.plot(history['epoch'], history['val_mean_absolute_error'], label='validation')
    ax.legend()
    ax.grid(True)
    ax.set(xlabel='epoch', ylabel='mean absolute error')
    fig.savefig(args.folderOUT+'mean_absolute_error.png')
    plt.clf()
    plt.close()

    #predictions
    def predict_energy(num,generator):
        X, Y = [], []
        for i in range(num):
            X_temp, E_temp = generator.next()
            X.append(model.predict(X_temp,1)[:,0])
            Y.append(E_temp)
        return (np.array(X)[:,0] , np.array(Y)[:,0])

    E_train, E_true = predict_energy(2000,generate_batch(generate_event(files_val['th']), 1))
    print E_train, E_true
    dE_train = E_train - E_true
    mean_train, std_train = np.mean(dE_train), np.std(dE_train)
    if epoch=="":
        print 'Model performance (dE = predicted - true E)'
        print 'Train: <dE>=%.2f, std(dE)=%.2f' % (mean_train, std_train)

    # spectrum of predicted/true energy
    hist_train, bin_edges = np.histogram(E_train, bins=280, range=(700, 3500), density=True)
    hist_true , bin_edges = np.histogram(E_true , bins=280, range=(700, 3500), density=True)
    bin_width = ((bin_edges[1] - bin_edges[0]) / 2.0)
    plt.plot(bin_edges[:-1] + bin_width, hist_train, label='Predict' )
    plt.plot(bin_edges[:-1] + bin_width, hist_true , label='True'    )
    plt.gca().set_yscale('log')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.grid()
    plt.savefig(args.folderOUT+'2prediction-spectrum/spectrum_'+epoch+'.png')
    plt.clf()
    plt.close()

    # histogram of the data
    plt.hist(dE_train, 150, normed=True, facecolor='green', alpha=0.75)
    plt.xlabel('dE (Train - True) [keV]')
    plt.ylabel('Probability')
    plt.grid()
    plt.savefig(args.folderOUT+'4residual-histo/histogram_'+epoch+'.png')
    plt.clf()
    plt.close()

    # scatter
    label = '%s\n$\mu=%.1f, \sigma=%.1f$'
    plt.scatter(E_true, E_train, label=label%('training set', mean_train, std_train))
    plt.plot((300, 3300), (300, 3300), 'k--')
    plt.legend()
    plt.xlabel('true Energy [keV]')
    plt.ylabel('predicted Energy [keV]')
    plt.grid()
    plt.savefig(args.folderOUT+'3prediction-scatter/prediction_'+epoch+'.png')
    plt.clf()
    plt.close()

    # scatter-residual
    plt.scatter(E_true, dE_train)
    plt.plot((300, 3000), (0,0))
    plt.xlabel('true Energy [keV]')
    plt.ylabel('residual (Train - True) [keV]')
    plt.grid()
    plt.savefig(args.folderOUT+'5residual-scatter/residual_'+epoch+'.png')
    plt.clf()
    plt.close()

    # mean-residual
    bin_edges = [ x for x in range(650,3350,50) ]
    bins = [ [] for x in range(650,3300,50) ]
    for i in range(len(dE_train)):
        bin = np.digitize(E_true[i], bin_edges) - 1
        bins[bin].append(dE_train[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bins = [ np.array(bin) for bin in bins]
        mean = [ np.mean(bin)  for bin in bins]
        stda = [ np.std(bin)   for bin in bins]

    bin_width=((bin_edges[1]-bin_edges[0])/2.0)
    plt.errorbar((np.array(bin_edges[:-1])+bin_width), mean, xerr=bin_width, yerr=stda, fmt="none")
    plt.plot((300, 3300), (0,0))
    plt.grid(True)
    plt.xlabel('true Energy [keV]')
    plt.ylabel('residual (Train - True) [keV]')
    plt.savefig(args.folderOUT+'6residual-mean/residual_mean_'+epoch+'.png')
    plt.clf()
    plt.close()

# ----------------------------------------------------------
# Program Structure
# ----------------------------------------------------------
args, files = make_organize()
splitted_files = split_data(args, files, frac_val=0.1, frac_test=0.1)
get_energy_spectrum_mixed(args, files)

if not args.plotsonly:
    modeltest = train_model(args, splitted_files, get_model(args), args.nb_batch*args.nb_GPU)

try:
    model
except NameError:
    from keras.models import load_model
    model = load_model(args.folderOUT + "models/model.h5")
    model.load_weights(args.folderOUT + "models/weights.h5", by_name=True)

print model.get_config()

make_plots(args, splitted_files, model)
