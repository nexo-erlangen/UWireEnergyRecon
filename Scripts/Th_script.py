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

start = time.clock()

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-out', dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderOUT Path')
parser.add_argument('-in' , dest='folderIN' , default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Data', help='folderIN Path')
parser.add_argument('-model', dest='folderMODEL', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderMODEL Path')
parser.add_argument('-gpu', type=int, dest='nb_GPU' , default=1, choices=[1,2,3,4], help='nb of GPU')
parser.add_argument('-epoch', type=int, dest='nb_epoch', default=1 , help='nb Epochs')
parser.add_argument('-batch', type=int, dest='nb_batch', default=16, help='Batch Size')
parser.add_argument('--resume', dest='resume', action='store_true', help='Resume Training')

args, unknown = parser.parse_known_args()
args.folderIN=os.path.join(args.folderIN,'')
args.folderOUT=os.path.join(args.folderOUT,'')
args.folderMODEL=os.path.join(args.folderMODEL,'')

files = [os.path.join(args.folderIN, f) for f in os.listdir(args.folderIN) if os.path.isfile(os.path.join(args.folderIN, f))]

if not os.path.exists(args.folderOUT+'models'):
    os.makedirs(args.folderOUT+'models')

print 'Input  Folder:\t\t'  , args.folderIN
print 'Output Folder:\t\t'  , args.folderOUT
if args.resume:
    print 'Model Folder:\t\t', args.folderMODEL
print 'Number of GPU:\t\t', args.nb_GPU
print 'Number of Epoch:\t', args.nb_epoch
print 'BatchSize:\t\t', args.nb_batch
print

def split_data(files,frac_val,frac_test):
    random.shuffle(files)
    num_val  = int(len(files)*frac_val)
    num_test = int(len(files)*frac_test)
    return (files[num_val+num_test:], files[0:num_val], files[num_val:num_val+num_test])

def num_events(files):
    counter=0
    for filename in files:
        f = h5py.File(str(filename))
        counter += len(np.array(f.get('trueEnergy')))
        f.close()
    return counter

files_train, files_val, files_test = split_data(files,frac_val=0.1,frac_test=0.1)
# print "\tFiles\tEvents"
# print "Total:\t%i\t%i" % (len(files) ,num_events(files))
# print "Train:\t%i\t%i" % (len(files_train) ,num_events(files_train))
# print "Valid:\t%i\t%i" % (len(files_val), num_events(files_val))
# print "Test:\t%i\t%i" % (len(files_test), num_events(files_test))
print "\tFiles"
print "Total:\t%i" % (len(files))
print "Train:\t%i" % (len(files_train))
print "Valid:\t%i" % (len(files_val))
print "Test:\t%i" % (len(files_test))
print

def get_energy_spectrum(files):
    entry = []
    for filename in files:
        f = h5py.File(str(filename))
        temp=np.array(f.get('trueEnergy'))
        for i in range(len(temp)):
            entry.append(temp[i])
        f.close()
    hist, bin_edges = np.histogram(entry, bins=210, range=(700,3000), density=True)
    # plt.plot(bin_edges[:-1], hist)
    # # plt.plot(bin_edges[:-1], hist, "ro")
    # plt.gca().set_yscale('log')
    # plt.grid(True)
    # plt.xlabel('Energy')
    # plt.ylabel('Probability')
    # plt.xlim(xmin=700, xmax=3000)
    # plt.savefig(args.folderOUT + 'spectrum.png')
    # plt.close()
    # plt.clf()
    hist_inv=np.zeros(hist.shape)
    for i in range(len(hist)):
        try:
            hist_inv[i]=1.0/float(hist[i])
        except:
            pass
            # print "WARNING: zero weight -->\t Energy:\t", bin_edges[i], "  --  ", bin_edges[i+1]
    hist_inv = hist_inv / hist_inv.sum(axis=0, keepdims=True)
    plt.plot(bin_edges[:-1], hist_inv)
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy')
    plt.ylabel('Weight')
    plt.xlim(xmin=700, xmax=3000)
    plt.savefig(args.folderOUT + 'spectrum_inverse.png')
    plt.close()
    plt.clf()
    return (hist_inv, bin_edges[:-1])

def get_weight(Y, hist, bin_edges):
    return hist[np.digitize(Y, bin_edges) - 1]

def round_down(num, divisor):
    return num - (num%divisor)

def generate_event(files):
    hist_inv, bin_edges = get_energy_spectrum(files)
    while 1:
        random.shuffle(files)
        for filename in files:
            f=h5py.File(str(filename))
            Y_true_array=np.array(f.get('trueEnergy'))
            X_init_array=np.array(f.get('wfs'))
            lst = range(len(Y_true_array))
            random.shuffle(lst)
            for i in lst:
                Y_true=Y_true_array[i]
                X_norm=X_init_array[i]
                # yield (X_norm, Y_true, 1.0)
                yield (X_norm, Y_true, get_weight(Y_true, hist_inv, bin_edges))
            f.close()

print "weights differ"

def generate_batch(generator, batchSize):
    while 1:
        X,Y,W=[],[],[]
        for i in range(batchSize):
            temp=generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            W.append(temp[2])
        yield (np.array(X), np.array(Y), np.array(W))

# for j in range(50):
#     X,Y=generate_event(files_train).next()
#     T=np.array(range(512,512+1024))
#     fig, ax = plt.subplots(1)
#     for i in range(76):
#         ax.plot(T,X[:,i]+15.*i, label=str(i), color='black')
#     ax.set(xlabel='time [$\mu$s]', ylabel='amplitude + offset')
#     fig.savefig(args.folderOUT + 'waveforms/'+str(j)+'.png')
#     plt.close()
#     plt.clf()
#
#     ax = plt.gca()
#     plt.imshow((np.swapaxes(X,0,1))[:,:,0])
#     ax.set_aspect(1./ax.get_data_ratio())
#     plt.colorbar()
#     plt.savefig(args.folderOUT+"images/"+str(j)+'.png')
#     plt.close()
#     plt.clf()

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
# Define model
# ----------------------------------------------------------

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D

def add_block(nfilters, dropout=False, **kwargs):
    try:
        model
    except NameError:
        model = Sequential()
    model.add(Convolution2D(nfilters, 5, 3, border_mode='same', init="he_normal", **kwargs))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    else:
        model.add(MaxPooling2D((4, 2), border_mode='same'))

def def_model():
    model = Sequential()
    # convolution part
    model.add(Convolution2D( 16, 5, 3, border_mode='same', init="he_normal", input_shape=(1024, 76, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((4, 2), border_mode='same'))
    model.add(Convolution2D( 32, 5, 3, border_mode='same', init="he_normal"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((4, 2), border_mode='same'))
    model.add(Convolution2D( 64, 3, 3, border_mode='same', init="he_normal"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', init="he_normal"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    # regression part
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))
    return model

if not args.resume:
    print "new Model"
    model = def_model()
    epoch_start = 0
else:
    print "load Model"
    os.system("cp %s %s" % (args.folderMODEL+"models/model.h5", args.folderOUT+"models/model.h5"))
    os.system("cp %s %s" % (args.folderMODEL + "history.csv", args.folderOUT + "history.csv"))
    model = load_model(args.folderOUT + "models/model.h5")
    # model.load_weights(args.folderOUT + "models/model.h5")
    epoch_start = 1+int(np.genfromtxt(args.folderOUT + 'history.csv', delimiter=',', names=True)['epoch'][-1])

print model.summary()
print "first epoch:\t", epoch_start

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------

# model.save(args.folderOUT+"models/model.h5")
batchSize=args.nb_batch*args.nb_GPU

if args.nb_GPU>1:
    model = make_parallel(model, args.nb_GPU)

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_absolute_error'])

model.fit_generator(
    generate_batch(generate_event(files_train), batchSize),
    samples_per_epoch=round_down(num_events(files_train), batchSize),
    nb_epoch=args.nb_epoch+epoch_start,
    verbose=1,
    validation_data=generate_batch(generate_event(files_val), batchSize),
    nb_val_samples=round_down(num_events(files_val), batchSize),
    initial_epoch=epoch_start,
    callbacks=[
        keras.callbacks.CSVLogger(args.folderOUT + 'history.csv', append=args.resume),
        keras.callbacks.ModelCheckpoint(args.folderOUT + 'models/weights-{epoch:02d}.hdf5', save_weights_only=True, period=int(args.nb_epoch/10))
    ])

model.save(args.folderOUT+"models/model.h5")
# model.save_weights(args.folderOUT+"models/weights_final.h5")

end = time.clock()
print
print 'Elapsed time: %.2f minutes' % ((end-start)/60.)
print 'Elapsed time: %.2f hours' % ((end-start)/60./60.)
print

print 'Model performance (loss, mean_abs_err)'
print 'Train: %.4f, %.4f' % tuple(model.evaluate_generator(generate_batch(generate_event(files_train), batchSize), val_samples=128))
print 'Valid: %.4f, %.4f' % tuple(model.evaluate_generator(generate_batch(generate_event(files_val)  , batchSize), val_samples=128))
print 'Test:  %.4f, %.4f' % tuple(model.evaluate_generator(generate_batch(generate_event(files_test) , batchSize), val_samples=128))
print

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------

try:
    model
except NameError:
    model = load_model(args.folderOUT + "models/model.h5")
    # model.load_weights(args.folderOUT + "models/weights-699.hdf5")

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

fig, ax = plt.subplots(1)
ax.plot(history['epoch'], history['mean_absolute_error'],     label='training')
ax.plot(history['epoch'], history['val_mean_absolute_error'], label='validation')
ax.legend()
ax.grid(True)
ax.set(xlabel='epoch', ylabel='mean absolute error')
fig.savefig(args.folderOUT+'mean_absolute_error.png')
plt.clf()

#predictions
def predict_energy(num,files):
    X, Y = [], []
    for i in range(num):
        X_temp,E_temp,dummy=generate_batch(generate_event(files), 1).next()
        X.append(model.predict(X_temp,1)[:,0])
        Y.append(E_temp)
    return (np.array(X)[:,0] , np.array(Y)[:,0])

E_train, E_true = predict_energy(1000,files_val)
dE_train = E_train - E_true
mean_train, std_train = np.mean(dE_train), np.std(dE_train)
print 'Model performance (dE = predicted - true E)'
print 'Train: <dE>=%.2f, std(dE)=%.2f' % (mean_train, std_train)

# the histogram of the data
plt.hist(dE_train, 50, normed=True, facecolor='green', alpha=0.75)
plt.xlabel('dE (Train - True) [keV]')
plt.ylabel('Probability')
plt.grid()
plt.savefig(args.folderOUT+'histogram.png')
plt.clf()

# scatter
fig, ax = plt.subplots(1)
label = '%s\n$\mu=%.1f, \sigma=%.1f$'
ax.scatter(E_true, E_train, label=label%('training set', mean_train, std_train))
ax.plot((300, 3000), (300, 3000), 'k--')
ax.legend()
ax.set_xlabel('true Energy [keV]')
ax.set_ylabel('predicted Energy [keV]')
ax.grid()
fig.savefig(args.folderOUT+'prediction.png')
plt.clf()

# scatter-residual
plt.scatter(E_true, dE_train)
plt.plot((300, 3000), (0,0))
plt.xlabel('true Energy [keV]')
plt.ylabel('residual (Train - True) [keV]')
plt.grid()
plt.savefig(args.folderOUT+'residual.png')
plt.clf()

# mean-residual
bin_edges = [x for x in range(700,3050,50)]
bins = [ [] for x in range(700,3000,50)]
for i in range(len(dE_train)):
    bin = np.digitize(E_true[i], bin_edges) - 1
    bins[bin].append(dE_train[i])
    # print E_true[i] , dE_train[i], bin, len(bins[bin])

with np.errstate(divide='ignore'):
    bins = [np.array(bin) for bin in bins]
    # mean = [np.nan_to_num(np.mean(bin)) for bin in bins]
    # stda = [np.nan_to_num(np.std(bin)) for bin in bins]
    mean = [np.mean(bin) for bin in bins]
    stda = [np.std(bin) for bin in bins]

bin_width=((bin_edges[1]-bin_edges[0])/2.0)
plt.errorbar((np.array(bin_edges[:-1])+bin_width), mean, xerr=bin_width, yerr=stda, fmt="none")
plt.plot((300, 3000), (0,0))
plt.grid(True)
plt.xlabel('true Energy [keV]')
plt.ylabel('residual (Train - True) [keV]')
plt.savefig(args.folderOUT+'residual_mean.png')
plt.clf()
