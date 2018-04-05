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
    args, files = make_organize()
    # frac_train = {'thss': 0.0, 'thms': 0.975, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.0} #th training
    # frac_val   = {'thss': 0.0, 'thms': 0.025, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.0}
    # frac_train = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.95} #normal
    # frac_val   = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.05}
    # # frac_train = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.8}  # normal + test
    # # frac_val = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.2}
    # splitted_files = split_data(args, files, frac_train=frac_train, frac_val=frac_val)
    #
    # plot.get_energy_spectrum_mixed(args, splitted_files['train'], add='train')
    # plot.get_energy_spectrum_mixed(args, splitted_files['val'], add='val')
    # plot.get_energy_spectrum_mixed(args, files, add='all')
    #
    # train_model(args, splitted_files, get_model(args), args.nb_batch * args.nb_GPU)

    print 'final plots \t start'
    plot.final_plots(folderOUT=args.folderOUT, obs=pickle.load(open(args.folderOUT + "save.p", "rb")))
    print 'final plots \t end'

    print '===================================== Program finished =============================='

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def make_organize():
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-out', dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderOUT Path')
    parser.add_argument('-in', dest='folderIN', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/Data_MC', help='folderIN Path')
    parser.add_argument('-model', dest='folderMODEL', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderMODEL Path')
    parser.add_argument('-gpu', type=int, dest='nb_GPU', default=1, choices=[1, 2, 3, 4], help='nb of GPU')
    parser.add_argument('-epoch', type=int, dest='nb_epoch', default=1, help='nb Epochs')
    parser.add_argument('-batch', type=int, dest='nb_batch', default=16, help='Batch Size')
    # parser.add_argument('-multi', dest='multiplicity', default='SS', help='Choose Event Multiplicity (SS / SS+MS)')
    parser.add_argument('-weights', dest='nb_weights', default='final', help='Load weights from Epoch')
    # parser.add_argument('-position', dest='position', default=['S5'], choices=['S2', 'S5', 'S8'], help='sources position')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume Training')
    parser.add_argument('--test', dest='test', action='store_true', help='Only reduced data')
    args, unknown = parser.parse_known_args()

    folderIN, files = {}, {}
    args.sources = ["thss", "thms", "coss", "coms", "rass", "rams", "gass", "gams", "unss", 'unms']
    # args.sources = ["thss", "thms", "coss", "rass", "gass", "unss", 'unms']
    args.label = {
        'thss': "Th228-SS",
        'rass': "Ra226-SS",
        'rams': "Ra226-SS+MS",
        'coss': "Co60-SS",
        'coms': "Co60-SS+MS",
        'gass': "Gamma-SS",
        'gams': "Gamma-SS+MS",
        'unss': "Uniform-SS",
        'thms': "Th228-SS+MS",
        'unms': "Uniform-SS+MS"}
    endings = {
        'thss': "Th228_Wfs_SS_S5_MC/",
        'thms': "Th228_Wfs_SS+MS_S5_MC/",
        'rass': "Ra226_Wfs_SS_S5_MC/",
        'rams': "Ra226_Wfs_SS+MS_S5_MC/",
        'coss': "Co60_Wfs_SS_S5_MC/",
        'coms': "Co60_Wfs_SS+MS_S5_MC/",
        'gass': "Gamma_Wfs_SS_S5_MC/",
        'gams': "Gamma_Wfs_SS+MS_S5_MC/",
        'unss': "Uniform_Wfs_SS_S5_MC/",
        'unms': "Uniform_Wfs_SS+MS_S5_MC/"}

    for source in args.sources:
        folderIN[source] = os.path.join(args.folderIN, endings[source])
        files[source] = [os.path.join(folderIN[source], f) for f in os.listdir(folderIN[source]) if os.path.isfile(os.path.join(folderIN[source], f))]
        print 'Input  Folder: (', source, ')\t', folderIN[source]
    args.folderOUT = os.path.join(args.folderOUT,'')
    args.folderMODEL = os.path.join(args.folderMODEL,'')
    args.folderIN = folderIN

    if args.nb_weights != 'final': args.nb_weights=str(args.nb_weights).zfill(3)
    if not os.path.exists(args.folderOUT+'models'):
        os.makedirs(args.folderOUT+'models')

    print 'Output Folder:\t\t'  , args.folderOUT
    if args.resume:
        print 'Model Folder:\t\t', args.folderMODEL
    print 'Number of GPU:\t\t', args.nb_GPU
    print 'Number of Epoch:\t', args.nb_epoch
    print 'BatchSize:\t\t', args.nb_batch, '\n'
    return args, files

def split_data(args, files, frac_train, frac_val):
    import cPickle as pickle
    if args.resume:
        os.system("cp %s %s" % (args.folderMODEL + "splitted_files.p", args.folderOUT + "splitted_files.p"))
        print 'load splitted files from %s' % (args.folderMODEL + "splitted_files.p")
        return pickle.load(open(args.folderOUT + "splitted_files.p", "rb"))
    else:
        import random
        splitted_files= {'train': {}, 'val': {}, 'test': {}}
        print "Source\tTotal\tTrain\tValid\tTest"
        for source in args.sources:
            if (frac_train[source] + frac_val[source]) > 1.0 : print 'check file fractions!' ; exit()
            num_train = int(round(len(files[source]) * frac_train[source]))
            num_val   = int(round(len(files[source]) * frac_val[source]))
            random.shuffle(files[source])
            if not args.test:
                splitted_files['train'][source] = files[source][0 : num_train]
                splitted_files['val'][source]   = files[source][num_train : num_train + num_val]
                splitted_files['test'][source]  = files[source][num_train + num_val : ]
            else:
                splitted_files['val'][source]   = files[source][0:1]
                splitted_files['test'][source]  = files[source][1:2]
                splitted_files['train'][source] = files[source][2:3]
            print "%s\t%i\t%i\t%i\t%i" % (source, len(files[source]), len(splitted_files['train'][source]), len(splitted_files['val'][source]), len(splitted_files['test'][source]))
        pickle.dump(splitted_files, open(args.folderOUT + "splitted_files.p", "wb"))
        return splitted_files

def num_events(files):
    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += len(np.asarray(f.get('trueEnergy')))
        f.close()
    return counter

def generate_event(files):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            X_True_i = np.asarray(f.get('trueEnergy'))
            wfs_i = np.asarray(f.get('wfs'))
            f.close()
            lst = range(len(X_True_i))
            random.shuffle(lst)
            for i in lst:
                yield (wfs_i[i], X_True_i[i])

def generate_batch(generator, batchSize):
    while 1:
        X, Y = [], []
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
        yield (np.asarray(X), np.asarray(Y))

def generate_batch_mixed(generator, batchSize, numEvents):
    import random
    order = [ [source] * numEvents[source] for source in numEvents.keys() ]
    order = [ item for sublist in order for item in sublist ]
    X, Y = [], []
    while 1:
        random.shuffle(order)
        for source in order:
            temp = generator[source].next()
            X.append(temp[0])
            Y.append(temp[1])
            if len(Y)==batchSize:
                yield (np.asarray(X), np.asarray(Y))
                X, Y = [], []

def predict_energy(model, generator):
    E_CNN_wfs, E_True = generator.next()
    E_CNN = np.asarray(model.predict(E_CNN_wfs, 100)[:,0])
    return (E_CNN, E_True)

def generate_event_reconstruction(files):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            X_True_i = np.asarray(f.get('trueEnergy'))
            X_EXO_i = np.asarray(f.get('reconEnergy'))
            wfs_i = np.asarray(f.get('wfs'))
            isSS_i = ~np.asarray(f.get('isSS'))  # inverted because of logic error in file production
            f.close()
            lst = range(len(X_True_i))
            random.shuffle(lst)
            for i in lst:
                isSS = isSS_i[i]
                X_True = X_True_i[i]
                X_EXO = X_EXO_i[i]
                wfs = wfs_i[i]
                yield (wfs, X_True, X_EXO, isSS)

def generate_batch_reconstruction(generator, batchSize):
    while 1:
        X, Y, Z, SS = [], [], [], []
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            Z.append(temp[2])
            SS.append(temp[3])
        yield (np.asarray(X), np.asarray(Y), np.asarray(Z), np.asarray(SS))

def predict_energy_reconstruction(model, generator):
    E_CNN_wfs, E_True, E_EXO, isSS = generator.next()
    E_CNN = np.asarray(model.predict(E_CNN_wfs, 100)[:, 0])
    return (E_CNN, E_True, E_EXO, isSS)

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
        self.events_val = 10000
        self.events_per_batch = 1000
        self.val_iterations = plot.round_down(self.events_val, self.events_per_batch) / self.events_per_batch
        # self.gen_val = generate_batch_reconstruction(generate_event_reconstruction(np.concatenate(self.files['val'].values()).tolist()), self.events_per_batch)
        self.gen_val = generate_batch_reconstruction(generate_event_reconstruction(self.files['val']['thms']+self.files['test']['thms']), self.events_per_batch)

    def on_train_begin(self, logs={}):
        self.losses = []
        if self.args.resume: os.system("cp %s %s" % (self.args.folderMODEL + "save.p", self.args.folderOUT + "save.p"))
        else: pickle.dump({}, open(self.args.folderOUT + "save.p", "wb"))
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        E_CNN, E_EXO, E_True, isSS = [], [], [], []
        for i in xrange(self.val_iterations):
            E_CNN_temp, E_True_temp, E_EXO_temp, isSS_temp = predict_energy_reconstruction(self.model, self.gen_val)
            E_True.extend(E_True_temp)
            E_CNN.extend(E_CNN_temp)
            E_EXO.extend(E_EXO_temp)
            isSS.extend(isSS_temp)
        dataIn = {'E_CNN': np.asarray(E_CNN), 'E_EXO': np.asarray(E_EXO), 'E_True': np.asarray(E_True), 'isSS': np.asarray(isSS)}
        obs = plot.make_plots(self.args.folderOUT, dataIn=dataIn, epoch=str(epoch), sources='th', position='S5')
        self.dict_out = pickle.load(open(self.args.folderOUT + "save.p", "rb"))
        self.dict_out[str(epoch)] = {'E_CNN': E_CNN, 'E_True': E_True, 'E_EXO': E_EXO,
                                     'peak_pos': obs['peak_pos'],
                                     'peak_sig': obs['peak_sig'],
                                     'resid_pos': obs['resid_pos'],
                                     'resid_sig': obs['resid_sig'],
                                     'loss': logs['loss'], 'mean_absolute_error': logs['mean_absolute_error'],
                                     'val_loss': logs['val_loss'], 'val_mean_absolute_error': logs['val_mean_absolute_error']}
        pickle.dump(self.dict_out, open(self.args.folderOUT + "save.p", "wb"))
        plot.final_plots(folderOUT=self.args.folderOUT, obs=pickle.load(open(self.args.folderOUT + "save.p", "rb")))
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
        model.add(MaxPooling2D((2, 2), border_mode=padding))

        model.add(Convolution2D(128, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), border_mode=padding))

        model.add(Convolution2D(256, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), border_mode=padding))

        model.add(Convolution2D(256, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), border_mode=padding))

        # regression part
        model.add(Flatten())
        model.add(Dense(32, activation=activation, init=init, W_regularizer=regul))
        model.add(Dense(8, activation=activation, init=init, W_regularizer=regul))
        model.add(Dense(1 , activation=activation, init=init))
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
    print "\nFirst Epoch:\t", epoch_start
    print model.summary(), "\n"
    print "\n"
    return model, epoch_start

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------
def train_model(args, files, (model, epoch_start), batchSize):
    from keras import callbacks
    start = time.time()

    gen_train, gen_val, numEvents_train, numEvents_val = {}, {}, {}, {}
    for source in args.sources:
        gen_train[source] = generate_event(files['train'][source])
        gen_val[source]   = generate_event(files['val'][source])
        numEvents_train[source] = num_events(files['train'][source])
        numEvents_val[source]   = num_events(files['val'][source])

    if args.nb_GPU>1:
        model = make_parallel(model, args.nb_GPU)

    model.save(args.folderOUT + "models/model-initial.hdf5")
    model.save_weights(args.folderOUT + "models/weights-initial.hdf5")
    print 'training los'
    model.fit_generator(
        generate_batch_mixed(gen_train, batchSize, numEvents_train),
        samples_per_epoch=plot.round_down(sum(numEvents_train.values()), batchSize),
        nb_epoch=args.nb_epoch+epoch_start,
        verbose=1,
        validation_data=generate_batch_mixed(gen_val, batchSize, numEvents_val),
        nb_val_samples=plot.round_down(sum(numEvents_val.values())  , batchSize),
        initial_epoch=epoch_start,
        callbacks=[
            callbacks.CSVLogger(args.folderOUT + 'history.csv', append=args.resume),
            callbacks.ModelCheckpoint(args.folderOUT + 'models/weights-{epoch:03d}.hdf5', save_weights_only=True, period=int(args.nb_epoch/100)),
            Histories(args, files)
        ])
    print 'training stop'
    model.save(args.folderOUT+"models/model-final.hdf5")
    model.save_weights(args.folderOUT+"models/weights-final.hdf5")

    end = time.time()
    print "\nElapsed time:\t%.2f minutes\tor rather\t%.2f hours\n" % (((end-start)/60.),((end-start)/60./60.))

    print 'Model performance\tloss\t\tmean_abs_err'
    print '\tTrain:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(np.concatenate(files['train'].values()).tolist()), batchSize), val_samples=128))
    print '\tValid:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(np.concatenate(files['val'].values()).tolist())  , batchSize), val_samples=128))
    return model

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
