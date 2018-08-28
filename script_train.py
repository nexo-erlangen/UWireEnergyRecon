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
    frac_train = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.95} #normal
    frac_val   = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.05}
    # frac_train = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.05}
    # frac_val = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.05}
    splitted_files = split_data(args, files, frac_train=frac_train, frac_val=frac_val)

    plot.get_energy_spectrum_mixed(args, splitted_files['train'], add='train')
    plot.get_energy_spectrum_mixed(args, splitted_files['val'], add='val')
    plot.get_energy_spectrum_mixed(args, files, add='all')

    train_model(args, splitted_files, get_model(args), args.nb_batch * args.nb_GPU)

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
        random.shuffle(files)  # TODO maybe omit in future? # TODO shuffle events between files
        for filename in files:
            f = h5py.File(str(filename), "r")
            X_True_i = np.asarray(f.get('trueEnergy'))
            lst = range(len(X_True_i))
            random.shuffle(lst)
            for i in lst:
                xs_i = f['wfs'][ i ]
                xs_i = np.asarray(np.split(xs_i, 2, axis=1))
                yield (xs_i, X_True_i[i])
            f.close()

# def generate_event(files):
#     import random
#     while 1:
#         random.shuffle(files)
#         for filename in files:
#             f = h5py.File(str(filename), 'r')
#             X_True_i = np.asarray(f.get('trueEnergy'))
#             wfs_i = np.asarray(f.get('wfs'))
#             f.close()
#             lst = range(len(X_True_i))
#             random.shuffle(lst)
#             for i in lst:
#                 yield (wfs_i[i], X_True_i[i])

def generate_batch(generator, batchSize):
    while 1:
        X, Y = [], []
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
        X = np.swapaxes(np.asarray(X), 0, 1)
        yield (list(X), np.asarray(Y))
        # yield (np.asarray(X), np.asarray(Y))

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
                # yield (np.asarray(X), np.asarray(Y))
                X = np.swapaxes(np.asarray(X), 0, 1)
                yield (list(X), np.asarray(Y))
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
            isSS_i = ~np.asarray(f.get('isSS'))  # inverted because of logic error in file production
            lst = range(len(X_True_i))
            random.shuffle(lst)
            for i in lst:
                isSS = isSS_i[i]
                X_True = X_True_i[i]
                X_EXO = X_EXO_i[i]
                xs_i = f['wfs'][ i ]
                xs_i = np.asarray(np.split(xs_i, 2, axis=1))
                yield (xs_i, X_True, X_EXO, isSS)
            f.close()

def generate_batch_reconstruction(generator, batchSize):
    while 1:
        X, Y, Z, SS = [], [], [], []
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            Z.append(temp[2])
            SS.append(temp[3])
        X = np.swapaxes(np.asarray(X), 0, 1)
        yield (list(X), np.asarray(Y), np.asarray(Z), np.asarray(SS))
        # yield (np.asarray(X), np.asarray(Y), np.asarray(Z), np.asarray(SS))

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
    def def_shared_model_default():
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers.convolutional import Conv2D
        from keras.layers.pooling import MaxPooling2D
        from keras.layers.merge import concatenate
        from keras import regularizers

        regu = regularizers.l2(1.e-2)
        init = "glorot_uniform"
        act = "relu"
        padding = "same"

        # Input layers
        visible_1 = Input(shape=(1024, 38, 1), name='Wire_1')
        visible_2 = Input(shape=(1024, 38, 1), name='Wire_2')

        # Define U-wire shared layers
        shared_conv_1 = Conv2D(16, kernel_size=(5, 3), name='Shared_1', padding=padding, kernel_initializer=init,
                               activation=act, kernel_regularizer=regu)
        shared_pooling_1 = MaxPooling2D(pool_size=(4, 2), name='Shared_2', padding=padding)
        shared_conv_2 = Conv2D(32, kernel_size=(5, 3), name='Shared_3', padding=padding, kernel_initializer=init,
                               activation=act, kernel_regularizer=regu)
        shared_pooling_2 = MaxPooling2D(pool_size=(4, 2), name='Shared_4', padding=padding)
        shared_conv_3 = Conv2D(64, kernel_size=(3, 3), name='Shared_5', padding=padding, kernel_initializer=init,
                               activation=act, kernel_regularizer=regu)
        shared_pooling_3 = MaxPooling2D(pool_size=(2, 2), name='Shared_6', padding=padding)
        shared_conv_4 = Conv2D(128, kernel_size=(3, 3), name='Shared_7', padding=padding, kernel_initializer=init,
                               activation=act, kernel_regularizer=regu)
        shared_pooling_4 = MaxPooling2D(pool_size=(2, 2), name='Shared_8', padding=padding)
        shared_conv_5 = Conv2D(256, kernel_size=(3, 3), name='Shared_9', padding=padding, kernel_initializer=init,
                               activation=act, kernel_regularizer=regu)
        shared_pooling_5 = MaxPooling2D(pool_size=(2, 2), name='Shared_10', padding=padding)
        shared_conv_6 = Conv2D(256, kernel_size=(3, 3), name='Shared_11', padding=padding, kernel_initializer=init,
                               activation=act, kernel_regularizer=regu)
        shared_pooling_6 = MaxPooling2D(pool_size=(2, 2), name='Shared_12', padding=padding)

        # U-wire feature layers
        encoded_1_1 = shared_conv_1(visible_1)
        encoded_1_2 = shared_conv_1(visible_2)
        pooled_1_1 = shared_pooling_1(encoded_1_1)
        pooled_1_2 = shared_pooling_1(encoded_1_2)

        encoded_2_1 = shared_conv_2(pooled_1_1)
        encoded_2_2 = shared_conv_2(pooled_1_2)
        pooled_2_1 = shared_pooling_2(encoded_2_1)
        pooled_2_2 = shared_pooling_2(encoded_2_2)

        encoded_3_1 = shared_conv_3(pooled_2_1)
        encoded_3_2 = shared_conv_3(pooled_2_2)
        pooled_3_1 = shared_pooling_3(encoded_3_1)
        pooled_3_2 = shared_pooling_3(encoded_3_2)

        encoded_4_1 = shared_conv_4(pooled_3_1)
        encoded_4_2 = shared_conv_4(pooled_3_2)
        pooled_4_1 = shared_pooling_4(encoded_4_1)
        pooled_4_2 = shared_pooling_4(encoded_4_2)

        encoded_5_1 = shared_conv_5(pooled_4_1)
        encoded_5_2 = shared_conv_5(pooled_4_2)
        pooled_5_1 = shared_pooling_5(encoded_5_1)
        pooled_5_2 = shared_pooling_5(encoded_5_2)

        encoded_6_1 = shared_conv_6(pooled_5_1)
        encoded_6_2 = shared_conv_6(pooled_5_2)
        pooled_6_1 = shared_pooling_6(encoded_6_1)
        pooled_6_2 = shared_pooling_6(encoded_6_2)

        shared_flat = Flatten(name='flat1')

        # Flatten
        flat_1 = shared_flat(pooled_6_1)
        flat_2 = shared_flat(pooled_6_2)

        # Define shared Dense Layers
        shared_dense_1 = Dense(32, name='Shared_1_Dense', activation=act, kernel_initializer=init,
                               kernel_regularizer=regu)  # 32
        shared_dense_2 = Dense(8, name='Shared_2_Dense', activation=act, kernel_initializer=init,
                               kernel_regularizer=regu)  # 8

        # Dense Layers
        dense_1_1 = shared_dense_1(flat_1)
        dense_1_2 = shared_dense_1(flat_2)

        dense_2_1 = shared_dense_2(dense_1_1)
        dense_2_2 = shared_dense_2(dense_1_2)

        # Merge Dense Layers
        merge_1_2 = concatenate([dense_2_1, dense_2_2], name='Flat_1_and_2')

        # Output
        output = Dense(1, name='Output', activation=act, kernel_initializer=init)(merge_1_2)

        return Model(inputs=[visible_1, visible_2], outputs=[output])

    def def_shared_model():
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers.convolutional import Conv2D
        from keras.layers.pooling import MaxPooling2D
        from keras.layers.merge import concatenate, add
        from keras import regularizers

        regu = regularizers.l2(1.e-2)
        init = "glorot_uniform"
        act = "relu"
        padding = "same"

        # Input layers
        visible_1 = Input(shape=(1024, 38, 1), name='Wire_1')
        visible_2 = Input(shape=(1024, 38, 1), name='Wire_2')

        # Define U-wire shared layers
        shared_conv_1 = Conv2D(16 , kernel_size=(5, 3),   name='Shared_1',  padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_conv_888 = Conv2D(16, kernel_size=(5, 3), name='Shared_888', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_pooling_1 = MaxPooling2D(pool_size=(4, 1), name='Shared_2',  padding=padding) #from (4,2)
        shared_conv_2 = Conv2D(32 , kernel_size=(5, 3),   name='Shared_3',  padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_conv_777 = Conv2D(32, kernel_size=(5, 3), name='Shared_777', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_pooling_2 = MaxPooling2D(pool_size=(4, 2), name='Shared_4',  padding=padding)
        shared_conv_3 = Conv2D(64 , kernel_size=(3, 3),   name='Shared_5',  padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_conv_666 = Conv2D(64, kernel_size=(3, 3), name='Shared_666', padding=padding, kernel_initializer=init,
                               activation=act, kernel_regularizer=regu)
        shared_pooling_3 = MaxPooling2D(pool_size=(2, 2), name='Shared_6',  padding=padding)
        shared_conv_4 = Conv2D(128, kernel_size=(3, 3),   name='Shared_7',  padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_conv_8 = Conv2D(128, kernel_size=(3, 3), name='Shared_999', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_pooling_4 = MaxPooling2D(pool_size=(2, 2), name='Shared_8',  padding=padding)
        shared_conv_5 = Conv2D(256, kernel_size=(3, 3),   name='Shared_9',  padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_pooling_5 = MaxPooling2D(pool_size=(2, 2), name='Shared_10', padding=padding)
        shared_conv_6 = Conv2D(256, kernel_size=(3, 3),   name='Shared_11', padding=padding, kernel_initializer=init, activation=act, kernel_regularizer=regu)
        shared_pooling_6 = MaxPooling2D(pool_size=(2, 2), name='Shared_12', padding=padding)

        # U-wire feature layers
        encoded_1_1 = shared_conv_1(visible_1)
        encoded_1_2 = shared_conv_1(visible_2)
        encoded_888_1 = shared_conv_888(encoded_1_1)
        encoded_888_2 = shared_conv_888(encoded_1_2)
        pooled_1_1 = shared_pooling_1(encoded_888_1)
        pooled_1_2 = shared_pooling_1(encoded_888_2)

        encoded_2_1 = shared_conv_2(pooled_1_1)
        encoded_2_2 = shared_conv_2(pooled_1_2)
        encoded_7_1 = shared_conv_777(encoded_2_1)
        encoded_7_2 = shared_conv_777(encoded_2_2)
        pooled_2_1 = shared_pooling_2(encoded_7_1)
        pooled_2_2 = shared_pooling_2(encoded_7_2)

        encoded_3_1 = shared_conv_3(pooled_2_1)
        encoded_3_2 = shared_conv_3(pooled_2_2)
        encoded_6_1 = shared_conv_666(encoded_3_1)
        encoded_6_2 = shared_conv_666(encoded_3_2)
        pooled_3_1 = shared_pooling_3(encoded_6_1)
        pooled_3_2 = shared_pooling_3(encoded_6_2)

        encoded_4_1 = shared_conv_4(pooled_3_1)
        encoded_4_2 = shared_conv_4(pooled_3_2)
        encoded_8_1 = shared_conv_8(encoded_4_1)
        encoded_8_2 = shared_conv_8(encoded_4_2)
        pooled_4_1 = shared_pooling_4(encoded_8_1)
        pooled_4_2 = shared_pooling_4(encoded_8_2)

        encoded_5_1 = shared_conv_5(pooled_4_1)
        encoded_5_2 = shared_conv_5(pooled_4_2)
        pooled_5_1 = shared_pooling_5(encoded_5_1)
        pooled_5_2 = shared_pooling_5(encoded_5_2)

        encoded_6_1 = shared_conv_6(pooled_5_1)
        encoded_6_2 = shared_conv_6(pooled_5_2)
        pooled_6_1 = shared_pooling_6(encoded_6_1)
        pooled_6_2 = shared_pooling_6(encoded_6_2)

        shared_flat = Flatten(name='flat1')

        # Flatten
        flat_1 = shared_flat(pooled_6_1)
        flat_2 = shared_flat(pooled_6_2)


        # Define shared Dense Layers
        shared_dense_1 = Dense(32, name='Shared_1_Dense', activation=act, kernel_initializer=init, kernel_regularizer=regu) #32
        shared_dense_2 = Dense(8,  name='Shared_2_Dense', activation=act, kernel_initializer=init, kernel_regularizer=regu) #8

        # Dense Layers
        dense_1_1 = shared_dense_1(flat_1)
        dense_1_2 = shared_dense_1(flat_2)

        dense_2_1 = shared_dense_2(dense_1_1)
        dense_2_2 = shared_dense_2(dense_1_2)

        # Merge Dense Layers
        #merge_1_2 = concatenate([dense_2_1, dense_2_2], name='Flat_1_and_2')
        merge_1_2 = add([dense_2_1, dense_2_2], name='Flat_1_and_2')

        # Output
        output = Dense(1, name='Output', activation=act, kernel_initializer=init)(merge_1_2)

        return Model(inputs=[visible_1, visible_2], outputs=[output])


    def def_model():
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
        from keras.regularizers import l2, l1

        init = "glorot_uniform"
        activation = "relu"
        padding = "same"
        regul = l2(1.e-2)
        model = Sequential()

        # convolution part
        model.add(Conv2D(16, kernel_size=(5, 3), padding=padding, kernel_initializer=init, kernel_regularizer=regul, input_shape=(1024, 76, 1)))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), padding=padding))

        model.add(Conv2D(32, kernel_size=(5, 3), padding=padding, kernel_initializer=init, kernel_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), padding=padding))

        model.add(Conv2D(64, kernel_size=(3, 3), padding=padding, kernel_initializer=init, kernel_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), padding=padding))

        model.add(Conv2D(128, kernel_size=(3, 3), padding=padding, kernel_initializer=init, kernel_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), padding=padding))

        model.add(Conv2D(256, kernel_size=(3, 3), padding=padding, kernel_initializer=init, kernel_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), padding=padding))

        model.add(Conv2D(256, kernel_size=(3, 3), padding=padding, kernel_initializer=init, kernel_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), padding=padding))

        # regression part
        model.add(Flatten())
        model.add(Dense(32, activation=activation, kernel_initializer=init, kernel_regularizer=regul))
        model.add(Dense(8, activation=activation, kernel_initializer=init, kernel_regularizer=regul))
        model.add(Dense(1 , activation=activation, kernel_initializer=init))
        return model

    if not args.resume:
        from keras import optimizers
        print "===================================== new Model =====================================\n"
        # model = def_model()
        model = def_shared_model()
        epoch_start = 0
        # optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)  # normal
        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)  # Test
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

    from keras import backend as K
    print 'Learning Rate:\t', K.get_value(model.optimizer.lr)

    print "\nFirst Epoch:\t", epoch_start
    print model.summary(), "\n"
    print "\n"

    # plot model, install missing packages with conda install if it throws a module error
    from keras import utils as ksu
    try:
        ksu.plot_model(model, to_file=args.folderOUT+'/plot_model.png', show_shapes=True, show_layer_names=True)
    except OSError:
        print 'could not produce plot_model.png ---- run generate_model_plot on CPU\n'
        save_plot_model_script(args.folderOUT)

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

    train_steps_per_epoch = int(sum(numEvents_train.values()) / batchSize)
    validation_steps = int(sum(numEvents_val.values()) / batchSize)
    print 'training steps:\t\t', train_steps_per_epoch
    print 'validation steps:\t', validation_steps

    print 'training los'
    model.fit_generator(
        generate_batch_mixed(gen_train, batchSize, numEvents_train),
        steps_per_epoch=train_steps_per_epoch,
        epochs=args.nb_epoch+epoch_start,
        verbose=2,
        validation_data=generate_batch_mixed(gen_val, batchSize, numEvents_val),
        validation_steps=validation_steps,
        initial_epoch=epoch_start,
        callbacks=[
            callbacks.CSVLogger(args.folderOUT + 'history.csv', append=args.resume),
            callbacks.ModelCheckpoint(args.folderOUT + 'models/weights-{epoch:03d}.hdf5', save_weights_only=True, period=int(args.nb_epoch/100)),
            callbacks.LearningRateScheduler(LRschedule_stepdecay, verbose=1),
            Histories(args, files)
        ])

    print 'training stop'
    model.save(args.folderOUT+"models/model-final.hdf5")
    model.save_weights(args.folderOUT+"models/weights-final.hdf5")

    end = time.time()
    print "\nElapsed time:\t%.2f minutes\tor rather\t%.2f hours\n" % (((end-start)/60.),((end-start)/60./60.))

    print 'Model performance\tloss\t\tmean_abs_err'
    print '\tTrain:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(np.concatenate(files['train'].values()).tolist()), batchSize), steps=100))
    print '\tValid:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(np.concatenate(files['val'].values()).tolist())  , batchSize), steps=100))
    return model

def LRschedule_stepdecay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 20. #10.0
    lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lrate

def save_plot_model_script(folderOUT):
    """
    Function for saving python script for producing model_plot.png
    """
    with open(folderOUT+'generate_model_plot.py', 'w') as f_out:
        f_out.write('#!/usr/bin/env python' + '\n')
        f_out.write('try:' + '\n')
        f_out.write('\timport keras as ks' + '\n')
        f_out.write('except ImportError:' + '\n')
        f_out.write('\tprint "Keras not available. Activate tensorflow_cpu environment"' + '\n')
        f_out.write('\traise SystemExit("=========== Error -- Exiting the script ===========")' + '\n')
        f_out.write('model = ks.models.load_model("%smodels/model-initial.hdf5")'%(folderOUT) + '\n')
        f_out.write('try:' + '\n')
        f_out.write('\tks.utils.plot_model(model, to_file="%s/plot_model.png", show_shapes=True, show_layer_names=True)'%(folderOUT) + '\n')
        f_out.write('except OSError:' + '\n')
        f_out.write('\tprint "could not produce plot_model.png ---- try on CPU"' + '\n')
        f_out.write('\traise SystemExit("=========== Error -- Exiting the script ===========")' + '\n')
        f_out.write('print "=========== Generating Plot Finished ==========="' + '\n')
        f_out.write('\n')

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
