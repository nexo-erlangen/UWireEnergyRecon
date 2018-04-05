#!/usr/bin/env python

import numpy as np
import h5py
import argparse

import os
from os import listdir
from os.path import isfile,join

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-out', dest='folderOUT', help='folderOUT Path')
parser.add_argument('-in' , dest='folderIN' , help='folderIN Path')
#parser.add_argument('-file' , dest='filename' , help='folderIN Path')

args = parser.parse_args()
args.folderIN=os.path.join(args.folderIN,'')
args.folderOUT=os.path.join(args.folderOUT,'')

files = [f for f in listdir(args.folderIN) if isfile(join(args.folderIN, f))]

start, length = 512, 1024
slice = 1000
counter = 0
for index, filename in enumerate(files):
	print "reading:\t", args.folderIN + filename
	fIN = h5py.File(args.folderIN + str(filename), "r")
	recon_energy_i = np.array(fIN.get('reconEnergy'))
	light_energy_i = np.array(fIN.get('lightEnergy'))
	gains = np.array(fIN.get('gains'))
	wfs_i = np.array(fIN.get('wfs'))[:,:,start:start + length]
	fIN.close()

	wfs_i = np.array(wfs_i / gains[:, None])
	wfs_i = np.swapaxes(wfs_i, 1, 2)
	wfs_i = wfs_i[..., np.newaxis]
	if index==0:
		wfs=wfs_i
		recon_energy = recon_energy_i
		light_energy = light_energy_i
	else:
		recon_energy = np.append(recon_energy, recon_energy_i)
		light_energy = np.append(light_energy, light_energy_i)
		wfs = np.concatenate((wfs, wfs_i))

	while len(recon_energy)>slice:
		starting, ending = filename.split(".")
		print "creating:\t", args.folderOUT + str(counter) + ".hdf5"

		fOUT = h5py.File(args.folderOUT + str(counter) + ".hdf5", "w")
		counter += 1
		dset1 = fOUT.create_dataset("wfs", data=wfs[:slice], dtype=np.float32)
		dset2 = fOUT.create_dataset("reconEnergy", data=recon_energy[:slice], dtype=np.float32)
		dset3 = fOUT.create_dataset("lightEnergy", data=light_energy[:slice], dtype=np.float32)
		fOUT.close()
		wfs = wfs[slice:]
		light_energy = light_energy[slice:]
		recon_energy = recon_energy[slice:]