import h5py
import matplotlib.pyplot as plt

folder="/home/vault/capm/sn0515/PhD/Th_U-Wire/"
file = h5py.File("/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/ED_SourceS5_Th228_0.hdf5")


print "Print items in the file", file.items()
data = file.get('wfs')
TrueEnergy = file.get('trueEnergy')
ReconEnergy = file.get('reconEnergy')
gains = file.get('gains')

plt.ion()

print data.shape
print gains

for i in xrange(data.shape[0]):
    if(i==10) :
        break

    ax = plt.gca()
    plt.imshow(data[i])
    ax.set_aspect(1./ax.get_data_ratio())
    plt.colorbar()
    plt.savefig(folder+str(i)+'.png')
    plt.close()

    print i
    print TrueEnergy[i], ReconEnergy[i]
    #raw_input()
    plt.clf()

















