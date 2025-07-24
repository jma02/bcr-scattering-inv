import h5py

print('Reading data...')
fin = h5py.File("data/merged_data.hdf5", 'r')
InputArray = fin['farfield.real'][:]

Nsamples = InputArray.shape[1]
InputArray = np.array(InputArray).T.reshape(Nsamples, 100, 100)
OutputArray = fin['image'][:]
OutputArray = np.array(OutputArray).T.reshape(Nsamples, 100, 100)
print('done')

