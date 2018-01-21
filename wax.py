import h5py
import numpy as np
models = ['compact','compact_open','neutral','neutral_open','extended','extended_open']

for m in models:
    f = h5py.File('%s_shots.hdf5'%m,'r')
    tags= f.keys()
    img = np.zeros_like(f[tags[0]])
    for tt in tags:img+=f[tt].value/len(tags) 
    wax = img.mean(-1)
    wax.shape
    
    np.save('%s_wax.npy'%m,wax)

