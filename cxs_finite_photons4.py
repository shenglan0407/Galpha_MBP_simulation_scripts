import h5py
import os
import numpy as np
from loki.RingData import DiffCorr
from joblib import Parallel, delayed
import time

def norm_cor( minQ, minPhi,maxPhi, dat ):
    dat_ = dat[minQ:,minPhi:maxPhi]
    dat_ -= dat_.min()
    dat_ /= dat_.max()
    return dat_

def main():
    N = 15
    pref = 'finite_shots/neutral_open-downsamp-5phot-50mol-q0.047-q0.367'
    init_N = 0

    fnames = ['%s-%d.hdf5'%(pref,x) for x in range(init_N,N+init_N)]

    save_name = '%s-cxs'%pref

    all_cors = []

    for fname in fnames:
        assert( os.path.exists(fname))
        print(fname)
        h5 = h5py.File(fname, 'r')
        shots = np.concatenate( [h5[k].value for k in h5 ] )
        h5.close()
        print("closing file...")
        shots = shots[:len(shots)/2] - shots[len(shots)/2:]
        shots = np.array_split( shots,N,axis=0)
        
        results = [ Parallel(n_jobs=4)(delayed(corr)(s, proc_i) 
            for proc_i, s in enumerate(shots))][0]

        cors = np.concatenate( results )
        
        print cors.shape
        print cors.dtype
        time.sleep(2)
         
        output_fname = fname.replace('.hdf5', '-cors.hdf5')
        print("opening file to write...")
        output_f = h5py.File(output_fname, 'w')
        print("writing...")
        output_f.create_dataset('cor',data=cors)
        print("closing...")
        output_f.close()

        all_cors.append( cors.mean(0) )

        del cors
        del shots

    cxs = np.mean( all_cors,0)
    np.save(save_name, cxs)
    print("Done.")

def corr(diffs, proc_i):
    print ('\t%d: Correlating...'%proc_i)
    dc = DiffCorr(diffs)
    cor = dc.autocorr()
    print("finished correlating...")
    return cor

if __name__ == '__main__':
    main()


