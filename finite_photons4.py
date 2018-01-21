#!/usr/bin/python -Wall

import time
from collections import Counter

import numpy as np
import h5py

from joblib import Parallel, delayed


class ShotMaker:
    def __init__(self,ring_fname, minq_ind, maxq_ind=None):
        ifile = h5py.File(ring_fname, "r")
        print("Loading ring data...")
        if maxq_ind==None:
            self.inf_shots = ifile['polar_intensities'][:,minq_ind:]
        else:
            self.inf_shots = ifile['polar_intensities'][:,minq_ind:maxq_ind]
        self.total_omega = self.inf_shots.shape[0]
        self.img_sh = self.inf_shots[0].shape
        self.Npix = self.img_sh[0] * self.img_sh[1]

        scatt_s = self.inf_shots.sum(2).sum(1)
        self.inf_shots /= scatt_s[:,None,None]
        self.inf_shots = self.inf_shots.ravel() 

   
    def make_shot(self, Nmol, nbar ): 
        shot = np.zeros(self.img_sh, dtype=np.float32)
        
        photons = np.random.poisson( nbar, Nmol)
        
        #np.random.seed()
        for n in xrange(1, photons.max() ):
            n_omega = np.sum( photons == n)
            if n_omega ==0:
                continue
            #print("\tSampling %d,  %d photon molecules"%(n_omega,n))
            rnd_omega = np.random.randint( 0, self.total_omega, n_omega)
            omega_count = Counter( rnd_omega)

            shot_n = np.zeros( self.Npix, dtype=np.float32)
            for o,count in omega_count.iteritems():
                o_shot = self.inf_shots[o*self.Npix:o*self.Npix+self.Npix]
                shot_n += np.random.multinomial(count*n, o_shot)

            shot += shot_n.reshape( self.img_sh)

        return shot

def main():
    input_fname = 'shots/neutral_opendownsamp_shots.hdf5'
    init_num = 0
    Nshots = 9000000
    Nprocs=15
    min_q_ind = 0
    max_q_ind = 6 # maxmimum index not inclusive
    
    f = h5py.File(input_fname,'r')
    qs = f['q_values2'].value
    f.close()

    Nphot_per_mol = 5 # mean number of photons
    Nmol_per = 50 # number of mols per shot (SNR should be independent)
    output_prefix = 'neutral_open-downsamp-%dphot-%dmol-q%.3f-q%.3f'%( Nphot_per_mol, Nmol_per
        ,qs[min_q_ind], qs[ (max_q_ind-1)] )

    
    S = ShotMaker( input_fname, 
    minq_ind= min_q_ind,
    maxq_ind = max_q_ind)
    
    output_fnames = ['%s-%d.hdf5'%(output_prefix,i+init_num) 
        for i in xrange( Nprocs) ]
    print output_fnames
    args=[(S, int(Nshots/Nprocs), o, Nmol_per, Nphot_per_mol, proc_i) 
        for proc_i,o in enumerate(output_fnames)]
   
    results = [ Parallel(n_jobs=Nprocs)(delayed(make_some_shots)(*arg) 
        for arg in args)]

    print("Done.")
    

def make_some_shots(ShotMaker, Nshots, output_fname, 
    Nmol_per, Nphot_per_mol, proc_i,block_sz=100000):
    
    np.random.seed( ) # IMPORTANT!
    
    ofile = h5py.File( output_fname, 'w')
   
    if Nshots < block_sz:
        shot_inds = np.array([np.arange( Nshots)])
    else:
        shot_inds = np.array_split( np.arange(Nshots), int(Nshots / block_sz))

    for i_block, inds in enumerate(shot_inds):
        t = time.time()
        print ("%d: Block %d/%d"%( proc_i, i_block+1, len(shot_inds)))

        shots = np.zeros( ( len(inds), ShotMaker.img_sh[0], ShotMaker.img_sh[1] ) )
        for i_shot in xrange( len(inds)):
            shots[i_shot] = ShotMaker.make_shot(Nmol_per, Nphot_per_mol) 
            
        print("Finished block...")
        ofile.create_dataset('shot_block-%d' % i_block, data=shots)
        t_iter = (time.time() - t)*1
        print ( "Time per %d shots: %.4f sec"%(len(inds),t_iter))

    return ofile.close()

if __name__ == '__main__':
    main()




