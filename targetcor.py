"""
make single mol correlations fast: author: donians.mender
damende3@asu.edu
"""

import numpy as np
import h5py

from loki.RingData import DiffCorr
from joblib import Parallel, delayed
from bornagain.simulate import thorn

import mdtraj
from bornagain.simulate import clcore
import os

def main(shots, fname, jid):
    
    print(  "\tProc %d"%jid)
    f = h5py.File(fname, 'r')
    shot_data = np.array( [ f[s].value for s in shots ] )
    n = len(shot_data)
    if n%2 != 0:
        shot_data = shot_data[:-1]
        n = n - 1
    
    diffs = shot_data[:n/2] - shot_data[n/2:]
    DC = DiffCorr( diffs, pre_dif=1)
    cors = DC.autocorr()
    s_norm = (shot_data.mean(0).mean(1)[:,None])**2
    return  cors, cors/s_norm

os.environ[ 'PYOPENCL_CTX' ] = '0'

n_jobs = 15
Nshots = 15000
img_sh, qvecs =  thorn.make_q_vectors( qmin=.0, qmax=0.92, 
    dq=0.005, dphi=0.015, wavelen=1., pow2='next' )

qs = qvecs.reshape( (img_sh[0],img_sh[1],3) )
qs = np.sqrt((qs**2).sum(-1)).mean(1)
    


with  h5py.File('/shiner/work/qiaoshen/g_alpha_MBP/models/1gp2_MBP_wlig/frames/all_dif_cor.hdf5', 'a') as masteroutput:
    masteroutput.create_dataset( 'qvalues',data = qs )
    masteroutput.create_dataset( 'wavlen_in_angstrom', data=1.0)
    
    pdbs = [ os.path.join( '/shiner/work/qiaoshen/g_alpha_MBP/models/1gp2_MBP_wlig/frames', f) 
        for f in os.listdir('/shiner/work/qiaoshen/g_alpha_MBP/models/1gp2_MBP_wlig/frames') \
        if f.endswith('pdb') and f.startswith('MBP_1gp2') ]

    for pdb in pdbs:

        pdb_base = os.path.basename(pdb).split('.pdb')[0]
        if pdb_base in masteroutput.keys():
            print('already simulated shots for %d... Skip!'%pdb_base)
            continue

        traj = mdtraj.load_pdb( pdb)
        print("trajectory loaded successfully!")
        atom_vecs = traj.xyz[0]*10.
        atom_Z = np.array( [list(a.element)[0] for a in traj.topology.atoms] ) 

        core = clcore.ClCore()
        core.prime_cromermann_simulator(qvecs, atom_Z)

        q = core.get_q_cromermann()
        r = core.get_r_cromermann(atom_vecs)

        out = pdb.replace('.pdb', '_shots.hdf5')

        with h5py.File(out,'w') as output:
            for i in xrange( Nshots ) :
                print("shot %d/%d"%(i+1, Nshots))
                core.run_cromermann(q,r,rand_rot=True)
                amps = core.release_amplitudes(reset=True)
                I = np.abs( amps )**2.
                output.create_dataset('shot%d'%i, data=I.reshape( img_sh))


        with h5py.File(out,'r') as f:

            shotkeys = [key 
                for key in f.keys() if key.startswith('shot')]

        shots_per_batch = np.array_split(shotkeys, n_jobs)
        
        results = Parallel(n_jobs=n_jobs)(delayed(main)(shots, out, jid)  
            for jid, shots in enumerate(shots_per_batch)) 


        cors = np.concatenate( [ r[0] for r in results ] )
        cors_n = np.concatenate( [ r[1] for r in results ] )


        masteroutput.create_dataset( str( pdb_base) + '/cor', data=cors.mean(0))
        masteroutput.create_dataset( str( pdb_base) + '/cor_normed', data=cors_n.mean(0))



