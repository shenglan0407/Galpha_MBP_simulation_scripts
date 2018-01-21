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

from scipy.interpolate import interp1d

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
big_img_sh, qvecs =  thorn.make_q_vectors( qmin=.0, qmax=0.92, 
    dq=0.005, dphi=0.015, wavelen=1., pow2='next' )

qvals = qvecs.reshape( (big_img_sh[0],big_img_sh[1],3) )
qvals = np.sqrt((qvals**2).sum(-1)).mean(1)
wave=1.0


with  h5py.File('/shiner/work/qiaoshen/g_alpha_MBP/models/1gp2_MBP_wlig/open_domains_confs/all_dif_cor.hdf5', 'a') as masteroutput:
    if not 'qvalues' in masteroutput.keys():
        masteroutput.create_dataset( 'qvalues',data = qvals )
        masteroutput.create_dataset( 'wavlen_in_angstrom', data=wave)

    pdbs = [ os.path.join( '/shiner/work/qiaoshen/g_alpha_MBP/models/1gp2_MBP_wlig/open_domains_confs', f) 
        for f in os.listdir('/shiner/work/qiaoshen/g_alpha_MBP/models/1gp2_MBP_wlig/open_domains_confs') \
        if f.endswith('pdb') and f.startswith('conf') ]

    for pdb in pdbs:

        pdb_base = os.path.basename(pdb).split('.pdb')[0]
        if pdb_base in masteroutput.keys():
            print('already simulated shots for %s... Skip!'%pdb_base)
            continue
        print ("loading %s"%pdb)
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
                output.create_dataset('shot%d'%i, data=I.reshape( big_img_sh))


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
        ##############################################################################
        # now do the downsampling
        # compute q intervals needed from data
        waxs_rmin = 100 # pixel units
        waxs_rmax = 450
        c = 2.998e8
        h = 6.63e-34
        ev_to_j = 1.602e-19
        num_q = 35
        interval = int( (waxs_rmax-waxs_rmin)/num_q)

        pixsize = 110e-6 #meter
        photon_energy = 9.5e3 #eV
        wavlen = c * h/(photon_energy * ev_to_j) * 1e10 # angstrom
        det_dist = 260e-3 # meter

        thetas = np.arctan ( np.arange(waxs_rmin, waxs_rmax) * pixsize / det_dist)  /2.    
        qs =np.sin( thetas ) * 4.*np.pi/wavlen
        q_intervals = np.zeros( (num_q+1,2) )
        for idx in range(1, num_q+1):
            
            try:
                q_intervals[idx] = np.array([qs[(idx-1)*interval],qs[idx*interval] ])
            except IndexError:
                q_intervals[idx] = np.array([qs[(idx-1)*interval],qs[-1] ])

        d = q_intervals[1,1]-q_intervals[1,0]
        q_intervals[0] = np.array([q_intervals[1,0]-d,q_intervals[1,0]] ) 
        # print q_intervals

        corrs = []
        print ( "downsampling for %s"%out)
        f = h5py.File(out,'r')
        tags = f.keys()

        num_phi = f[tags[0]].shape[-1]
        new_num_phi = 354

        phis = np.linspace(0,np.pi*2,num_phi)
        new_phis = np.linspace(0,np.pi*2, new_num_phi)

        output_file = out.replace('_shots.hdf5',
            '_downsamp_q%.2f_%.2f_Nq%d.hdf5'%(q_intervals.min(),q_intervals.max(),num_q))

        all_downsamp_shots  = np.zeros( (len(tags), q_intervals.shape[0], new_num_phi)
            , dtype = np.float64)

        for qidx, qint in enumerate(q_intervals):
            # qmax, qmin
            if qint[0] < qvals.min() or qint[1] > qvals.max():
                continue
                 
            qmin = qint[0]
            ind1 = np.argmin( np.abs( qvals -qmin))
            print "index of minimum q value: %d, q value: %.3f"% (ind1, qvals[ind1])

            qmax = qint[1]
            ind2= np.argmin( np.abs( qvals -qmax))
            print "index of maximum q value: %d, q value: %.3f"% (ind2, qvals[ind2])
            
            q1 = qvals[ind1:ind2]
            print("total number of qs to downsize: %d"% ( len(q1) ) )
            num_q = 1
            q_fac =len(q1)
            phi_fac = 1 # factor by which to donwsample phi
            
            img_sh = f[tags[0]].value[ind1:ind2,:].shape
            
            assert(img_sh[1]%phi_fac==0)
            
            new_img_sh = ( 1, int(img_sh[1]/phi_fac) )
            
            #load all the shots and downsample them
            pi_down = np.zeros( (len(tags), new_img_sh[0], new_img_sh[1] ) )
            for idx, tt in enumerate(tags):
                img = f[tt].value[ind1:ind2,:]
                img = img.reshape( (new_img_sh[0], q_fac, new_img_sh[1], phi_fac) )
                img_down = img.sum(axis=1).sum(axis=2)
                pi_down[idx] = img_down
            
            # downsample qs
            q_down = q1.reshape((num_q,q_fac)).mean(-1)
            
            
            dc = DiffCorr(pi_down, pre_dif = False)
            ac = dc.autocorr().mean(0)
            corrs.append(ac)

            # print phis.shape, pi_down.shape

            interp_pi = interp1d(phis, pi_down, kind='cubic')
            all_downsamp_shots[:,qidx, :] = interp_pi(new_phis)[:,0,:]


            del pi_down

       # save
        corrs = np.array(corrs)
        num_q = corrs.shape[0]
        
        f_save = h5py.File(output_file,'w')
        f_save.create_dataset('wavlen_in_angstrom',data=wave)
        f_save.create_dataset('q_intervals',data=q_intervals)
        f_save.create_dataset('downsamp_shots', data = all_downsamp_shots)
        f_save.create_dataset('cors',data=corrs)
        f_save.close()
        print("removing shots file: %s"%out)
        os.remove(out)




