import numpy as np
import h5py
from loki.RingData import DiffCorr

from scipy.interpolate import interp1d

#models = ['compact','compact_open','neutral','neutral_open','extended','extended_open']
# models = ['compact','compact_open','neutral','neutral_open','extended','extended_open']
models=['MBP_1gp2_nowater_0',
 'MBP_1gp2_nowater_2',
 'MBP_1gp2_water_shell_5A_0',
 'MBP_1gp2_water_shell_5A_2']


# load qvalues
f2 = h5py.File('models/1gp2_MBP_wlig/frames/all_dif_cor.hdf5','r')
qvals=f2['qvalues'].value
# qx = f2['qvalues'][:,:,0]
# qy = f2['qvalues'][:,:,1]
# qz = f2['qvalues'][:,:,2]
# qmag = np.sqrt(qx**2+qy**2+qz**2)
# qvals = qmag.mean(-1)


# wavlength in angstrom
wave = f2['wavlen_in_angstrom'].value
print("wavlength is %.3f"%wave)
f2.close()

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
print q_intervals



for name in models:
    
    corrs = []
    input_file = 'models/1gp2_MBP_wlig/frames/%s_shots.hdf5'%name
    print ( "downsampling for %s"%input_file)
    f = h5py.File(input_file,'r')
    tags = f.keys()

    num_phi = f[tags[0]].shape[-1]
    new_num_phi = 354

    phis = np.linspace(0,np.pi*2,num_phi)
    new_phis = np.linspace(0,np.pi*2, new_num_phi)

    output_file = input_file.replace('_shots.hdf5',
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

