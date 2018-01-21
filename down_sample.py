import numpy as np
import h5py

models = ['compact','compact_open','neutral','neutral_open','extended','extended_open']

# load qvalues
f2 = h5py.File('cors/all_dif_cor.hdf5','r')

qx = f2['qvalues'][:,:,0]
qy = f2['qvalues'][:,:,1]
qz = f2['qvalues'][:,:,2]
qmag = np.sqrt(qx**2+qy**2+qz**2)
qvals = qmag.mean(-1)

# wavlength in angstrom
wave = f2['wavlen_in_angstrom'].value
print("wavlength is %.3f"%wave)
f2.close()

# qmax
qmin = 0.21
ind1 = np.argmin( np.abs( qvals -qmin))
print "index of minimum q value: %d, q value: %.3f"% (ind1, qvals[ind1])

qmax = 0.82
ind2= np.argmin( np.abs( qvals -qmax))
print "index of maximum q value: %d, q value: %.3f"% (ind2, qvals[ind2])

q1 = qvals[ind1:ind2]
print("total number of qs to downsize: %d"% ( len(q1) ) )

q_fac =3 # factor by which to down sample q
phi_fac = 2 # factor by which to donwsample phi

num_q = int(q1.size/q_fac)
# down sample for each model
for name in models:
    input_file = 'shots/%s_shots.hdf5'%name
    print ( "downsampling for %s"%input_file)
    f = h5py.File(input_file,'r')

    tags = f.keys()
    img_sh = f[tags[0]].value[ind1:ind2,:].shape

    assert(img_sh[0]%q_fac==0)
    assert(img_sh[1]%phi_fac==0)

    new_img_sh = ( int(img_sh[0]/q_fac), int(img_sh[1]/phi_fac) )
    
    #load all the shots and downsample them
    pi_down = np.zeros( (len(tags), new_img_sh[0], new_img_sh[1] ) )
    for idx, tt in enumerate(tags):
        img = f[tt].value[ind1:ind2,:]
        img = img.reshape( (new_img_sh[0], q_fac, new_img_sh[1], phi_fac) )
        img_down = img.sum(axis=1).sum(axis=2)
        pi_down[idx] = img_down

    # downsample qs
    q_down = q1.reshape((num_q,q_fac)).mean(-1)

    # save
    output_file = input_file.replace('_shots.hdf5','downsamp_35qs_shots.hdf5')
    f_save = h5py.File(output_file,'w')
    f_save.create_dataset('wavlen_in_angstrom',data=wave)
    f_save.create_dataset('q_values',data=q1)
    f_save.create_dataset('q_values2',data=q_down)
    f_save.create_dataset('polar_intensities',data=pi_down)
    f_save.close()

