import numpy as np
from joblib import Parallel, delayed
import h5py

Nprocs = 8
prefix = 'cors/5phot/neutral_open-downsamp-5phot-50mol-q0.047-q0.367'
# =- =========

# list of cor files to look at
num_shots = int(4.5e6)
num_shots_per_file = int(3e5)
num_files = int(num_shots/num_shots_per_file)
max_num = num_shots_per_file * num_files

filelist = ['%s-%d-cors.hdf5'%(prefix,nn) for nn in range(num_files)] 
#filelist = ['infPhot_100mol/2rh1A_3p0gA_infPhot_100mol_a0.40-0-cors.hdf5']
print filelist

# q values
#ff = h5py.File("1igt_nolig.pdb_10000_111_downsamp.ring",'r')
#all_qs = ff['q_values2'][3:13]# this is the low q region
#ff.close()

# gather correlators
#qmin = 0.7
#qmax = 2.5
#q_ind_min = np.argmin(np.abs(all_qs-qq))

#qq = all_qs[q_ind]

min_shots = int(1e5)
max_shots = int(4e6)

d_shots = 30

#inds = np.logspace( np.log10(min_shots), np.log10(max_shots), d_shots).astype(int)
inds = np.linspace( min_shots, max_shots, d_shots).astype(int)
inds1 = inds[ np.where( inds <= (max_num) )[0] ]


###########
#

def draw(filelist,inds):
    print("process is drawing shots...")
    aves = []
    np.random.seed()

    h5_handles = [h5py.File(path, 'r') for path in filelist]
    #shuffle the h5_handles
    np.random.shuffle(h5_handles)
    num_files = len(h5_handles)
    
    
    max_ind = int(h5_handles[0]['cor'].shape[2]/2)
    min_ind = 0
    num_shots_per_file=h5_handles[0]['cor'].shape[0]

    for ind in inds:
        print("Drawing %d shots..."%ind)
        num_files_needed = int(ind/num_shots_per_file)
        remainder = ind%num_shots_per_file

        c_ave = np.zeros( (h5_handles[0]['cor'].shape[1],h5_handles[0]['cor'].shape[2] ) )
        ind_max = h5_handles[0]['cor'].shape[0]
       
        for kk in range(num_files_needed):
            h5 = h5_handles[kk]
            c_ave += (h5['cor'].value).mean(0) * num_shots_per_file/float(ind)
        if remainder>0:
            c_ave += (h5_handles[num_files_needed]['cor'].value[:remainder]).mean(0) * remainder/float(ind)

       
############
        #print c_ave.shape
        c_ave = c_ave[min_ind:max_ind]
        #c_ave -= c_ave.min()
        #c_ave /= c_ave.max()
        ###############
        aves.append( c_ave)
    
    for h5 in h5_handles:
        h5.close()

    return  np.array( aves)

# = ==================


draw_iters = 1
print("Drawing shots")
results1 = []
for _ in xrange( draw_iters):
    results1.extend( [ Parallel(n_jobs=Nprocs,max_nbytes=1e6)(delayed(draw)(filelist,inds1) 
        for _ in xrange(Nprocs))][0] )
print("finished drawing")
r1 = np.array( results1) 
print "r1 shape:"
print r1.shape

fsave = h5py.File('%s-conv.hdf5'%(prefix),'w')
fsave.create_dataset('num_shots',data=inds1)
fsave.create_dataset('cors',data=r1)
fsave.close()

# = = ====







