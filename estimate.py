import numpy as np
import h5py
import sys

ring_f = sys.argv[1]
cor_f = sys.argv[2] # where the q values are stored

f = h5py.File(ring_f, 'r')
f2 = h5py.File(cor_f,'r')

# get the q magnitudes and wavlength
print ('Loading q values and wavlength...')
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
qmin = 0.02
ind1 = np.argmin( np.abs( qvals -qmin))
print "index of minimum q value: %d, q value: %.3f"% (ind1, qvals[ind1])

qmax = 0.40
ind2= np.argmin( np.abs( qvals -qmax))
print "index of maximum q value: %d, q value: %.3f"% (ind2, qvals[ind2])

# load the intensities
print("Loading polar intensities...")
tags = f.keys()
img_shape = f[tags[0]][ind1:ind2].shape
Sq_omega = np.zeros( (len(tags), img_shape[0], img_shape[1]), dtype = np.float32)
for idx, tt in enumerate(tags):
    Sq_omega[idx] = (f[tt].value[ind1:ind2])
print "mean Sq: %.3f"% ( Sq_omega.mean())
print "Sq shape:" 
print Sq_omega.shape

# beam parameters
phot_per_pulse = 2e12

focal_diam = 1.0 #um

focal_area = 1e-8 * np.pi * (focal_diam / 2)**2  # cm^2
#focal_area = 0.055 * 0.03 * 1e-8

J = phot_per_pulse / focal_area # flux (per cm^2)

r_e = 2.82e-13 # cm

print ("flux: %e"%J)


thetas = np.arcsin(qvals[ind1:ind2] * wave / 4 / np.pi)
del_phi = 2 * np.pi / Sq_omega.shape[-1]
del_th = np.mean(-1*(thetas[:-1] - thetas[1:])/2)
print "del_phi: %g, del_th: %g"% ( del_phi, del_th )
nbars = ( Sq_omega[:,:].sum(-1) * 4* np.sin(thetas[None,:]) * np.cos(thetas[None,:]) ).sum(-1) * J * r_e * r_e * del_th * del_phi 


print ("photons per molecule: %.4f"%nbars.mean())



def pol2cart( pol_img,
    det_dist=0.1, 
    pixsize=0.0015, 
    phi_values = np.arange(360) * 2 * np.pi / 360.,
    q_values = np.arange( 2.1, 3.5, 0.03) ,
    wave=1.23948):
    """
    pol_img     - a polar image with shape pol_im.shape -> (Nq, Nphi)
    det_dist    - sample-to-detector distance in meters
    pixsize     - square pixel size in meters
    wave        - photon wavelength in angstrom
    phi_values  - a list of phis corresponding to the 2nd axis
                of the polar image (units radians from 0 - 6.28)
    q_values    - a list of scattering vector magnitudes 
                corresponding to the polar image 1st axis
                units are inverse angstroms
                    
    """


    Nphi = phi_values.shape[0]
    Nq = q_values.shape[0]
    
    assert( pol_img.shape == (Nq, Nphi ) )
    
    th_values = np.arcsin( q_values * wave / 4 / np.pi)
    r = np.tan( 2* th_values ) *det_dist
    PHIS = np.array([phi_values]*Nq)
    THETAS = np.array([th_values]*Nphi).T
    R = np.array([r]*Nphi).T

    Rp = R / pixsize
    a,b = Rp.max() + 10, Rp.max()+10

    x = (Rp*np.cos(PHIS) + a).astype(int)
    y = (Rp*np.sin(PHIS) + b).astype(int)

    cart_img = np.zeros( ( x.max()+10, y.max()+10) )
    for x,y,i in  izip(x.ravel(), y.ravel(), pol_img.ravel()):
        cart_img[x,y] = i
    return cart_img

#Nphi = img.shape[-1]
#phis = arange( Nphi) * 2 * pi / Nphi
#qs = qvals[ ind:]
#cimg =  pol2cart( pimg, 1, 0.009, phi_values=phis, q_values=qs )


