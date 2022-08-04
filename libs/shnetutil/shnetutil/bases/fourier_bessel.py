import numpy as np
from scipy import special
import math
import pkg_resources		#TODO: replace with importlib.resources


def load_bessel(besselfile='./bessel.npy'):
	#print(f"load_bessel.__name__={__name__}")	#__name__ = shnetutil.bases.fourier_bessel
	stream = pkg_resources.resource_stream(__name__, besselfile)
	bessel = np.load(stream)
	return bessel

def cart2pol(x, y):
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	return (phi, rho)

def calculate_FB_bases(L1, Ntheta, K):
	"""
	compute discrete FB bases on 3x3 and 5x5 patches
	`
	Args:
	 	L1: L1 = 1 (3x3) or 2 (5x5)
		Ntheta: number of rotations(discretization of [0,2pi])
		K: num of bases
	Return:
		psi, c, kq_Psi, fts(rotation matrices)
	"""
	# the max index of the 1-axis reshaped bases
	maxK = (2 * L1 + 1)**2 - 1
	L = L1 + 1
	R = L1 + 0.5
	truncate_freq_factor = 1.5

	if L1 < 2:
		truncate_freq_factor = 2

	xx, yy = np.meshgrid(range(-L, L+1), range(-L, L+1))
	xx = xx/R
	yy = yy/R

	ugrid = np.concatenate([yy.reshape(-1,1), xx.reshape(-1,1)], 1)
	tgrid, rgrid = cart2pol(ugrid[:,0], ugrid[:,1])
	num_grid_points = ugrid.shape[0]
	kmax = 15

	# with shape [k,q, R_kq, R_{k,q+1}]
	bessel = load_bessel('./bessel.npy')
	B = bessel[(bessel[:,0] <=kmax) & (bessel[:,3]<= np.pi*R*truncate_freq_factor)]
	idxB = np.argsort(B[:,2])
	mu_ns = B[idxB, 2]**2

	ang_freqs = B[idxB, 0]
	rad_freqs = B[idxB, 1]
	R_ns = B[idxB, 2]

	num_kq_all = len(ang_freqs)
	max_ang_freqs = max(ang_freqs)
	Phi_ns=np.zeros((num_grid_points, num_kq_all), np.float32)
	Psi = []
	kq_Psi = []
	num_bases=0
	for i in range(B.shape[0]):
		ki = ang_freqs[i]
		qi = rad_freqs[i]
		rkqi = R_ns[i]
		r0grid=rgrid*R_ns[i]
		F = special.jv(ki, r0grid)
		Phi = 1./np.abs(special.jv(ki+1, R_ns[i]))*F
		Phi[rgrid >=1]=0
		Phi_ns[:, i] = Phi

		if ki == 0:
			Psi.append(Phi)
			kq_Psi.append([ki,qi,rkqi])
			num_bases = num_bases+1

		else:
			Psi.append(Phi*np.cos(ki*tgrid)*np.sqrt(2))
			Psi.append(Phi*np.sin(ki*tgrid)*np.sqrt(2))
			kq_Psi.append([ki,qi,rkqi])
			kq_Psi.append([ki,qi,rkqi])
			num_bases = num_bases+2

	Psi = np.array(Psi)
	kq_Psi = np.array(kq_Psi)

	num_bases = Psi.shape[0]

	if num_bases > maxK:
		Psi = Psi[:maxK]
		kq_Psi = kq_Psi[:maxK]
	num_bases = Psi.shape[0]
	p = Psi.reshape(num_bases, 2*L+1, 2*L+1).transpose(1,2,0)
	psi = p[1:-1, 1:-1, :]
	# with shape [len_bases, num_bases]
	psi = psi.reshape((2*L1+1)**2, num_bases)
	kq_Psi = kq_Psi.transpose(1,0)

	# normalize by the sqrt of mean square, c with shape [num_bases]
	c = np.sqrt(np.sum(psi**2, 0).mean())
	psi = psi/c

	# fetch first-K bases
	psi = psi[:, :K]
	kq_Psi = kq_Psi[:, :K]

	## calculate rotation matrices for spatial bases
	maxK = psi.shape[1]
	fts = []
	for it in range(Ntheta):
		k = 0
		ft = np.zeros((maxK, maxK))
		
		while k+1 <= K:
			m = kq_Psi[0,k]
			if m == 0:
				ft[k, k] = 1
				k += 1
			else:
				c = np.cos(it/Ntheta * 2*math.pi * m)
				s = np.sin(it/Ntheta * 2*math.pi * m)
				if k+2 > K:
					ft[k, k] = c
				else:
					ft[k:k+2, k:k+2] = np.array([[c,-s], [s,c]])
				k += 2
		
		fts.append(ft)

	return psi, c, kq_Psi, fts

def initialize_spatial_bases_FB(kernel_size, Ntheta, num_bases, verbose=False):
	if kernel_size % 2 == 0:
			raise Exception('Kernel size for FB initialization only supports odd number for now.')
	base_np, _, _, fts = calculate_FB_bases(int((kernel_size-1)/2), Ntheta, num_bases)
	
	base_np = base_np.reshape(kernel_size, kernel_size, num_bases)
	if verbose:
		print("finish generation fb bases, bases is:")
		print(np.around(base_np, decimals=2))

	return base_np, fts

# def initialize_temporal_bases_DCT(mode, length, num_bases, verbose=True):
#     # with shape [num_bases, kernel_size]
#     cvt_mtx = np.zeros([num_bases, length])

#     # fill the first base with sqrt(1/length)
#     for i in range(length):
#         cvt_mtx[0, i] = math.sqrt(1 / length)

#     # fill the other bases according to formula
#     for j in range(1, num_bases):
#         for i in range(length):
#             cvt_mtx[j, i] = math.sqrt(2 / length) * math.cos(((i + 0.5) * j * math.pi) / length)

#     # save the matrix as npy file
#     if verbose:
#         print("finish generation dct bases, bases is:")
#         print(np.around(cvt_mtx, decimals=4))
    
#     return cvt_mtx

def initialize_rotation_bases(Ntheta, Kalpha, verbose=True):
	maxL = int(np.ceil(Ntheta/2))

	num_bases = 0
	phi = []
	l_Phi = []

	tgrid = np.arange(Ntheta)/Ntheta * 2*math.pi

	for l in range(maxL+1):
		if l == 0:
			phi.append(np.ones(Ntheta))
			l_Phi.append(l)
			num_bases += 1
		else:
			phi.append(np.cos(l*tgrid) * np.sqrt(2))
			l_Phi.append(l)
			phi.append(np.sin(l*tgrid) * np.sqrt(2))
			l_Phi.append(l)
			num_bases += 2

	## rotation bases with shape [num_bases, Ntheta]
	phi = np.stack(phi, axis=0)
	l_Phi = np.array(l_Phi)

	## create 'rotation matrices' for rotation bases
	maxKalpha = phi.shape[1]
	gts = []

	for it in range(Ntheta):
		gt = np.zeros((maxKalpha, maxKalpha))
		k = 0
		
		while k+1 <= Kalpha:
			l = l_Phi[k]
			if l == 0:
				gt[k, k] = 1
				k += 1
			else:
				c = np.cos(it/Ntheta * 2*math.pi * l)
				s = np.sin(it/Ntheta * 2*math.pi * l)
				if k+2 > Kalpha:
					gt[k, k] = c
				else:
					gt[k:k+2, k:k+2] = np.array([[c,-s], [s,c]])
				k += 2
		
		gts.append(gt[:Kalpha, :Kalpha])
	
	## select bases
	phi = phi[:Kalpha]
	l_Phi = l_Phi[:Kalpha]

	return phi, l_Phi, gts


def initialize_bases(mode, kernel_size_s, Ntheta, K, K_a, verbose=True):
	"""
        initialize both spatial bases and rotation bases
        
        args:
            mode: random or use pre-computed bases
            kernel_size: ( num_rotations(Ntheta), spa_size )
            num_bases: ( num_rot_bases(Kalpha), num_spatial_bases(K) )

        return:
            s_bases_np: spatial bases in ndarray, with shape 
                        (s_kernel_size, s_kernel_size, s_num_bases)
			fts: list of rotation matrices for spatial bases, each with shape [K, K]
            
			r_bases_np: rotation bases in ndarray, with shape (t_num_bases, t_kernel_size)
			gts: list of rotation matrices for rotation bases, each with shape [Kalpha, Kalpha]
	"""
	if mode == 'FB_FOUR':
		s_bases_np, fts = initialize_spatial_bases_FB(kernel_size_s, Ntheta, K, verbose=verbose)
		r_bases_np, _, gts = initialize_rotation_bases(Ntheta, K_a, verbose=verbose)

	if verbose:
		print("finish generation fourier bases, bases is:")
		print(np.around(r_bases_np, decimals=4))

	return s_bases_np, fts, r_bases_np, gts



if __name__ == '__main__':
	initialize_bases('FB_FOUR', kernel_size=(16,5), num_bases=(9, 8))
