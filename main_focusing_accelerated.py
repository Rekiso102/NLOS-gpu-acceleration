""""
The following code implements the NLOS image reconstruction in python from the following article:
    https://www.nature.com/articles/s41467-020-15157-4

    The code uses the following libraries:
    matplotlib
    numpy
    scipy
    cupy

Author: Nachiket Kerai
- Undergraduate research assistant in the Computational Optics Lab
- Graduating in Fall 2025
"""
import numpy as np
import cupy as cp
from scipy.io import loadmat
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import mat73
import time

# Defined rounding function to replicate MATLAB rounding
def mat_round(x):
    # Check if the number is positive
    y = (x >= 0)*1.0
    x = x +0.5*y - 0.5*(1-y)
    x = np.array(x,np.int32)
    return x

id = 6 # Choose from the 10 avaliable scenese to visualize image reconstruction

#parameters
tag_maxdepth = 2;  #signal of interest respect to the targets, unit meter, FDH capture parameters
depth_min = 0.5;  #depth minimal volumn
depth_max = 1.5;  #depth maximum volumn
v_apt_Sz = 2;  #virtual aperture maximum size, automatical pad to symmetry (square), unit meter, standard size 2
d_offset = 0.1;  #electrical delay as an constant offset ""

match id:
    case 1:
        str_name = 'letter4' 
        tag_maxdepth = 1.5
    case 2:
        str_name = 'resolutionbar'
        tag_maxdepth = 1.5
    case 3:
        str_name = 'NLOSletter'
        tag_maxdepth = 1.5
    case 4:
        str_name = 'shelf_targets_lighton'
        tag_maxdepth = 1.5
    case 5:
        str_name = 'letter44i'
        tag_maxdepth = 1.5
    case 6:
        str_name = 'officescene'  
        tag_maxdepth = 3
        v_apt_Sz = 3
        depth_max = 2.5
    case 7:
        str_name = 'officescene_corrected_1ms'
        tag_maxdepth = 3
        v_apt_Sz = 3; 
        depth_max = 3
        d_offset = 0
    case 8:
        str_name = 'officescene_corrected_5ms'  
        tag_maxdepth = 3
        v_apt_Sz = 3
        depth_max = 3
        d_offset = 0
    case 9:
        str_name = 'officescene_corrected_10ms'
        tag_maxdepth = 3
        v_apt_Sz = 3
        depth_max = 3
        d_offset = 0
    case 10:
        str_name = 'officescene_corrected_20ms'; 
        tag_maxdepth = 3
        v_apt_Sz = 3
        depth_max = 3
        d_offset = 0


# loading data from captured Fourier domain histogram
fileName = './input/' + str_name + '_' + str(tag_maxdepth) + '.mat'
print(fileName)
loadData = mat73.loadmat(fileName)

# Basic parameters for wavefront cube
c_light = 299792458 # speed of light
aperturefullsize = cp.array([loadData['aperturefullsize']])
u_total = cp.array([loadData['u_total']])
u_total = cp.squeeze(u_total)
lambda_loop = cp.squeeze(cp.array([loadData['lambda_loop']]))
omega_space = cp.squeeze(cp.array([loadData['omega_space']]))
weight = cp.squeeze(cp.array([loadData['weight']]))
sample_spacing = aperturefullsize[0, 0]/((u_total.shape[0]) -1) # calculate smaping space on the captured wavefront

# Pad Virtual Aperture Wavefront
# additional if physical dimension is odd number
if (((u_total.shape[1]) % 2 == 1)):
    tmp_0, tmp_x, tmp_y, tmp_z = u_total.shape
    tmp3D = cp.zeros(int(cp.round(tmp_x / 2)*2), int(cp.round(tmp_y / 2)),tmp_z)
    aperturefullsize = (cp.array(tmp3D.shape[:2] - 1)) * sample_spacing
    tmp3D[:u_total.shape[0], :u_total.shape[1], :] = u_total
    u_total = tmp3D

# preallocated memory for padding wavefront
u_tmp = cp.zeros(((2 * int(cp.round(cp.round(v_apt_Sz / sample_spacing) / 2))), (2 * int(cp.round(cp.round(v_apt_Sz / sample_spacing) / 2))), u_total.shape[2]), dtype=cp.int64)

"""
    Parameters input: 
        1. uin: captured virtual wavefront, unitless
        2. apt_in: captured physical aperture sizem, uniter meter (array x-y)
        3. maxSz: virtual aperture physical maximum size, unit meter (real valed scalar)
        4. posL: virtual point source location on the aperture (origional
        physical captured) index [x y] unitless

    Parameters output: 
        1. uout: output virtual wavefront, square size, unitless
        2. apt_out: output virtual aperture physical maximum size, unit meter (array x-y)
        3. posO: output new virtual point source location on the apeture,
        index [x y] unitless

    Description:
        This function works for creating virtual aperture, independent from
        RSD propagation. Input can be symmetry or not, output dimension
        follow by the input sampling resolution and the maxSz requirement.
        
        Padding process refers to the center of the captured wavefront
        center.
"""
def Create_VirtualAperture(uin, apt_in, maxSz, posL):
    """
        Input: 
            u1: input complex field
            phyinapt: Input physical aperture dimension
        Output: 
            u2: output complex field
            phyoutapt: Output physical aperture dimension

        This program input x-y sampling interval is the same
    """
    def tool_symmetry(u1, phyinapt):
        M, N = u1.shape
        delta = phyinapt[0] / (M -1)

        if M > N:
            square_pad = int(cp.ceil(0.5 * (M - N)))
            u2 = cp.pad(u1, ((0, 0), (square_pad, square_pad)), mode='constant', constant_values=0)
            phyoutapt = cp.multiply(delta, u2.shape)
        elif M < N:
            square_pad = int(cp.ceil(0.5 * (N - M)))
            u2 = cp.pad(u1, ((square_pad, square_pad), (0, 0)), mode='constant', constant_values=0)
            phyoutapt = cp.multiply(delta, u2.shape)
        else:
            u2 = u1
            phyoutapt = phyinapt

        return u2, phyoutapt
    
    # Get input wavefront dimension
    M, N = uin.shape

    # Calculate the spatial sampling density, unit meter
    delta = apt_in[0][0]/(M-1)

    # Calculate the output size
    M_prime = mat_round(maxSz/delta)

    # symmetry square padding 
    if M == N :
        pass
    else:
        uin, _ = tool_symmetry(uin, apt_in)

    # Padding size difference
    diff = cp.array(uin.shape) - cp.array([M,N]) # Have to put uin.shape into array to subtract M,N from it
    posL = posL + (diff /2) # update the virual point source location after symmetry

    # Update the symmetry aperture size
    M, _ = uin.shape

    # Virtual aperture extented padding on the boundary
    # difference round to nearest even number easy symmetry padding
    dM = 2 * cp.ceil((M_prime - M)/2)

    # symmetry padding on the x
    if dM > 0:
        # Using 0 boundary condition
        pad_width = int(dM/2)
        uout = cp.pad(uin, pad_width, mode='constant', constant_values=0)
        posO = posL + (pad_width) # update the virual point source location after padding
    else:
        uout = uin
        posO = posL

    # update the virual aperture size 
    apt_out = (int(delta * uout.shape[0]), int(delta * uout.shape[1]))

    return uout, apt_out, posO

"""
    Parameters input: 
        1. uin: input virtual wavefront
        2. L: aperture size, unit meter
        3. lambda: virtual wavelength, unit meter
        4. depth: propagation (focusing) depth, unit meter
        5. method: string for choosing variety wave propagation kernel 
        6. alpha: coefficient for accuracness, ideally >= 1, 0.1 is
        acceptable for small amplitude error (large phase error)

    Parameters output: 
        1. uout: output wavefront at the destination plane
"""
def Camera_Focusing(uin, L, lambda_val, depth, method, alpha):
    """
        RSD in the convolution foramt
    
	    assumes same x and y side lengths and uniform sampling
        u1 - source plane field
	    L - source and observation plane side lengths
	    lambda - wavelength
	    z - propagation distance
	    u2 - observation plane field
    """
    def propRSD_conv(u1, L, lambda_val, z):
        M, N = u1.shape
    
        l_1 = L[0]
        l_2 = L[1]

        dx = l_1/(M-1)
        dy = l_2/(N-1)

        z_hat = z/dx
        mul_square = lambda_val * z_hat/(M * dx)

        m = cp.linspace(0, M-1, M)
        m = m - M/2
        n = cp.linspace(0, N-1, N)
        n = n - N/2

        g_m, g_n = cp.meshgrid(m, n)

        h = cp.divide(cp.exp(1j * 2 * cp.pi * (cp.square(z_hat) * cp.sqrt(1 + (cp.square(g_m) / cp.square(z_hat) + cp.square(g_n) / cp.square(z_hat))) / (mul_square * M))), cp.sqrt(1 + cp.square(g_m) / cp.square(z_hat) + cp.square(g_n) / cp.square(z_hat)))

        H = cp.fft.fft2(h)
        U1 = cp.fft.fft2(u1)
        U2 = U1 * H
        u2 = cp.fft.ifftshift(cp.fft.ifft2(U2))

        return u2

    """"
        Input: 
            u1: input complex field
            phyinapt: Input physical aperture dimension
            lambda : wavelength, unit m
            depth: unit m
            alpha: uncertainty parameter
        Output: 
            u2: output complex field
            phyoutapt: Output physical aperture dimension
            pad_size: one sided padding size
            sM: ola square symmetry, parameter for redoing the padding

        This program input x-y sampling interval is the same, this program
        also check if input field is symmetry (square)
    """
    def tool_fieldzeropadding(u1, phyinapt, lambda_val, depth, alpha):
        M, N = u1.shape
        delta = phyinapt[0]/(M-1)
        if (M == N):
            pass
        else:
            u1, _ = tool_symmetry(u1, phyinapt)

        sM, _ = u1.shape

        N_uncertaint = (lambda_val * cp.abs(depth)) / cp.square(delta)
        pad_size = (N_uncertaint - sM)/ 2

        if (pad_size > 0):
            pad_size = cp.round(alpha * pad_size)
        else:
            pad_size = 0

        u2 = cp.pad(u1, pad_width=pad_size, mode='constant', constant_values=0)
        phyoutapt = delta * (u2.shape - 1)
        
        """
            Input: 
                u1: input complex field
                phyinapt: Input physical aperture dimension
            Output: 
                u2: output complex field
                phyoutapt: Output physical aperture dimension

            This program input x-y sampling interval is the same
        """
        def tool_symmetry(u1, phyinapt):
            M, N = u1.shape
            delta = phyinapt[0] / (M -1)

            if M > N:
                square_pad = int(cp.ceil(0.5 * (M - N)))
                u2 = cp.pad(u1, ((0, 0), (square_pad, square_pad)), mode='constant', constant_values=0)
                phyoutapt = cp.multiply(delta, u2.shape)
            elif M < N:
                square_pad = int(cp.ceil(0.5 * (N - M)))
                u2 = cp.pad(u1, ((square_pad, square_pad), (0, 0)), mode='constant', constant_values=0)
                phyoutapt = cp.multiply(delta, u2.shape)
            else:
                u2 = u1
                phyoutapt = phyinapt

            return u2, phyoutapt
        return u2, phyoutapt, pad_size, sM

    if (alpha != 0):
        u1_prime, aperturefullsize_prime, pad_size, Nd = tool_fieldzeropadding(uin, L, lambda_val, depth, alpha)
    else:
        u1_prime = uin
        aperturefullsize_prime = L
        pad_size = 0
        Nd = u1_prime.shape[0] 

    match method:
        case 'RSD convolution':
            uout = cp.fliplr(propRSD_conv(u1_prime, aperturefullsize_prime, lambda_val, depth))
    
    uout = uout[pad_size:pad_size + Nd, pad_size : pad_size + Nd]
    
    return uout

# create Virtual Aperture by zero padding
for index in range(u_total.shape[2]):
    tmp = u_total[:, :, index]
    u_tmp_slice, apt_tmp, _ = Create_VirtualAperture(tmp, aperturefullsize, v_apt_Sz, 0)
    u_tmp = u_tmp.astype(np.complex128)
    u_tmp[:, :, index] = u_tmp_slice
aperturefullsize = apt_tmp # update virtual aperture size
u_total = u_tmp

# Spatial downsampling by bounded resolution
u_total = u_total[0::2, 0::2, :] + u_total[1::2, 1::2, :]

# create depth slice for the volume
depth_loop = cp.arange(depth_min, depth_max + sample_spacing, 2 * sample_spacing)

# Reconstruction using fast RSD
nZ = len(depth_loop)
u_volume = cp.zeros((u_total.shape[0], u_total.shape[1], nZ))

print('Reconstruction ... ...')
start_time = time.time() # Starts the time measurement

for i in range(0,depth_loop.shape[0]):
    depth = depth_loop[i]
    u_tmp = cp.zeros((u_total.shape[0], u_total.shape[1]))

    for spectrum_index in range(0, lambda_loop.size):
        u_field = u_total[:, :, spectrum_index]
        lambda_val = lambda_loop[spectrum_index]
        omega_val = omega_space[spectrum_index]

        u1 = u_field * cp.exp(1j * omega_val * (depth + d_offset)/c_light * cp.ones(u_field.shape))
        u2_RSD_conv = Camera_Focusing(u1, aperturefullsize, lambda_val, depth, 'RSD convolution', 0)
        u_tmp = u_tmp + weight[spectrum_index] * u2_RSD_conv

    u_volume = u_volume.astype(cp.complex128)
    u_volume[:,:,i ] = u_tmp

end_time = time.time() # Ends the time measurment
mgn_volume = cp.abs(u_volume)
img = cp.max(mgn_volume, axis=2)

plt.figure()
plt.imshow(cp.asnumpy(img), cmap='hot')
plt.axis('image')
plt.axis('off')
plt.show()