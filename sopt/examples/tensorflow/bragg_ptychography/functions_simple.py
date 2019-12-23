#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tensorflow as tf 
from skimage import filters
from skimage import measure
from skimage import transform
from scipy import special as special
import scipy
from scipy.interpolate import RegularGridInterpolator
from ptychoSampling.utils.register_translation_3d import register_translation_3d
from tqdm import tqdm



def getAiryProbe(wavelength=0.142e-9, # 8.7 keV 
                 pixel_pitch=30e-9, # 55 micrometers 
                 npix=32,
                 n_photons=1e6,
                 beam_diam_pixels=5):
    """Calculates the beam profile given the final beam diameter. 
    
    Parameters:
    
    wavelength : 
    self-explanatory
    
    pixel_pitch : 
    object/probe pixel pitch. Usually calculated using the Nyquist theorem using the object-detector
    distance and the detector pixel pitch.
    
    n_pix:
    number of pixels along each axis in the probe view
    
    n_photons:
    self-explanatory
    
    beam_diam_pixels:
    the diameter (in pixels) of the first central lobe of the probe beam at the object (sample) plane.

    Assumption:
    - propagation distance (from aperture to sample) and initial aperture width are calculated 
    assuming a Fresnel number of 0.1
    """
    beam_width_dist = beam_diam_pixels * pixel_pitch
    radius = beam_width_dist / 2
    # Initial Aperture width
    w = 0.1 * 2 * np.pi * radius / (special.jn_zeros(1, 1))
    
    # Propagation dist from aperture to sample
    z = 0.1 * (2 * np.pi * radius)**2 / (special.jn_zeros(1, 1)**2 * wavelength)
    
    beam_center = npix // 2
    xvals = np.linspace( -beam_center * pixel_pitch, beam_center * pixel_pitch, npix)
    xx, yy = np.meshgrid(xvals, xvals)
    

    k = 2 * np.pi / wavelength
    lz = wavelength * z
    S = xx**2 + yy**2

    jinc_input = np.sqrt(S) * w / lz
    mask = (jinc_input != 0)
    jinc_term = np.pi * np.ones_like(jinc_input)
    jinc_term[mask] = special.j1(jinc_input[mask] * 2 * np.pi) / jinc_input[mask]

    # wavefield 
    #term1 = np.exp(1j * k * z) / (1j * lz)
    term1 = 1 / (1j * lz)
    term2 = np.exp(1j * k * S / (2 * z))
    term3 = w**2 * jinc_term
    field_vals = (term1 * term2 * term3).astype('complex64')

    scaling_factor = np.sqrt(n_photons / (np.abs(field_vals)**2).sum())
    field_vals = scaling_factor * field_vals
    return field_vals



def calcDiffractionIntensities(cell, probe, step_npix):
    cell_npix_p, cell_npix_r, cell_npix_c = cell.shape
    probe_npix = probe.shape[1]
    
    grid_positions_r = np.arange(0, cell_npix_r - probe_npix, step_npix)
    grid_positions_c = np.arange(0, cell_npix_c - probe_npix, step_npix)
    
    diff_intensities = []
    positions = []
    
    non_zero_pix = 0
    
    for r in grid_positions_r:
        for c in grid_positions_c:
            exit_wave = cell[:, r: r + probe_npix, c: c + probe_npix ] * probe
            non_zero_pix += np.sum(exit_wave > 0)
            exit_wave_proj = np.sum(exit_wave, axis=1)
            exit_wave_ft = np.fft.fftshift(np.fft.fft2(exit_wave_proj, norm='ortho', axes=(0,1)), axes=(0,1))
            diff_temp = np.abs(exit_wave_ft)**2
            diff_intensities.append(diff_temp)
            positions.append([r,c])
            
    print('Non zero data pixels', non_zero_pix)
    diff_intensities = np.array(diff_intensities)
    # Adding poisson noise
    diff_intensities = np.random.poisson(diff_intensities)
    return np.array(positions), diff_intensities



def genViewIndices(obj, positions, npix_r, npix_c):
    nx, ny, nz = obj.shape
    views_indices_all = []
    for py, pz in positions:
        
        #for pz in range(nz):
        #    R, C = np.ogrid[py:npix + py, px:npix + px]
        #    view_single = (R % obj.shape[0]) * obj.shape[0] + (C % obj.shape[1])
        X, Y, Z = np.ogrid[0:nx, py:npix_r + py, pz:npix_c + pz]
        view_single = ((X % nx) * ny + (Y % ny)) * nz + (Z % nz)
        #print(view_single.shape)
        views_indices_all.append(view_single)
    return np.array(views_indices_all)



def initDataset(ndiffs, batch_size, num_prefetch_batch_diffs=1):
        dataset_indices = tf.data.Dataset.range(ndiffs)
        dataset_indices = dataset_indices.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=ndiffs))
        
        dataset_batch = dataset_indices.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset_batch = dataset_batch.prefetch(num_prefetch_batch_diffs)
        return dataset_indices, dataset_batch



def tensor_clip(t: tf.Tensor, 
                  max_abs: float=1.,
                  min_abs: float=0.):
    
    absval = tf.abs(t)
    abs_clipped = tf.clip_by_value(absval, min_abs, max_abs)
    multiplier = tf.cast(abs_clipped / (absval + 1e-30), 'complex64')
    return t * multiplier



def batch_fftshift2d(tensor: tf.Tensor):
    # Shifts high frequency elements into the center of the filter
    indexes = len(tensor.get_shape()) - 1
    top, bottom = tf.split(tensor, 2, axis=indexes)
    tensor = tf.concat([bottom, top], indexes )
    left, right = tf.split(tensor, 2, axis=indexes - 1)
    tensor = tf.concat([right, left], indexes - 1 )
    return tensor

def plotReconstructionCrossSections(obj_true, obj_recons, fname_prefix='reconstructed_faceted_cell'):
    x_slices = np.linspace(0, obj_true.shape[0] - 1, 6).astype('int')
    y_slices = np.linspace(0, obj_true.shape[1] - 1, 6).astype('int')
    z_slices = np.linspace(0, obj_true.shape[2] - 1, 6).astype('int')

    figsize_r = np.max([x_slices.size, y_slices.size, z_slices.size])
    plt.figure(figsize=[figsize_r * 2.5, 6 * 2.5])

    for indx, x_slice in enumerate(x_slices):
        plt.subplot(6, figsize_r, indx+1)
        plt.pcolormesh(np.abs(obj_recons[x_slice, :, :]), rasterized=True)
        #plt.title(f'mod, x slice {x_slice} (rec)')
        plt.subplot(6, figsize_r, figsize_r + indx +1)
        plt.pcolormesh(np.abs(obj_true[x_slice, :, :]), rasterized=True)
        #plt.title(f'mod, x slice {x_slice} (true)')

    for indx, y_slice in enumerate(y_slices):
        plt.subplot(6, figsize_r, 2* figsize_r + indx+1)
        plt.pcolormesh(np.abs(obj_recons[:,y_slice, :]), rasterized=True)
        #plt.title(f'mod, y slice {y_slice} (rec)')
        plt.subplot(6, figsize_r, 3 * figsize_r + indx +1)
        plt.pcolormesh(np.abs(obj_true[:,y_slice, :]), rasterized=True)
        #plt.title(f'mod, y slice {y_slice} (true)')

    for indx, z_slice in enumerate(z_slices):
        plt.subplot(6, figsize_r, 4* figsize_r + indx+1)
        plt.pcolormesh(np.abs(obj_recons[:,:,z_slice]), rasterized=True)
        #plt.title(f'mod, z slice {z_slice} (rec)')
        plt.subplot(6, figsize_r, 5 * figsize_r + indx +1)
        plt.pcolormesh(np.abs(obj_true[:,:,z_slice]), rasterized=True)
        #plt.title(f'mod, z slice {z_slice} (true)')

    plt.tight_layout()
    #plt.savefig(f'{fname_prefix}_abs.png')
    plt.show()

    plt.figure(figsize=[figsize_r * 2.5, 6 * 2.5])
    for indx, x_slice in enumerate(x_slices):
        plt.subplot(6, figsize_r, indx+1)
        plt.pcolormesh(np.angle(obj_recons[x_slice, :, :]), rasterized=True)
        #plt.title(f'mod, x slice {x_slice} (rec)')
        plt.subplot(6, figsize_r, figsize_r + indx +1)
        plt.pcolormesh(np.angle(obj_true[x_slice, :, :]), rasterized=True)
        #plt.title(f'mod, x slice {x_slice} (true)')

    for indx, y_slice in enumerate(y_slices):
        plt.subplot(6, figsize_r, 2* figsize_r + indx+1)
        plt.pcolormesh(np.angle(obj_recons[:,y_slice, :]), rasterized=True)
        #plt.title(f'mod, y slice {y_slice} (rec)')
        plt.subplot(6, figsize_r, 3 * figsize_r + indx +1)
        plt.pcolormesh(np.angle(obj_true[:,y_slice, :]), rasterized=True)
        #plt.title(f'mod, y slice {y_slice} (true)')

    for indx, z_slice in enumerate(z_slices):
        plt.subplot(6, figsize_r, 4* figsize_r + indx+1)
        plt.pcolormesh(np.angle(obj_recons[:,:,z_slice]), rasterized=True)
        #plt.title(f'mod, z slice {z_slice} (rec)')
        plt.subplot(6, figsize_r, 5 * figsize_r + indx +1)
        plt.pcolormesh(np.angle(obj_true[:,:,z_slice]), rasterized=True)
        #plt.title(f'mod, z slice {z_slice} (true)')

    plt.tight_layout()
    #plt.savefig(f'{fname_prefix}_phase.png')
    plt.show()

def calcError(obj_true, obj_test):
    roll, err, phase = register_translation_3d(obj_test, obj_true, 10)
    obj_test_2 = obj_test * np.exp(-1j * phase)
    obj_test_2 = np.roll(obj_test_2, -roll.astype('int'), axis=(0,1,2))
    roll_2, err_2, phase_2 = register_translation_3d(obj_test_2, obj_true, 10)
    return roll, err, phase, roll_2, err_2, phase_2





