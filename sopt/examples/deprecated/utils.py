#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import skimage, skimage.transform, skimage.data
from scipy.stats import circmean
from typing import Optional, List
import attr



def getSampleObj(npix: int = 256, 
                 mod_range: float = 1, 
                 phase_range: float = np.pi,
                 boundary_npix: int = 0,
                 boundary_value: float = 0)-> np.ndarray:
    """Creates a sample object using stock data from the skimage module.
    
    Parameters:
    npix - 
    Number of pixels in each axis of the object
    
    mod_range - 
    Maximum value of the modulus for the object pixels.
    
    phase_range - 
    Maximum value of the phase for the object pixels.
    """
    npix_without_boundary = npix - 2 * boundary_npix
    
    mod_img = skimage.img_as_float(skimage.data.camera())[::-1,::-1]
    phase_img = skimage.img_as_float(skimage.data.immunohistochemistry()[:,:,0])[::-1,::-1]
    mod = skimage.transform.resize(mod_img, [npix_without_boundary, npix_without_boundary], 
                                   mode='wrap', preserve_range=True) 
    phase = skimage.transform.resize(phase_img, [npix_without_boundary, npix_without_boundary],
                                     mode='wrap', preserve_range=True)
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * phase_range
    mod = (mod - np.min(mod)) / (np.max(mod) - np.min(mod)) * mod_range
    obj_without_boundary = mod * np.exp(1j * phase)
    obj_with_boundary = np.pad(obj_without_boundary, 
                               [[boundary_npix, boundary_npix],
                                [boundary_npix, boundary_npix]],
                               mode='constant',
                               constant_values=boundary_value)
    return obj_with_boundary



def genGaussianProbe(npix: int, 
                     stdev: float) -> np.ndarray:
    center = npix // 2
    xvals = np.arange(npix)
    XX, YY = np.meshgrid(xvals, xvals)
    r_squared = (XX - center)**2 + (YY - center)**2
    gaussian = np.exp(-r_squared/ (2 * stdev)**2) 
    return gaussian



def genSpeckle(npix: int, 
               window_size: int) -> np.ndarray:
    
    ran = np.exp(1j * np.random.rand(npix,npix) * 2 * np.pi)
    
    window = np.zeros((npix, npix))
    indx1 = npix // 2 - window_size // 2
    indx2 = npix // 2 + window_size // 2
    window[indx1: indx2, indx1: indx2] = 1
    t = window * ran
    
    ft = np.fft.fftshift(np.fft.fft2(t, norm='ortho'))
    absvals = np.abs(ft)
    angvals = np.angle(ft)

    angvals[::2] = (angvals[::2] + np.pi) % (2 * np.pi)
    angvals[:,::2] = (angvals[:, ::2] + np.pi) % (2 * np.pi)
    return absvals * np.exp(1j * angvals)



def genTransferFunctionPropagator(support_side_length_pix: int,
                                  pixel_size: int,
                                  wavelength: float,
                                  prop_dist: float) -> np.ndarray:

    '''Propogation using the Transfer function method. '''
    M = support_side_length_pix
    #k = 2 * np.pi / wavelength

    fx = np.fft.fftfreq(M, d=pixel_size)
    fy = np.fft.fftfreq(M, d=pixel_size)

    FX, FY = np.meshgrid(fx, fy)
    FX = np.fft.fftshift(FX)
    FY = np.fft.fftshift(FY)

    H = np.exp(-1j * np.pi * wavelength * prop_dist * (FX**2 + FY**2))
    H = np.fft.fftshift(H)
    return H



class PtychographySimulation(object):
    def __init__(self, 
                 obj_npix: int = 256,
                 obj_mod_range: float = 1.0,
                 obj_phase_range: float = np.pi,
                 probe_filename: Optional[str] = 'probe_square_prop.npy',
                 probe_array: Optional[np.array] = None,
                 probe_n_photons: float = 1e6,
                 obj_padding_npix: int = 40,
                 obj_padding_const: float = 1.0,
                 positions_step_npix: int = 6,
                 poisson_noise: bool = True):
        self._obj_npix = obj_npix
        self._obj_mod_range = obj_mod_range
        self._obj_phase_range = obj_phase_range
        
        if probe_array is not None:
            self._probe_filename = None
            self._probe_array = probe_array.copy()
        elif probe_filename is not None:
            self._probe_filename = probe_filename
            self._probe_array = np.load(probe_filename)
        else:
            raise
        
        self._probe_n_photons = probe_n_photons
        self._obj_padding_npix = obj_padding_npix
        self._obj_padding_const = obj_padding_const
        self._positions_step_npix = positions_step_npix
        self._poisson_noise = poisson_noise
        
        self._genObj()
        self._genProbe()
        
        self._genPtychographyPositions()
        self._genDiffractionModuli()
        self._genViewIndices()
        self._ndiffs = self._diffraction_moduli.shape[0]
        
    def _genObj(self) -> None:
        self._obj_true = getSampleObj(self._obj_npix)
        pad = self._obj_padding_npix
        self._obj_padded = np.pad(self._obj_true, [[pad,pad],[pad,pad]], 
                                 mode='constant',
                                 constant_values=self._obj_padding_const)
        self._obj_padded_npix = self._obj_npix + pad * 2
        
    def _genProbe(self) -> None:
        
        self._probe_npix = self._probe_array.shape[0]
        self._probe_true = (self._probe_array / np.sqrt(np.sum(np.abs(self._probe_array)**2)) 
                            * np.sqrt(self._probe_n_photons))
        
    def _genPtychographyPositions(self) -> None:
        positions_x = np.arange(0, self._obj_padded_npix - self._probe_npix, 
                                self._positions_step_npix)
        positions = []
        for r in positions_x:
            for c in positions_x:
                positions.append([r,c])
        self._positions = np.array(positions)
    
    def _genDiffractionModuli(self) -> None:
        diffraction_intensities = []
        for indx, (r,c) in enumerate(self._positions):
            r2 = r + self._probe_npix
            c2 = c + self._probe_npix
            obj_slice = self._obj_padded[r:r2, c:c2]
            exit_wave = self._probe_true * obj_slice
            farfield_wave = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(exit_wave), norm='ortho'))
            diffraction_intensities.append(np.abs(farfield_wave)**2)
        if self._poisson_noise: 
            diffraction_intensities = np.random.poisson(diffraction_intensities)
        self._diffraction_moduli = np.sqrt(diffraction_intensities)
        
    def _genViewIndices(self) -> None:
        views_indices_all = []
        for py, px in self._positions:
            R, C = np.ogrid[py:self._probe_npix + py, px:self._probe_npix + px]
            view_single = ((R % self._obj_padded_npix) * self._obj_padded_npix + 
                           (C % self._obj_padded_npix))
            views_indices_all.append(view_single)
        self._view_indices =  np.array(views_indices_all)



class NearFieldObjParams(object):
    def __init__(self, 
                 npix: int = 192,
                 mod_range: float = 1.0,
                 phase_range: float = np.pi,
                 padding_npix: int = 32,
                 padding_const: float = 1.0) -> None:
        self.npix = npix
        self.mod_range = mod_range
        self.phase_range = phase_range
        self.padding_npix = padding_npix
        self.padding_const = padding_const
        self.padded_npix = self.npix + 2 * self.padding_npix
    
class NearFieldProbeParams(object):
    def __init__(self, 
                 npix: int = 512,
                 photons_flux: float = 1e4, # number of photons per pixel
                 gaussian_width_pix: Optional[int] = None,
                 speckle_window_pix: int = 40) -> None:
        self.npix = npix
        self.photons_flux = photons_flux
        self.n_photons = self.photons_flux * self.npix**2
        if gaussian_width_pix is not None:
            self.gaussian_width_pix = gaussian_width_pix
        else:
            self.gaussian_width_pix = self.npix // 8
        self.speckle_window = speckle_window_pix



class NearFieldPtychographySimulation(object):
    def __init__(self,
                 obj_args: dict = {},
                 probe_args: dict = {},
                 obj_detector_distance: float = 0.0468, #Effective Propagation distance from sample to detector(m)
                 detector_pixel_size: float = 3e-7, # (m)
                 wavelength: float = 0.142e-9, #(m)
                 positions_step_npix: int = 44,
                 positions_zero_buffer_npix: int = 20,
                 poisson_noise: bool = True) -> None:
            
        self._obj_params = NearFieldObjParams(**obj_args)
        self._probe_params = NearFieldProbeParams(**probe_args)
        
        self._obj_detector_distance = obj_detector_distance
        self._detector_pixel_size = detector_pixel_size
        self._wavelength = wavelength
        
        self._positions_step_npix = positions_step_npix
        self._positions_zero_buffer_npix = positions_zero_buffer_npix
        self._poisson_noise = poisson_noise
        
        self._genObj()
        self._genProbe()
        
        self._prop_kernel = genTransferFunctionPropagator(support_side_length_pix=self._probe_params.npix,
                                                          pixel_size=self._detector_pixel_size,
                                                          wavelength=self._wavelength,
                                                          prop_dist=self._obj_detector_distance)
        
        self._genPtychographyPositions()
        self._genDiffractionModuli()
        self._genScatterIndices()
        self._ndiffs = self._diffraction_moduli.shape[0]
    
    def _genObj(self) -> None:
        self._obj_true = getSampleObj(npix=self._obj_params.npix,
                                      mod_range=self._obj_params.mod_range,
                                      phase_range=self._obj_params.phase_range,
                                      boundary_npix=0,
                                      boundary_value=0.0).astype('complex64')
        pad = self._obj_params.padding_npix
        self._obj_padded = np.pad(self._obj_true, [[pad,pad],[pad,pad]], 
                                      mode='constant',
                                      constant_values=self._obj_params.padding_const)
        
    def _genProbe(self) -> None:
        gaussian = genGaussianProbe(self._probe_params.npix, 
                                    self._probe_params.gaussian_width_pix)
        speckle = genSpeckle(self._probe_params.npix, 
                             self._probe_params.speckle_window)
        data = gaussian * speckle
        self._probe_true = (data * np.sqrt(self._probe_params.n_photons / np.sum(np.abs(data)**2))).astype('complex64')
    
    def _genPtychographyPositions(self) -> None:
        probe_center = self._probe_params.npix // 2
        
        positions_x = np.arange(self._positions_zero_buffer_npix, 
                                self._probe_params.npix - self._obj_params.padded_npix - self._positions_zero_buffer_npix,
                                self._positions_step_npix)
        positions = []
        
        for r in positions_x:
            for c in positions_x:
                positions.append([r,c])
        self._positions = np.array(positions)
    
    def _genDiffractionModuli(self) -> None:
        diffraction_intensities = []
        npix_diff = self._probe_params.npix - self._obj_params.padded_npix
        obj_padded_to_probe = np.pad(self._obj_padded, 
                                     [[0, npix_diff], [0, npix_diff]],
                                     mode='constant',
                                     constant_values=1.0)
        for indx, (r,c) in enumerate(self._positions):
            exit_wave = self._probe_true * np.roll(obj_padded_to_probe, [r,c], axis=(0,1)) 
                       
            #r2 = r + self._obj_params.padded_npix
            #c2 = c + self._obj_params.padded_npix
            
            #exit_wave = self._probe_true.copy()
            #exit_wave[r:r2, c:c2] *= self._obj_padded
            nearfield_wave = np.fft.ifftshift(np.fft.ifft2(self._prop_kernel * np.fft.fft2(exit_wave)))
            diffraction_intensities.append(np.abs(nearfield_wave)**2)
        if self._poisson_noise: 
            diffraction_intensities = np.random.poisson(diffraction_intensities)
        self._diffraction_moduli = np.sqrt(diffraction_intensities)
        
    def _genScatterIndices(self) -> None:
        scatter_indices_all = []
        for py, px in self._positions:
            R, C = np.ogrid[py:self._obj_params.padded_npix + py, px:self._obj_params.padded_npix + px]
            scatter_single = ((R % self._probe_params.npix) * self._probe_params.npix + 
                              (C % self._probe_params.npix))
            scatter_indices_all.append(scatter_single)
        self._scatter_indices =  np.array(scatter_indices_all)

