#Author - Saugat Kandel
# coding: utf-8


from skimage.transform import rescale
from skimage.feature import register_translation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy
import tensorflow as tf
from optimizers.tensorflow import Curveball



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



def getPositions(obj_npix, probe_npix, step_size):
    #step_size = (obj_npix - probe_npix) // num_steps 
    positions_x = np.arange(0, obj_npix - probe_npix, step_size)
    positions = []
    for r in positions_x:
        for c in positions_x:
            positions.append([r, c])
    return np.array(positions)



def getDiffractionMods(obj_flat, probe_flat, obj_shape, probe_shape, positions):
    obj = np.reshape(obj_flat, obj_shape)
    probe = np.reshape(probe_flat, probe_shape)
    diffraction_mods = []
    #diffraction_mods = np.zeros(positions.shape[0] * probe.shape[0] * probe.shape[1], dtype='float32')
    for indx, (r, c) in enumerate(positions):
        r2 = r + probe.shape[0]
        c2 = c + probe.shape[0]
        obj_slice = obj[r:r2, c:c2]
        exit_wave = probe * obj_slice
        farfield_wave = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(exit_wave), norm='ortho'))
        diffraction_mods.append(np.abs(farfield_wave))
    return np.array(diffraction_mods)



#obj_npix = 512
obj_true_large = getTestImage()
obj_true_abs = rescale(np.abs(obj_true_large), preserve_range=True, scale=0.25)
obj_true_phase = rescale(np.angle(obj_true_large), preserve_range=True, scale=0.25)
obj_true = obj_true_abs * np.exp(1j * obj_true_phase)
pad = 40
obj_padded = np.pad(obj_true, [[pad,pad],[pad,pad]], mode='constant',constant_values=1.)
obj_npix = obj_padded.shape[0]

n_photons = 1e4
probe_true = np.load('../probe_square_prop.npy')#[16:48, 16:48]
probe_npix = probe_true.shape[0]
probe_true = probe_true / np.sqrt(np.sum(np.abs(probe_true)**2)) * np.sqrt(n_photons)

fig, axs = plt.subplots(1, 4, figsize=[14,3])
cax0 = axs[0].pcolormesh(np.abs(obj_padded), cmap='gray')
cax1 = axs[1].pcolormesh(np.angle(obj_padded), cmap='gray')
cax2 = axs[2].pcolormesh(np.abs(probe_true), cmap='gray')
cax3 = axs[3].pcolormesh(np.angle(probe_true), cmap='gray')
plt.colorbar(cax3)
plt.tight_layout()
plt.show()



positions = getPositions(obj_npix, probe_npix, 6)
diffraction_mods = getDiffractionMods(obj_padded.flatten(),
                                      probe_true.flatten(),
                                      obj_padded.shape, 
                                      probe_true.shape,
                                      positions)

diffraction_mods = np.sqrt(np.random.poisson(diffraction_mods**2))
#norm_factor = diffraction_mods.size**0.5



recons_pad = 3
obj_recons_shape = np.array(obj_true.shape) + recons_pad * 2
#obj_guess = np.random.randn(*obj_true.shape) + 1j * np.random.randn(*obj_true.shape)
obj_guess = np.random.randn(*obj_recons_shape) * np.exp(1j * np.random.randn(*obj_recons_shape))

pad_recons = pad - recons_pad

obj_true_target = obj_padded[pad-recons_pad:-pad+recons_pad, pad-recons_pad:-pad+recons_pad]

probe_guess = np.zeros_like(probe_true)
probe_guess[28:38, 28:38] = 1
probe_guess = probe_guess * np.sqrt(n_photons/np.sum(np.abs(probe_guess)**2))
plt.pcolormesh(np.abs(probe_guess))
plt.show()

#probe_guess = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.mean(diffraction_mods, axis=0)), norm='ortho'))
ndiffs = diffraction_mods.shape[0]



pr_adam = tfAdamPhaseRetriever(diffraction_mods, 
                             positions, 
                             obj_guess, 
                             probe_guess, 
                             batch_size=128, 
                             obj_padding=[[pad_recons, pad_recons],[pad_recons,pad_recons]],
                             obj_padding_const=1.,
                             learning_rate_probe=1e0,
                             learning_rate_obj=1e-2)



get_ipython().run_cell_magic('time', '', 'adam_errors = []\nfor i in tqdm(range(100)):\n    if i < 1:\n        pr_adam.run(10, n_probe_fixed_iterations=50, disable_progress_bar=True)\n    else:\n         pr_adam.run(10, n_probe_fixed_iterations=0, disable_progress_bar=True)\n    recons_obj = pr_adam.obj\n    shift, err, phase = register_translation(recons_obj, obj_true_target, upsample_factor=10)\n    shift, err, phase = register_translation(recons_obj * np.exp(-1j * phase), obj_true_target, upsample_factor=10)\n    adam_errors.append(err)\n    if i % 50 == 0: print(shift, err, phase)\n    if err < 0.02: break')



plt.pcolormesh(np.abs(pr_adam.obj), cmap='gray')
plt.colorbar()
plt.show()



fig, axs = plt.subplots(1,2,figsize=[8,3])
axs[0].plot(np.log(pr_adam.losses))
axs[1].plot(adam_errors)
plt.show()



pr_cg = tfCGPhaseRetriever(diffraction_mods, 
                             positions, 
                             obj_guess, 
                             probe_guess, 
                             batch_size=128, 
                             obj_padding=[[pad_recons, pad_recons],[pad_recons,pad_recons]],
                             obj_padding_const=1.,
                             max_iter_per_step=1)



get_ipython().run_cell_magic('time', '', 'cg_errors = []\nfor i in tqdm(range(100)):\n    if i < 1:\n        pr_cg.run(10, n_probe_fixed_iterations=50, disable_progress_bar=True)\n    else:\n         pr_cg.run(10, n_probe_fixed_iterations=0, disable_progress_bar=True)\n    recons_obj = pr_cg.obj\n    shift, err, phase = register_translation(recons_obj, obj_true_target, upsample_factor=10)\n    shift, err, phase = register_translation(recons_obj * np.exp(-1j * phase), obj_true_target, upsample_factor=10)\n    cg_errors.append(err)\n    if i % 20==0: print(shift, err, phase)\n    if err < 0.02: break')



fig, axs = plt.subplots(1,2,figsize=[8,3])
axs[0].plot(np.log(pr_cg.losses))
axs[1].plot(cg_errors)
plt.show()



pr_cb = tfCurveballPhaseRetriever(diffraction_mods, 
                                  positions, 
                                  obj_guess, 
                                  probe_guess, 
                                  batch_size=ndiffs, 
                                  obj_padding=[[pad_recons, pad_recons],[pad_recons,pad_recons]],
                                  obj_padding_const=1.0)



get_ipython().run_cell_magic('time', '', 'cb_errors = []\nfor i in tqdm(range(100)):\n    if i < 1:\n        pr_cb.run(10, n_probe_fixed_iterations=50, disable_progress_bar=True, obj_clip=False)\n    else:\n        pr_cb.run(10, n_probe_fixed_iterations=0, disable_progress_bar=True, obj_clip=False)\n    recons_obj = pr_cb.obj\n    shift, err, phase = register_translation(recons_obj, obj_true_target, upsample_factor=10)\n    shift, err, phase = register_translation(recons_obj * np.exp(-1j * phase), obj_true_target, upsample_factor=10)\n    cb_errors.append(err)\n    if i % 20 == 0: print(shift, err, phase)\n    if err < 0.02: break')



fig, axs = plt.subplots(1,2,figsize=[14,3])
axs[0].plot(np.log(pr_adam.losses), color='red')
axs[0].plot(np.log(pr_cg.losses), color='blue')
axs[0].plot(np.log(pr_cb.losses), color='green')
axs[1].plot(adam_errors, color='red')
axs[1].plot(cg_errors, color='blue')
axs[1].plot(cb_errors, color='green')
plt.show()



plt.pcolormesh(np.abs(pr_cb.obj), cmap='gray')
plt.colorbar()
plt.show()
    





