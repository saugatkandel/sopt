#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import abc
import tensorflow as tf
from typing import Optional
from sopt.examples.deprecated.utils import PtychographySimulation
from sopt.optimizers.tensorflow import Curveball, LMA
from skimage.feature import register_translation
from pandas import DataFrame
import matplotlib.pyplot as plt



class OptimizerParams(object):
    def __init__(self):        
        self.training_loss_tensor = None
        self.batch_predicted_data = None
        self.obj_optimizer = None
        self.obj_minimize_op = None
        self.probe_optimizer = None
        self.probe_minimize_op = None



class tfPtychoReconsSim(metaclass=abc.ABCMeta):
    def __init__(self, 
                 ptsim: PtychographySimulation,
                 obj_guess_cmplx_2d: Optional[np.ndarray] = None,
                 probe_recons: bool = False,
                 probe_guess_cmplx_2d: Optional[np.ndarray] = None,
                 batch_size: int = 0,
                 validation_ndiffs: int = 0,
                 loss_type: str = "gaussian",
                 precondition_probe: bool = False) -> None:
        
        self._ptsim = ptsim
        self._precondition_probe = precondition_probe
        
        if obj_guess_cmplx_2d is None:
            n = self._ptsim._obj_npix
            obj_guess_cmplx_2d = (np.random.random((n,n)) * 
                           np.exp(1j * np.random.random((n,n)) * np.pi))
            
        self._obj_guess_flat = np.array([np.real(obj_guess_cmplx_2d),
                                         np.imag(obj_guess_cmplx_2d)], dtype='float32').flatten() 
        
        self._probe_recons = probe_recons
        if not probe_recons:
            self._probe_guess_flat = np.array([np.real(self._ptsim._probe_true),
                                               np.imag(self._ptsim._probe_true)], dtype='float32').flatten()
        elif probe_guess_cmplx_2d is not None:
            self._probe_guess_flat = np.array([np.real(probe_guess_cmplx_2d),
                                               np.imag(probe_guess_cmplx_2d)], dtype='float32').flatten()
        else:
            diffs_avg = np.mean(ptsim._diffraction_moduli, axis=0)
            probe_guess = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(diffs_avg), norm='ortho'))
            self._probe_guess_flat = np.array([np.real(probe_guess),
                                               np.imag(probe_guess)], dtype='float32').flatten()

        if probe_recons and precondition_probe:
            self._probe_guess_flat /= self._ptsim._diffraction_moduli.max()
        
        self._ndiffs = self._ptsim._ndiffs
        
        self._validation_ndiffs = validation_ndiffs
        self._validation_indices = np.random.permutation(self._ndiffs)[:self._validation_ndiffs]
        
        self._train_ndiffs = self._ndiffs - self._validation_ndiffs
        self._train_indices = np.array([i for i in range(self._ndiffs) if i not in self._validation_indices])
        
        self._batch_size = self._train_ndiffs if batch_size==0 else batch_size
        self._loss_type = loss_type

        
        self._createGraphAndVars()
        self._initDataSet()
        self._initConvenienceFunctions()
        
        self._optimizers_defined = False
        self._optparams = OptimizerParams()
        
        self.data = DataFrame(columns=['loss','epoch','obj_error','probe_error','validation_loss','patience'],
                              dtype='float32')
        
    def _createGraphAndVars(self) -> None:
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/gpu:0'):
                self._tf_obj = tf.get_variable('obj_flat', dtype='float32',
                                               initializer=self._obj_guess_flat,
                                               constraint=self._getObjConstraint)
                self._tf_probe = tf.get_variable('probe_flat', dtype='float32',
                                                 initializer=self._probe_guess_flat)
                
                shifted_diffs = np.fft.fftshift(self._ptsim._diffraction_moduli, axes=(1,2))
                self._train_diff_mods_shifted = tf.constant(shifted_diffs[self._train_indices],
                                                            dtype='float32',  name='train_diff_mod')
                
                self._validation_diff_mods_shifted = tf.constant(shifted_diffs[self._validation_indices],
                                                                 dtype='float32', name='validation_diff_mod')
                
                self._train_view_indices = tf.constant(self._ptsim._view_indices[self._train_indices], 
                                                       name='train_obj_view_indices')
                self._validation_view_indices = tf.constant(self._ptsim._view_indices[self._validation_indices], 
                                                       name='validation_obj_view_indices')
            
            self._tf_obj_padded_cmplx = self._getComplexPaddedObj(self._tf_obj)
            self._tf_probe_cmplx = self._getComplexProbe(self._tf_probe)
                
    
    def _initDataSet(self) -> None:
        with self.graph.as_default():
            dataset_indices = tf.data.Dataset.range(self._train_ndiffs)
            dataset_indices = dataset_indices.apply(tf.data.experimental.shuffle_and_repeat(
                buffer_size=self._train_ndiffs))
            dataset_batch = dataset_indices.batch(self._batch_size, drop_remainder=True)
            self._dataset_batch = dataset_batch.prefetch(5)

            self._iterator = self._dataset_batch.make_one_shot_iterator()

            self._batchi = self._iterator.get_next()
            self._batch_indices = tf.get_variable(name='batch_indices', 
                                                  dtype=tf.int64, 
                                                  shape=[self._batch_size],
                                                  initializer=tf.zeros_initializer, 
                                                  trainable=False)
            self._assign_op = self._batch_indices.assign(self._batchi)
            
            self._batch_train_mods = tf.reshape(tf.gather(self._train_diff_mods_shifted, self._batch_indices), [-1])
            
            self._batch_train_obj_view_indices = tf.gather(self._train_view_indices, self._batch_indices)
    
    def _getComplexPaddedObj(self, obj_tensor_flat: tf.Tensor) -> tf.Tensor:
        with self.graph.as_default():
            obj_reshaped = tf.reshape(obj_tensor_flat, [2, self._ptsim._obj_npix, self._ptsim._obj_npix])
            obj_cmplx = tf.complex(obj_reshaped[0], obj_reshaped[1])

            pad = self._ptsim._obj_padding_npix
            obj_padded = tf.pad(obj_cmplx, [[pad, pad], [pad, pad]], 
                                   constant_values=self._ptsim._obj_padding_const) 
        return obj_padded
    
    def _getComplexProbe(self, probe_tensor_flat: tf.Tensor) -> tf.Tensor:
        with self.graph.as_default():
            probe_reshaped = tf.reshape(probe_tensor_flat, [2, self._ptsim._probe_npix, self._ptsim._probe_npix])
            probe_cmplx = tf.complex(probe_reshaped[0], probe_reshaped[1])
        return probe_cmplx
            
    
    def _getPredictedData(self,
                         obj_flat: tf.Tensor,
                         probe_flat: tf.Tensor,
                         batch_obj_view_indices: tf.Tensor) -> tf.Tensor:
        with self.graph.as_default():
            obj_padded = self._getComplexPaddedObj(obj_flat)
            probe_cmplx = self._getComplexProbe(probe_flat)

            batch_obj_views = tf.gather(tf.reshape(obj_padded, [-1]), batch_obj_view_indices)
            batch_exit_waves = batch_obj_views * probe_cmplx
            batch_farfield_waves = tf.fft2d(batch_exit_waves) / self._ptsim._probe_npix
            batch_guess_mods = tf.reshape(tf.abs(batch_farfield_waves), [-1])

            if self._precondition_probe:
                batch_guess_mods *= self._ptsim._diffraction_moduli.max()
            #epsilons = tf.ones_like(batch_guess_mods) * (1e-3)**0.5
            #batch_guess_mods_new = tf.where(batch_guess_mods > (1e-3)**0.5, batch_guess_mods, epsilons)
        return batch_guess_mods
    
    def _getLoss(self, 
                 batch_predictions: tf.Tensor,
                 batch_diff_mods: tf.Tensor) -> tf.Tensor:
        with self.graph.as_default():
            if self._loss_type=="gaussian":
                loss = tf.reduce_sum((batch_predictions - batch_diff_mods)**2)
            elif self._loss_type == "poisson":
                preds = batch_predictions**2
                diffs = batch_diff_mods**2
                loss = tf.reduce_sum(preds - diffs * tf.log(preds))
            elif self._loss_type == "poisson_surrogate":
                diffs = batch_diff_mods**2
                preds = batch_predictions**2
                epsilons = tf.ones_like(preds) * 1e-2
                preds = tf.where(preds > 1e-2, preds, epsilons)
                loss = tf.reduce_sum(preds - diffs * tf.log(preds))
            else:
                raise KeyError("loss type not supported")
        return loss
    
    def _initConvenienceFunctions(self) -> None:
        self._training_predictions_fn = lambda x, y: self._getPredictedData(x, y, self._batch_train_obj_view_indices)
        
        self._training_predictions_as_obj_fn = lambda x: self._training_predictions_fn(x, self._tf_probe)
        self._training_predictions_as_probe_fn = lambda x: self._training_predictions_fn(self._tf_obj, x)
        
        self._training_loss_fn = lambda x: self._getLoss(x, self._batch_train_mods)
        if self._loss_type=="poisson":
            self._training_loss_hessian_fn = lambda x: (2 + (2 * self._batch_train_mods**2 / (x**2 )))
        elif self._loss_type=="poisson_surrogate":
            def hessian_fn(x):
                diffs = self._batch_train_mods**2
                preds = x**2
                epsilons = tf.ones_like(preds) * 1e-2
                preds = tf.where(preds > 1e-2, preds, epsilons)
                return 2 + (2 * diffs / preds)
            self._training_loss_hessian_fn = hessian_fn
        
        with self.graph.as_default():
            with tf.name_scope('validation'):
                self._validation_predictions_tensor = self._getPredictedData(self._tf_obj, 
                                                                             self._tf_probe,
                                                                             self._validation_view_indices)
                validation_diff_mods = tf.reshape(self._validation_diff_mods_shifted, [-1])
                self._validation_loss_tensor = self._getLoss(self._validation_predictions_tensor, 
                                                                       validation_diff_mods)
    
    
    def _getObjConstraint(self, obj_flat: tf.Tensor, max_abs: float=1.0) -> None:
        with self.graph.as_default():
            with tf.name_scope('obj_clip'):
                obj_reshaped = tf.reshape(obj_flat, [2, -1])
                obj_clipped = tf.clip_by_norm(obj_reshaped, max_abs, axes=[0])
                obj_clipped_reshaped = tf.reshape(obj_clipped, [-1])
        return obj_clipped_reshaped
    
    @abc.abstractmethod
    def setOptimizingParams(self, *args, **kwargs):
        pass
    
    def initSession(self):
        assert self._optimizers_defined, "Create optimizers before initializing the session."
        with self.graph.as_default():
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
            self.session.run(tf.global_variables_initializer())
            self.session.run(self._assign_op)
            
    def _objRegistrationError(self):
        
        recons_obj_padded = self.session.run(self._tf_obj_padded_cmplx)
        npix = self._ptsim._obj_padding_npix
        recons_obj = recons_obj_padded[npix:-npix, npix:-npix]
        shift, err, phase = register_translation(recons_obj, self._ptsim._obj_true, upsample_factor=10)
        shift, err, phase = register_translation(recons_obj * np.exp(-1j * phase), self._ptsim._obj_true, upsample_factor=10)
        return err
    
    def run(self, 
            validation_frequency: int = 1,
            improvement_threshold: float = 5e-4,
            patience: int = 50,
            patience_increase_factor: float = 1.5,
            max_iters: int = 5000,
            debug_output: bool = True,
            debug_output_epoch_frequency: int = 10,
            probe_fixed_epochs=0,
            disable_progress_bar=False, 
            manual_assign=False) -> None:
        
        epochs_this = 0
        index = len(self.data)
        for i in range(max_iters):#, disable=disable_progress_bar):
            ix = index + i

            if self._probe_recons and epochs_this >= probe_fixed_epochs: 
                _ = self.session.run(self._optparams.probe_minimize_op)
                    
            self.session.run(self._optparams.obj_minimize_op)
            lossval = self.session.run(self._optparams.training_loss_tensor)

            if not manual_assign:
                _ = self.session.run(self._assign_op)
            self.data.loc[ix, 'loss'] = lossval
            
            if ix == 0:
                self.data.loc[0, 'epoch'] = 0
                continue
            elif ix % (self._train_ndiffs // self._batch_size) != 0:
                self.data.loc[ix, 'epoch'] = self.data['epoch'][ix-1]
                continue
            
            self.data.loc[ix, 'epoch'] = self.data['epoch'][ix-1] + 1 
            epochs_this += 1
            
            if epochs_this % validation_frequency != 0:
                continue
            validation_loss = self.session.run(self._validation_loss_tensor)
            self.data.loc[ix, 'validation_loss'] = validation_loss
            
            obj_registration_error = self._objRegistrationError()
            self.data.loc[ix, 'obj_error'] = obj_registration_error
            
            validation_best_loss = np.inf if ix == 0 else self.data['validation_loss'][:-1].min()
            
            if validation_loss <= validation_best_loss:
                if np.abs(validation_loss - validation_best_loss) > validation_best_loss * improvement_threshold:
                    patience = max(patience, epochs_this * patience_increase_factor)
                
            self.data.loc[ix, 'patience'] = patience
                
            if debug_output and epochs_this % debug_output_epoch_frequency == 0:
                print(f'{epochs_this} '
                       + f'{lossval:8.7g} '
                       + f'{obj_registration_error:8.7g} '
                       + f'{patience:8.7g} '
                       + f'{validation_loss:8.7g} '
                       + f'{validation_best_loss:8.7g}')
            
            if epochs_this >= patience:
                break
                
    
    def genPlotsRecons(self):
        npix = self._ptsim._obj_padding_npix
        recons_obj = self.session.run(self._tf_obj_padded_cmplx)[npix:-npix, npix:-npix]
        
        plt.figure(figsize=[8,3])
        plt.subplot(1,2,1)
        plt.pcolormesh(np.abs(recons_obj), cmap='gray')
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.pcolormesh(np.angle(recons_obj), cmap='gray')
        plt.colorbar()
        plt.show()
    
    def genPlotMetrics(self):
        fig, axs = plt.subplots(1,3,figsize=[12,3])
        axs[0].plot(np.log(self.data['loss'].dropna()))
        axs[0].set_title('loss')
        
        axs[1].plot(self.data['obj_error'].dropna())
        axs[1].set_title('obj_error')
        
        axs[2].plot(np.log(self.data['validation_loss'].dropna()))
        axs[2].set_title('validation_loss')
        plt.show()



class AdamPhaseRetriever(tfPtychoReconsSim):
     
    def setOptimizingParams(self, 
                            learning_rate_obj: float=1e-2,
                            learning_rate_probe: float=1e-1) -> None:
        
        self._optparams.learning_rate_obj = learning_rate_obj
        self._optparams.learning_rate_probe = learning_rate_probe
        
        with self.graph.as_default():
            self._optparams.batch_predicted_data = self._training_predictions_fn(self._tf_obj, self._tf_probe)
            
            self._optparams.training_loss_tensor = self._training_loss_fn(self._optparams.batch_predicted_data)
            
            self._optparams.obj_optimizer = tf.train.AdamOptimizer(self._optparams.learning_rate_obj)
            self._optparams.obj_minimize_op = self._optparams.obj_optimizer.minimize(self._optparams.training_loss_tensor,
                                                                                     var_list=[self._tf_obj])
            
            if self._probe_recons:
                self._optparams.probe_optimizer = tf.train.AdamOptimizer(self._optparams.learning_rate_probe)
                self._optparams.probe_minimize_op = self._optparams.probe_optimizer.minimize(self._optparams.training_loss_tensor, 
                                                                                   var_list=[self._tf_probe])
        self._optimizers_defined = True



class CurveballPhaseRetriever(tfPtychoReconsSim):
     
    def setOptimizingParams(self, 
                            damping_factor_obj: float=1.0,
                            damping_update_factor_obj: float=0.999,
                            damping_update_frequency_obj: int=5,
                            damping_factor_probe: float=1.0,
                            damping_update_factor_probe: float=0.999,
                            damping_update_frequency_probe: float=5,
                            update_cond_threshold_low: float = 0.5, 
                            update_cond_threshold_high: float = 1.5):
        
        if self._loss_type in ["poisson", "poisson_surrogate"]:
            loss_hessian_fn = self._training_loss_hessian_fn
            squared_loss = False
        else:
            loss_hessian_fn = None
            squared_loss = True
        with self.graph.as_default():
            self._optparams.obj_optimizer = Curveball(input_var=self._tf_obj,
                                                      predictions_fn=self._training_predictions_as_obj_fn,
                                                      loss_fn=self._training_loss_fn,
                                                      damping_factor=damping_factor_obj,
                                                      damping_update_factor=damping_update_factor_obj,
                                                      damping_update_frequency=damping_update_frequency_obj,
                                                      update_cond_threshold_low=update_cond_threshold_low,
                                                      update_cond_threshold_high=update_cond_threshold_high,
                                                      name='obj_opt',
                                                      diag_hessian_fn=loss_hessian_fn,
                                                      squared_loss=squared_loss)
            self._optparams.obj_minimize_op = self._optparams.obj_optimizer.minimize()
            
            if self._probe_recons:
                self._optparams.probe_optimizer = Curveball(input_var=self._tf_probe,
                                                            predictions_fn=self._training_predictions_as_probe_fn,
                                                            loss_fn=self._training_loss_fn,
                                                            damping_factor=damping_factor_probe,
                                                            damping_update_factor=damping_update_factor_probe,
                                                            damping_update_frequency=damping_update_frequency_probe,
                                                            name='probe_opt',
                                                            diag_hessian_fn=loss_hessian_fn,
                                                            squared_loss=squared_loss)
                self._optparams.probe_minimize_op = self._optparams.probe_optimizer.minimize()
            
            self._optparams.training_loss_tensor = self._optparams.obj_optimizer._loss_fn_tensor
        self._optimizers_defined = True



class LMAPhaseRetriever(tfPtychoReconsSim):
    def setOptimizingParams(self):
        
        if self._loss_type in ["poisson", "poisson_surrogate"]:
            loss_hessian_fn = self._training_loss_hessian_fn
        else:
            loss_hessian_fn = None
        
        size = self._batch_train_mods.shape.as_list()[0]
        
        with self.graph.as_default():
            self._optparams.obj_optimizer = LMA(input_var=self._tf_obj,
                                                predictions_fn=self._training_predictions_as_obj_fn,
                                                loss_fn=self._training_loss_fn,
                                                name='obj_opt',
                                                diag_hessian_fn=loss_hessian_fn,
                                                assert_tolerances=False)
            self._optparams.obj_minimize_op = self._optparams.obj_optimizer.minimize()
            
            if self._probe_recons:
                self._optparams.probe_optimizer = LMA(input_var=self._tf_probe,
                                                      predictions_fn=self._training_predictions_as_probe_fn,
                                                      loss_fn=self._training_loss_fn,
                                                      name='probe_opt',
                                                      diag_hessian_fn=loss_hessian_fn,
                                                      assert_tolerances=False)
                self._optparams.probe_minimize_op = self._optparams.probe_optimizer.minimize()
            
            self._optparams.training_loss_tensor = self._optparams.obj_optimizer._loss_t
        self._optimizers_defined = True


