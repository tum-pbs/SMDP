from physics import forward_full
import wandb
import numpy as np
from utils.plots import plot_fft_power, plot_fft_normalized
from utils.utils import radial_profile


class Optimizer(object):
    model_weights = None
    params = None
    forward_fn = None

    def __init__(self, generator, params):

        self.params = params
        self.train_generator = generator[0]
        self.val_generator = generator[1]
        self.test_generator = generator[2]

        self.build_model()

    def predict(self, data_keys, rng, realizations=1, **kwargs):
        raise NotImplementedError

    def eval_reconstruction(self, **kwargs):

        # TODO adapt predict so that we need to provide the file as well
        data_keys = [elem[1] for elem in list(self.test_generator.keys)[:500]]

        rng = self.params['sample_key']

        data = self.predict(data_keys, rng, realizations=1, **kwargs)

        forward = []
        mse = []

        forward_full_fn = lambda pred: forward_full(pred, self.params['t1'], self.params['t1'])

        # TODO fix bad jit compilation of this function
        # forward_full_fn = jax.jit(forward_full_fn)

        c = 0
        for pred in list(data['prediction_0']):
            forward.append(forward_full_fn(pred)[-1])

            mse.append(np.mean(np.square(forward[-1] - data['forward'][c])))

            c += 1

        mse_ = sum(mse) / len(mse)
        wandb.summary['test_mse'] = mse_

        return mse_

    def eval_power_spectrum(self, **kwargs):

        data_keys = [elem[1] for elem in list(self.test_generator.keys)[:2]]

        rng = self.params['sample_key']

        NUM_REALIZATIONS = 10

        data = self.predict(data_keys, rng, realizations=NUM_REALIZATIONS, **kwargs)

        shape = (self.params['resolution'], self.params['resolution'])
        unit_length = 1
        all_k = [np.fft.fftfreq(s, d=unit_length) for s in shape]

        kgrid = np.meshgrid(*all_k, indexing="ij")
        knorm = np.sqrt(np.sum(np.power(kgrid, 2), axis=0))

        power_k = np.zeros_like(knorm)
        mask = knorm > 0

        # TODO specify power as params value
        def power_spectrum(k):
            return np.power(k, -4)

        power_k[mask] = np.sqrt(power_spectrum(knorm[mask]))

        data_fft = {}
        data_normalized = {}

        data_fft_gt = {}
        data_normalized_gt = {}
        data_fft_input = {}
        data_normalized_input = {}

        for n in range(len(data['prediction_0'])):

            data_fft[n] = []
            data_normalized[n] = []

            elem = data['ground_truth'][n]

            elem = np.abs(np.fft.fftshift(np.real(np.fft.ifftn(elem))))
            radial_profile_elem = np.log(radial_profile(elem, (int(shape[0] / 2), int(shape[1] / 2))))
            data_fft_gt[n] = radial_profile_elem
            elem[mask] /= power_k[mask]
            data_normalized_gt[n] = elem

            elem = data['input'][n]

            elem = np.abs(np.fft.fftshift(np.real(np.fft.ifftn(elem))))
            radial_profile_elem = np.log(radial_profile(elem, (int(shape[0] / 2), int(shape[1] / 2))))
            data_fft_input[n] = radial_profile_elem
            elem[mask] /= power_k[mask]
            data_normalized_input[n] = elem

            for i in range(NUM_REALIZATIONS):
                elem = data[f'prediction_{i}'][n]

                elem = np.abs(np.fft.fftshift(np.real(np.fft.ifftn(elem))))

                radial_profile_elem = np.log(radial_profile(elem, (int(shape[0] / 2), int(shape[1] / 2))))

                data_fft[n].append(radial_profile_elem)

                elem[mask] /= power_k[mask]

                data_normalized[n].append(elem)

        plots_fft_power = plot_fft_power(data_fft, data_fft_gt, data_fft_input)
        plots_fft_normalized = plot_fft_normalized(data, data_normalized, data_normalized_gt, data_normalized_input)

        return plots_fft_normalized, plots_fft_power

    def train(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
