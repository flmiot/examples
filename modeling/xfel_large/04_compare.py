import os
import sys
import yaml
import h5py
import numpy as np
import operator
from functools import reduce
import logging
from multiprocessing import Pool

import tools

log_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.log'
yml_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.yml'
hdf_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.h5'

models = [
    tools.voigt,
    tools.lorentzian,
    tools.pearson7
]

models_1d = [
    tools.voigt_1d,
    tools.lorentzian_1d,
    tools.pearson7_1d
]

def get_height(x, model, params):

    height = np.empty(params.shape[:2])

    for i in range(params.shape[0]):
        for s in range(params.shape[1]):

            max_idx = np.argmax(model(x = x, params = params[i, s]))
            x_fine = np.linspace(x[max_idx] - 5, x[max_idx] + 5, 2048)
            y_fine = model(x = x_fine, params = params[i, s])
            height[i, s] = np.nanmax(y_fine)

    return height

def get_fwhm(x, model, params):

    fwhm = np.empty(params.shape[:2])

    for i in range(params.shape[0]):
        for s in range(params.shape[1]):

            max_idx = np.argmax(model(x = x, params = params[i, s]))
            x_fine = np.linspace(x[max_idx] - 5, x[max_idx] + 5, 2048)
            y_fine = model(x = x_fine, params = params[i, s])

            max_idx = np.argmax(y_fine)
            y_fine = y_fine / np.nanmax(y_fine)

            il = np.argmin(np.abs(y_fine[:max_idx] - 0.5))
            ir = np.argmin(np.abs(y_fine[max_idx:] - 0.5)) + max_idx

            fwhm[i, s] = np.abs(x_fine[il] - x_fine[ir])

    return fwhm

def get_max_position(x, model, params):

    position = np.empty(params.shape[:2])

    for i in range(params.shape[0]):
        for s in range(params.shape[1]):
            max_idx = np.argmax(model(x = x, params = params[i, s]))
            x_fine = np.linspace(x[max_idx] - 5, x[max_idx] + 5, 2048)
            y_fine = model(x = x_fine, params = params[i, s])
            position[i, s] = x_fine[np.nanargmax(y_fine)]

    return position

def get_iwae(x, model, params):

    iwae = np.empty(params.shape[:2])

    for i in range(params.shape[0]):
        for s in range(params.shape[1]):

            energies = params[i, s, :, 0]
            intensities= params[i, s, :, 1]
            iwae[i, s] = np.sum(intensities*energies) / np.sum(intensities)

    return iwae


if __name__ == '__main__':

    sdir = sys.argv[1]
    cdir = sys.argv[2]

    def scratchp(*paths):
        return os.path.join(sdir, *paths)

    def configp(*paths):
        return os.path.join(cdir, *paths)

    Log = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG,
                        filename=scratchp(log_file_name),
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    Log.addHandler(logging.StreamHandler())

    Log.info(("-"*10 +  os.path.split(os.path.splitext(__file__)[0])[1].upper() + "-"*10))

    with open(configp(yml_file_name), 'r') as file:
        config = yaml.load(file.read(), Loader = yaml.SafeLoader)

    files = [f for f in sorted(os.listdir(scratchp('fits'))) if '.h5' in f]

    Log.info("Found {} files with fitting results. Collecting ... ".format(len(files)))

    with h5py.File(scratchp('fits', files[0]), 'r') as h5_file:
        sh = h5_file['params/best'].shape
        vmin = h5_file['params/min'][()]
        vmax = h5_file['params/max'][()]
        vnames = h5_file['params/names'][()]
        x = h5_file['fit_data/x'][()]
        y = h5_file['fit_data/y'][()]
        motor = h5_file['fit_data/motor'][()]
        errors = h5_file['fit_data/errors'][()]

    d = {}
    d['fit_data'] = {}
    d['fit_data']['x'] = x
    d['fit_data']['y'] = y
    d['fit_data']['motor'] = motor
    d['fit_data']['errors'] = errors

    d['params'] = {}
    d['params']['min'] = vmin
    d['params']['max'] = vmax
    d['params']['names']= vnames

    new_sh = (len(files),) + sh
    Log.info("Creating new arrays with shape {}".format(new_sh))
    best = np.empty(new_sh)
    errors = np.empty(new_sh)
    guess = np.empty(new_sh)
    chisqr = np.empty(new_sh[:2])
    redchi = np.empty(new_sh[:2])

    for i in range(new_sh[0]):
        with h5py.File(scratchp('fits', files[i]), 'r') as h5_file:
            best[i] = h5_file['params/best'][()]
            errors[i] = h5_file['params/errors'][()]
            guess[i] = h5_file['params/guess'][()]
            chisqr[i] = h5_file['params/chisqr'][()]
            redchi[i] = h5_file['params/redchi'][()]


    # Read model, calculate sum of residuals squared and sort fitting results
    with open(configp('01_generate.yml'), 'r') as file:
        gconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)
    component_names = list(gconfig['model']['components'].keys())
    no_components = len(component_names)
    iterations, points, no_parameters = best.shape
    best_res = best.reshape((iterations, points, no_components, -1))

    model_id = gconfig['model']['type']
    func = models[model_id]
    func_1d = models_1d[model_id]

    residuals = np.zeros(best_res.shape[:2] + x.shape)
    for i in range(no_components):
        params = best_res[:, :, i, :]
        residuals = residuals + func(x, params)

    residuals = residuals - y[np.newaxis]

    steps = y.shape[0]
    best_sorted = np.empty(new_sh)
    errors_sorted = np.empty(new_sh)
    guess_sorted = np.empty(new_sh)
    sqr    = np.empty(new_sh[:2])


    for i in range(steps):
        # files, steps, values
        loc_res = residuals[:, i, :]
        loc_res = np.nansum(loc_res**2, axis = -1)
        idx = np.argsort(loc_res, axis = -1)
        best_sorted[:, i, :] = best[idx, i, :]
        guess_sorted[:, i, :] = guess[idx, i, :]
        errors_sorted[:, i, :] = errors[idx, i, :]
        sqr[:, i] = loc_res[idx]
        chisqr[:, i] = chisqr[idx, i]
        redchi[:, i] = redchi[idx, i]


    d['params']['best_sorted'] = best_sorted
    d['params']['errors_sorted'] = errors_sorted
    d['params']['guess_sorted'] = guess_sorted
    d['params']['sqr'] = sqr
    d['params']['chisqr'] = chisqr
    d['params']['redchi'] = redchi
    d['params']['model_id'] = model_id

    # Descriptive parameters
    d['description'] = {}
    for group in config['description']:

        def model(x, params):
            """ *params* should have shape (M,N), where
            M = number of components
            N = number of parameters per components
            """

            y = np.zeros(x.shape)
            for i in range(params.shape[0]): # Iterate components
                y = y + func_1d(x, params[i])

            return y

        label_suffix = '({})_sorted'.format(','.join([str(n) for n in group['components']]))

        Log.info("Calculating descriptive parameters for group {}".format(label_suffix))

        params_group = best_sorted.reshape(best_sorted.shape[:2] + (no_components,-1))
        params_group = params_group[:, :, np.array(group['components'])-1, :]

        d['description']['fwhm'+label_suffix] = get_fwhm(x, model, params_group)
        d['description']['position'+label_suffix] = get_max_position(x, model, params_group)
        d['description']['iwae'+label_suffix] = get_iwae(x, model, params_group)
        d['description']['height'+label_suffix] = get_height(x, model, params_group)

        area = np.zeros(best_sorted.shape[:2])
        vnames_group = vnames.reshape((no_components, -1))
        vnames_group = vnames_group[np.array(group['components'])-1]
        idx_parameters = [
            'amplitude' == v[3:].decode('utf-8') for v in vnames_group.flatten()[:no_components]]
            
        area = params_group[..., idx_parameters].sum(axis = (2,3))
        d['description']['area'+label_suffix] = area


    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
