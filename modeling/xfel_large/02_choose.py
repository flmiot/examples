import os
import sys
import yaml
import h5py
import numpy as np
import operator
from functools import reduce
import logging

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

if __name__ == '__main__':
    scratch_directory = sys.argv[1]
    config_directory = sys.argv[2]

    def scratchp(path):
        return os.path.join(scratch_directory, path)

    def configp(path):
        return os.path.join(config_directory, path)

    Log = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG,
                        filename=scratchp(log_file_name),
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    Log.addHandler(logging.StreamHandler())

    Log.info(("-"*10 +  os.path.split(os.path.splitext(__file__)[0])[1].upper() + "-"*10))

    with open(configp(yml_file_name), 'r') as file:
        config = yaml.load(file.read(), Loader = yaml.SafeLoader)

    with open(configp('01_generate.yml'), 'r') as file:
        gconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)

    # Model
    component_names = list(gconfig['model']['components'].keys())
    no_components = len(component_names)

    # Open fit data
    with h5py.File(config['fit_data']['file'], 'r') as h5_file:
        x = h5_file[config['fit_data']['x']][()]
        y = np.atleast_2d(h5_file[config['fit_data']['y']][()])
        motor = h5_file[config['fit_data']['motor']][()]
        errors = np.atleast_2d(h5_file[config['fit_data']['errors']][()])

    # Sanatize fitting data
    b = np.isnan(y.sum(axis = 0)) | np.isnan(errors.sum(axis = 0))
    x = x[~b]
    y = y[:, ~b]
    errors = errors[:, ~b]

    # Apply fitting range
    idx0, idx1 = [np.argmin(np.abs(x-v)) for v in config['fit_data']['x_range']]
    s = slice(idx0, idx1+1)
    x = x[s]
    y = y[:, s]
    errors = errors[:, s]

    d = {}
    d['fit_data'] = {}
    d['fit_data']['x'] = x
    d['fit_data']['y'] = y
    d['fit_data']['errors'] = errors
    d['fit_data']['motor'] = motor


    # Read guesses and compute residuals
    with h5py.File(scratchp('01_generate.h5'), 'r') as h5_file:
        guess = h5_file['guess'][()]


    iterations, points, no_parameters = guess.shape
    guess_res = guess.reshape((iterations, points, no_components, -1))


    Log.info("Read guess array for {} fit, {} points and {} parameters".format(iterations, points, no_parameters))

    Log.info("Reshape guess array to {}, corresponding to {} fits, {} points, {} components and {} parameters per component".format(guess_res.shape, iterations, points, no_components, guess_res.shape[-1]))

    residuals = np.zeros(guess.shape[:2] + x.shape)

    Log.info("Creating residuals array with shape {}".format(residuals.shape))

    func = models[gconfig['model']['type']]

    for i in range(no_components):

        params = guess_res[:, :, i, :]
        residuals = residuals + func(x, params)

    steps = y.shape[0]
    best_guess = np.empty((steps, iterations, no_parameters))

    for i in range(steps):
        loc_res =  residuals - y[np.newaxis, np.newaxis, i, :]
        loc_res = np.nansum(loc_res**2, axis = -1)
        idx = np.argmin(loc_res, axis = -1)
        r = np.take_along_axis(guess, idx[:, np.newaxis, np.newaxis], axis = 1)
        best_guess[i] = r.squeeze()


    d['params'] = {}
    d['params']['values'] = best_guess

    min_values = [p['min'] for c in gconfig['model']['components'].values() for p in c.values()]
    max_values = [p['max'] for c in gconfig['model']['components'].values() for p in c.values()]
    names = [c_name + k for (c_name, c) in gconfig['model']['components'].items() for k in c.keys()]

    d['params']['min'] = min_values
    d['params']['max'] = max_values
    d['params']['names'] = np.string_(names)


    # import matplotlib.pyplot as plt
    #
    # for i in range(5):
    #
    #     plt.figure()
    #     for c in best_guess[0, i].reshape((no_components, -1)):
    #         plt.plot(x, lorentzian_1d(x, c))
    #     plt.plot(x, y[0])
    #
    # plt.show()

    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
