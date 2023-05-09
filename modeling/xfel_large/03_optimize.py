import os
import sys
import yaml
import h5py
import lmfit
import numpy as np
import operator
from functools import reduce
import logging
from multiprocessing import Pool
import time

import tools

log_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.log'
yml_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.yml'
hdf_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.h5'

models = [
    lmfit.models.VoigtModel,
    lmfit.models.LorentzianModel,
    lmfit.models.Pearson7Model
]


def worker(scratch_directory, config_directory, guess_index):

    def scratchp(*path):
        return os.path.join(scratch_directory, *path)

    def configp(*path):
        return os.path.join(config_directory, *path)

    with open(configp(yml_file_name), 'r') as file:
        config = yaml.load(file.read(), Loader = yaml.SafeLoader)

    with open(configp('01_generate.yml'), 'r') as file:
        gconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)

    # Model
    lmfit_model = models[gconfig['model']['type']]
    component_names = list(gconfig['model']['components'].keys())
    no_components = len(component_names)
    model = reduce(operator.add,
        [lmfit_model(prefix = name) for name in component_names])

    # Read guess parameters from file
    with h5py.File(scratchp('02_choose.h5'), 'r') as h5_file:
        values_min = h5_file['params']['min'][()]
        values_max = h5_file['params']['max'][()]
        values = h5_file['params']['values'][:, guess_index]
        names = h5_file['params']['names'][()]

    params = model.make_params()

    d = {}

    d['params'] = {}
    d['params']['names'] = names
    d['params']['guess'] = values
    d['params']['min'] = values_min
    d['params']['max'] = values_max
    d['params']['best'] = np.empty(values.shape)
    d['params']['errors'] = np.empty(values.shape)
    d['params']['chisqr'] = np.empty(values.shape[0])
    d['params']['redchi'] = np.empty(values.shape[0])


    d['fit_data'] = {}

    # Fit
    # Read fitting data from file
    with h5py.File(scratchp('02_choose.h5'), 'r') as h5_file:
        x = h5_file['fit_data']['x'][()]
        y = h5_file['fit_data']['y'][()]
        motor = h5_file['fit_data']['motor'][()]
        errors = h5_file['fit_data']['errors'][()]

    d['fit_data']['x'] = x
    d['fit_data']['y'] = y
    d['fit_data']['motor'] = motor
    d['fit_data']['errors'] = errors

    steps = values.shape[0]
    for step in range(steps):

        for name, vmin, vmax, v in zip(names, values_min, values_max, values[step]):

            name = name.decode('utf-8')

            params[name].value = v
            params[name].min   = vmin
            params[name].max   = vmax
            params[name].expr = ""
            params[name].vary = True

        r = model.fit(data = y[step], x = x, params = params, weights = 1/errors[step], scale_covar = False)

        step_values = [r.params[n.decode('utf-8')].value for n in names]
        step_errors = [r.params[n.decode('utf-8')].stderr for n in names]
        step_errors = [np.nan if v == None else v for v in step_errors]

        d['params']['best'][step] = step_values
        d['params']['errors'][step] = step_errors
        d['params']['chisqr'][step] = r.chisqr
        d['params']['redchi'][step] = r.redchi

    filename = 'fits/{:0>4d}.h5'.format(guess_index)
    tools.write_dict_to_hdf5(d, scratchp(filename), override = True)


if __name__ == '__main__':
    sdir = sys.argv[1]
    cdir = sys.argv[2]

    start = time.time()

    def scratchp(*path):
        return os.path.join(sdir, *path)

    def configp(*path):
        return os.path.join(cdir, *path)

    Log = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG,
                        filename=scratchp(log_file_name),
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    Log.addHandler(logging.StreamHandler())

    Log.info(("-"*10 +  os.path.split(os.path.splitext(__file__)[0])[1].upper() + "-"*10))

    # Delete old fit files
    try:
        old_files = [f for f in sorted(os.listdir(scratchp('fits'))) if '.h5' in f]
        Log.info("Deleting old fit files ({} files)".format(len(old_files)))
        for f in old_files:
            os.remove(scratchp('fits', f))
    except FileNotFoundError as e:
        Log.info("Creating fits directory")
        os.mkdir(scratchp('fits'))


    with open(configp(yml_file_name), 'r') as file:
        config = yaml.load(file.read(), Loader = yaml.SafeLoader)

    with h5py.File(scratchp('02_choose.h5'), 'r') as h5_file:
        steps, iterations = h5_file['params/values'].shape[:2]

    max_workers = config['workers']
    Log.info("Spawning worker pool (max {} workers) to crunch {} fits for {} steps".format(max_workers, iterations, steps))

    iter = [(sdir, cdir, i) for i in range(iterations)]

    with Pool(max_workers) as p:
        p.starmap(worker, iter)

    d = {}
    d['timing'] = {}
    d['timing']['workers'] = max_workers
    d['timing']['total_time'] = time.time() - start

    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
