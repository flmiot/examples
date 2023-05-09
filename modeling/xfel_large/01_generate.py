import os
import sys
import h5py
import yaml
import numpy as np
import logging

from scipy.stats import qmc

log_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.log'
yml_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.yml'
hdf_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.h5'

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

    component_names = list(config['model']['components'].keys())
    no_components = len(component_names)
    no_parameters = sum([len(p) for p in config['model']['components'].values()])

    Log.info("Generating guess parameter values for model with {} components and {} total parameters".format(no_components, no_parameters))

    iterations = config['no_fits']
    points_m = config['points_m']
    guess = np.empty((iterations, 2**points_m, no_parameters))

    p_per_comp = int(no_parameters / no_components)
    sampler = qmc.Sobol(d=no_components*p_per_comp*iterations, scramble = True)
    guess = sampler.random_base2(m=points_m).reshape((iterations, 2**points_m, no_components*p_per_comp))

    # Scale parameter guesses according to their bounds
    min_values = [p['min'] for c in config['model']['components'].values() for p in c.values()]
    max_values = [p['max'] for c in config['model']['components'].values() for p in c.values()]
    for i in range(no_parameters):
        guess[:, :, i] = guess[:, :, i] * (max_values[i] - min_values[i]) + min_values[i]

    with h5py.File(scratchp(hdf_file_name), 'w') as h5_file:
        h5_file['guess'] = guess
