""" 09_assemble: Read ROI data from multiple h5 files in this directory and
    assemble them inside one collector file for further data processing. 

    Call as: python 09_assemble.py /scratch/directory /config/directory

    Depends on the 09_assemble.yml config file.
"""

import os
import sys
import yaml
import h5py
import logging
import numpy as np
import tools
import scipy.optimize as optim
import scipy.interpolate as interp

log_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.log'
yml_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.yml'
hdf_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.h5'

def calculate_errors(arr):
    return np.std(arr, axis = 0)

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
        detectors = yaml.load(file.read(), Loader = yaml.SafeLoader)

    d = {}
    bin_d = tools.load_dict_from_hdf5(os.path.join(scratch_directory, '06_bin.h5'))
    subtract_d = tools.load_dict_from_hdf5(os.path.join(scratch_directory, '07_subtract.h5'))
    energy_d = tools.load_dict_from_hdf5(os.path.join(scratch_directory, '08_energy.h5'))

    for key in bin_d.keys():
        name = str(key)
        d[name] = {}
        
        pixels = energy_d[name]['cal_pixels']
        energies = energy_d[name]['cal_energies']
        f = interp.interp1d(pixels, energies)
        d[name]['energy'] = f(bin_d[name]['x'])
        
        for k in ['on', 'off', 'motor']:
            d[name][k] = bin_d[name][k]
            
        d[name]['difference'] = subtract_d[name]['difference'] 
        std = calculate_errors(d[name]['off'])
        d[name]['on_errors'] = std
        d[name]['off_errors'] = std
        d[name]['difference_errors'] = np.sqrt(std**2+std**2)
            
    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
