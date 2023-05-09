""" 07_subtract: Read ROI data from 06_bin.h5 in the same directory and
    compute difference spectra

    Call as: python 07_subtract.py /scratch/directory /config/directory

    Depends on the 07_subtract.yml config file.
"""

import os
import sys
import yaml
import h5py
import logging
import numpy as np
import tools

log_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.log'
yml_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.yml'
hdf_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.h5'

def normalize(arr, i0, i1):
    s = np.nansum(arr[..., i0:i1], axis = -1)
    return arr / s[:, None] * 1000.

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
    
    with open(configp('01_filter.yml'), 'r') as file:
        fconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)

    with open(configp(yml_file_name), 'r') as file:
        sconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)
        
    d = {}
    integrate_d = tools.load_dict_from_hdf5(scratchp('06_bin.h5'))
    for key, value in integrate_d.items():
        d[key] = {}
        d[key]['motor'] = value['motor']
        d[key]['x'] = value['x']
        
        x0, x1 = sconfig['roi'][int(key)]['normalization_region']
        norm_region = [
            np.argmin(np.abs(value['x'] - x0)),
            np.argmin(np.abs(value['x'] - x1))
        ]
        
        laser_on = normalize(value['on'], *norm_region)
        laser_off = normalize(value['off'], *norm_region)
        
        d[key]['difference'] = laser_on - laser_off
                
    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True) 
        
        
        
            
        
        