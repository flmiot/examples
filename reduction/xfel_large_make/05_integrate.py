""" 05_integrate: Read ROI data from 04_fill.h5 in the same directory and
    integrate ROI along y. Specified background ROI will be subtracted. 

    Call as: python 05_integrate.py /scratch/directory /config/directory

    Depends on the 05_integrate.yml config file.
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
        iconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)
        
    d = {}
    sort_d = tools.load_dict_from_hdf5(scratchp('04_fill.h5'))
    for idx, roi in enumerate(iconfig['roi']):
        
        name = str(idx)
        signal_name = str(roi['signal'])
        axes = tuple(roi['integration_axes'])
        
        d[name] = {}
        d[name]['motor'] = sort_d[signal_name]['motor']
        d[name]['x'] = np.arange(*fconfig['roi'][iconfig['roi'][int(name)]['signal']]['dim1'])
        
        Log.info("Integrating roi {}".format(name))
        
        norm_region = iconfig['roi'][int(name)]['normalization_region']
        
        if norm_region != False:
            x0, x1 = norm_region
            norm_region = [
                np.argmin(np.abs(d[name]['x'] - x0)),
                np.argmin(np.abs(d[name]['x'] - x1))
            ]

        for lstate in ['on', 'off']:
            d[name][lstate] = np.nanmean(sort_d[signal_name][lstate], axis = axes)         
            l = len(roi['background'])
            
            for background_id in roi['background']:
                background_id = str(background_id)
                d[name][lstate] -= np.nanmean(sort_d[background_id]['on'], axis = axes) / l
                
            # Normalize
            if norm_region != False:
                d[name][lstate] = normalize(d[name][lstate], *norm_region)
            
                
    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
        
        
        
            
        
        