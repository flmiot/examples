""" 06_bin: Read ROI data from 05_integrate.h5 in the same directory and
    bin integrated spectra along x. 

    Call as: python 06_bin.py /scratch/directory /config/directory

    Depends on the 06_bin.yml config file.
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

def bin_data(x, y, bin_size = 2):
    """ Will bin among first dimension """
    

   
    shape = (int(y.shape[0] / bin_size),) + y.shape[1:]
    binned_y = np.zeros( (bin_size, ) + shape )
    binned_x = np.zeros( (bin_size, ) + (shape[0], ) )
    
    for i in range(bin_size):
        i0 = i
        i1 = i - bin_size +1  if i < bin_size-1 else None 
        step = bin_size
        binned_y[i] = y[i0:i1:step] 
        binned_x[i] = x[i0:i1:step]
        
    return np.nanmean(binned_x, axis = 0), np.nanmean(binned_y, axis = 0)
    

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
        
    with open(configp('01_filter.yml'), 'r') as file:
        fconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)
        
    with open(configp('05_integrate.yml'), 'r') as file:
        iconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)
        
    d = {}
    integrate_d = tools.load_dict_from_hdf5(scratchp('05_integrate.h5'))
    
    for key, value in integrate_d.items():
        name = str(key)
        d[name] = {}
        roi = config['roi'][int(key)] 
        
        Log.info("Binning roi {} with bin size {}".format(name, roi['bin_size']))
        
        for lstate in ['on', 'off']:
            bin_x, bin_y = bin_data(value['x'], value[lstate].T, bin_size = roi['bin_size'])
            d[name]['x'] = bin_x
            d[name][lstate] = bin_y.T
            
        d[name]['motor'] = value['motor']
        
    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
        
        
        
            
        
        