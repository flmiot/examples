""" 04_fill: Read ROI data from 03_sort.h5 in the same directory and
    fill specified pixels (x,y) with the mean of the their closest x 
    neighbors

    Call as: python 04_fill.py /scratch/directory /config/directory

    Depends on the 04_fill.yml config file.
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

def interpolate_horizontal(arr, pixel_x, pixel_y):
    
    marr = arr.mean(axis = (0,1))
    for pix, piy in zip(pixel_x, pixel_y):
    
        value = np.nan
        left_idx = pix
        while(np.isnan(value)):
            left_idx = left_idx - 1
            value = marr[piy, left_idx]
            
        value = np.nan
        right_idx = pix
        while(np.isnan(value)):
            right_idx = right_idx + 1
            value = marr[piy, right_idx]
            
        fill_value = 0.5 * (arr[:, :, piy, left_idx] + arr[:, :, piy, right_idx])
        Log.debug("Filling pixel at x= {} / y= {}".format(pix, piy))

        arr[:, :, piy, pix] = fill_value
        
    return arr
    

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
        fconfig = yaml.load(file.read(), Loader = yaml.SafeLoader)
        
    d = {}
    sort_d = tools.load_dict_from_hdf5(scratchp('03_sort.h5'))
    for idx, roi in enumerate(fconfig['roi']):
        
        
            
        name = str(idx)
        
        if not name in sort_d.keys():
            continue
        
        d[name] = {}
        d[name]['motor'] = sort_d[name]['motor']
        d[name]['on'] = interpolate_horizontal(sort_d[name]['on'], roi['x'], roi['y'])
        d[name]['off'] = interpolate_horizontal(sort_d[name]['off'], roi['x'], roi['y'])

        
                
    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)