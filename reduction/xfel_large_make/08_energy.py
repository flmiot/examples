""" 08_energy: Read ROI data from 05_integrate.h5 in ANOTHER directory,
    search for scanned elastic peaks and fit a function to describe the 
    position of the leastic peaks as a function of incoming photon energy.
    
    Then, determine the calibrated energy axes for the ROI in this directory.

    Call as: python 08_energy.py /scratch/directory /config/directory

    Depends on the 08_energy.yml config file.
"""

import os
import sys
import yaml
import h5py
import logging
import numpy as np
import tools
import scipy.optimize as optim

log_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.log'
yml_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.yml'
hdf_file_name = os.path.split(os.path.splitext(__file__)[0])[1] + '.h5'


def energy_from_mono_2(position):
    x0 = -96.6671
    l1 = 218.484
    d = 1.977
    t0 = 0.359
    
    return d / np.sin( np.arcsin( (position+x0)/l1 ) + t0 ) * 1e3

def energy_from_pixel_pos(position, miller_h, miller_k, miller_l, lattice_constant):
    d = lattice_constant / np.sqrt(miller_h**2+miller_k**2+miller_l**2)
    factor = 4.135e-15 * 299792458 /(2 * d) 
    sin_angle = 1/np.sqrt((position / 500 / 2)**2 + 1)
    return factor / sin_angle

def calculate_com(e, i, window=None):

    if window is not None:
        e0, e1 = window
        i0, i1 = np.argmin(np.abs(e - e0)), np.argmin(np.abs(e - e1))
        i, e = i[i0:i1+1], e[i0:i1+1]

    return np.nansum(i * e) / np.nansum(i)

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
        
    with open(configp('detectors.yml'), 'r') as file:
        detectors = yaml.load(file.read(), Loader = yaml.SafeLoader)

    d = {}
    
    for idx, roi in enumerate(config['roi']):
        
        name = str(idx)
        d[name] = {}
        
        base = roi['directory']
        
        # Load integrate.h5 from base directory
        input_d = tools.load_dict_from_hdf5(os.path.join(base, '05_integrate.h5'))
        
        roi_name = str(roi['roi'])
        input_roi = input_d[roi_name]
        
        indizes = np.array(roi['indizes'])
        motor = input_roi['motor']
        signal = (input_roi['on'] + input_roi['off'])[indizes]
        energies = energy_from_mono_2(motor)[indizes]
        pixels = np.arange(signal.shape[1])

        pos = []
        for s in signal:
            idx_max = np.nanargmax(s)
            com = calculate_com(pixels, s, window = [idx_max - 10, idx_max + 10])
            pos.append(com)

        positions = np.array(pos) * 75e-3 #75um pixel size -> mm

        def residuals(params, position, energies, miller_indizes, lattice_constant):
            height_offset = params
            return energy_from_pixel_pos(
                position + height_offset, *miller_indizes, lattice_constant) - energies

        x0 = [roi['guess_height']]
        r = optim.least_squares(fun = residuals, x0 = x0, 
                                args = (positions, energies, roi['miller_indices'], roi['lattice_constant']))

        res = residuals(r.x, positions, energies, roi['miller_indices'], roi['lattice_constant'])
        calibrated_x = energy_from_pixel_pos(pixels * 75e-3 + r.x, *roi['miller_indices'], roi['lattice_constant'])

        d[name]['cal_pixels'] = pixels
        d[name]['cal_energies'] = calibrated_x
        d[name]['miller_indices'] = roi['miller_indices']
        d[name]['lattice_constant'] = roi['lattice_constant']
        d[name]['fit_residuals'] = res

        d[name]['raw_pixels'] = pos
        d[name]['raw_energies'] = energies
        d[name]['fitted_height_offset'] = r.x[0]
            
            
    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
