""" 02_calibrate: Calibrate JungFrau detector data read from 01_filter.h5
    in the same directory. 

    Call as: python 02_calibrate.py /scratch/directory /config/directory

    Depends on the 02_calibrate.yml config file.
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

def calibrate(adc, gain, gain_constants, mask, pedestal, max_pixels_non_g0):
    
    gain[gain == 3] = 2
    cal_gain = np.choose(gain, [gain_constants[0], gain_constants[1], gain_constants[2]])
    cal_pedestal = np.choose(gain, [pedestal[0], pedestal[1], pedestal[2]])
    cal_mask = np.choose(gain, [mask[0], mask[1], mask[2]])
    
    corr = (adc - cal_pedestal)/cal_gain
    
    non_g0_pixels = (gain > 0).sum(axis = (1,2,3))

    corr[cal_mask == 1] = np.nan
    
    # Exclude all frames, where number of g1 or g2 pixels is too big
    corr[non_g0_pixels > max_pixels_non_g0] = np.nan 
    
    # Exclude all pixels in gain g1 or g2 
    corr[gain > 0] = np.nan 
    
    return corr


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
        fconf = yaml.load(file.read(), Loader = yaml.SafeLoader)
     
    with open(configp(yml_file_name), 'r') as file:
        cconf = yaml.load(file.read(), Loader = yaml.SafeLoader)
    
    with open(configp('detectors.yml'), 'r') as file:
        detectors = yaml.load(file.read(), Loader = yaml.SafeLoader)

    d = {}
    filter_d = tools.load_dict_from_hdf5(scratchp('01_filter.h5'))
    
    for idx, roi in enumerate(fconf['roi']):
        
        name = str(idx)
        
        y0, y1 = roi['dim0']
        x0, x1 = roi['dim1']
        det_name = roi['detector']
        det = cconf[det_name]
        det.update(detectors[det_name])
        
        # Get gain constants
        # Gain: dim0 and dim1 are switched
        gain_constants = np.empty((x1-x0, y1-y0, 1, 3))
        source_sel = (slice(x0, x1), slice(y0, y1), slice(None, None), slice(None, None))
        dest_sel = (slice(0, x1-x0), slice(0, y1-y0), slice(None, None), slice(None, None))
        
        Log.debug("Loading calibration gain data for {}".format(det_name))
        with h5py.File(det['gain']['file'], 'r') as h5_file:
            h5_file[det['gain']['dataset']].read_direct(
                        gain_constants, source_sel  = source_sel, dest_sel = dest_sel)
            
        gain_constants = gain_constants.transpose((3,2,1,0))
        
        # Get mask data
        border_mask = np.zeros((512, 1024, 1, 3))
        border_mask[(255, 256)] = 1
        border_mask[:, (255, 256, 511, 512, 767, 768)] = 1
        
        mask = np.empty((y1-y0, x1-x0, 1, 3))
        source_sel = (slice(y0, y1), slice(x0, x1), slice(None, None), slice(None, None))
        dest_sel = (slice(0, y1-y0), slice(0, x1-x0), slice(None, None), slice(None, None))
        
        border_mask = border_mask[source_sel]
        
        Log.debug("Loading calibration mask data for {}".format(det_name))
        with h5py.File(det['mask']['file'], 'r') as h5_file:
            h5_file[det['mask']['dataset']].read_direct(
                        mask, source_sel  = source_sel, dest_sel = dest_sel)
        
        mask[border_mask > 0] = 1
        mask = mask.transpose((3,2,0,1))

        # Get pedestal data
        pedestal = np.empty((3, 1, y1-y0, x1-x0))
        slice_indizes = det['pedestal']['indizes']
        source_sel = (slice(*slice_indizes), slice(0, 1),slice(y0, y1), slice(x0, x1))
        dest_sel = (slice(0, 1), slice(0, y1-y0), slice(0, x1-x0))
    
        
        Log.debug("Loading calibration pedestal data for {}".format(det_name))
        keys = ['g0', 'g1', 'g1']
        for fidx, key in enumerate(keys):
            folder = det['pedestal'][key]
            filenames = [f for f in sorted(os.listdir(folder)) if det['da_name'] in f]

            # We only need one file
            file_index = det['pedestal']['file_index']
            with h5py.File(os.path.join(folder, filenames[file_index]), 'r') as h5_file:
                pedestal[fidx][dest_sel] = np.mean(h5_file[det['datasets']['adc']][source_sel], axis = 0)
                
        d[name] = {}
        
                
        d[name]['mask'] = mask
        d[name]['pedestal'] = pedestal
        d[name]['gain_constants'] = gain_constants
        
        Log.info("Calibrating roi {}".format(name))
        d[name]['adc'] = calibrate(
            adc = filter_d[name]['adc'],
            gain = filter_d[name]['gain'],
            gain_constants = gain_constants,
            mask = mask,
            pedestal = pedestal,
            max_pixels_non_g0 = det['gain']['max_allowed_non_g0_pixels']
        )
        d[name]['trains'] = filter_d[name]['trains']
        
    # Process veto groups: Only keep data for trains which are not NaN in any of the other ROI
    Log.info("Processing veto groups")
    
    vg = {}
    for idx, roi in enumerate(fconf['roi']):
        if roi['veto_group'] in vg.keys():
            vg[roi['veto_group']].append(idx)
        else:
            vg[roi['veto_group']] = [idx]
            
    veto = np.zeros((len(list(vg.keys())), filter_d['0']['adc'].shape[0]), dtype = bool) + 1
    
    for idx, key in enumerate(vg.keys()):
        for roi in vg[key]:
            veto[idx] = np.logical_and(veto[idx], np.sum(~np.isnan(d[str(roi)]['adc']), axis = (1,2,3)))

        veto[idx] = veto[idx] == False
        
    for idx, roi in enumerate(fconf['roi']):
        veto_mask = np.nonzero(veto[int(roi['veto_group'])])
        d[str(idx)]['adc'][veto_mask] = np.nan
        Log.debug("Total trains vetoed for roi {}: {} of {}".format(idx, np.sum(veto_mask), d[str(idx)]['adc'].shape[0]))
        

    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
