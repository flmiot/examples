""" 01_filter: Extract raw ROI data from hdf5 files

    Call as: python 01_filter.py /scratch/directory /config/directory

    Depends on the 01_filter.yml config file.
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
        da = detectors[roi['detector']]['da_name']
        b = config['directory']
        files = [os.path.join(b, f) for f in sorted(os.listdir(b)) if da in f]
        h5_path = detectors[roi['detector']]['datasets']['trainId']

        l, sequence = tools.get_dataset_length(files, h5_path)
        y0, y1 = roi['dim0']
        x0, x1 = roi['dim1']
        shape = (l, 1, y1-y0, x1-x0)

        name = str(idx)
        d[name] = {}
        d[name]['adc'] = np.empty(shape, dtype = np.int16)
        d[name]['gain'] = np.empty(shape, dtype = np.int8)
        d[name]['trains'] = np.empty(l)
        d[name]['timestamp'] = np.empty(l)

        for idx, file in enumerate(files):
            try:
                with h5py.File(file, 'r') as h5_file:
                    Log.info('Reading file: {}'.format(file))

                    from_slice_trains  = slice(*sequence[idx][0])
                    to_slice_trains    = slice(*sequence[idx][1])

                    h5_file[
                        detectors[roi['detector']]['datasets']['trainId']
                    ].read_direct(
                        d[name]['trains'],
                        source_sel  = from_slice_trains,
                        dest_sel    = to_slice_trains)

                    h5_file[
                        detectors[roi['detector']]['datasets']['timestamp']
                    ].read_direct(
                        d[name]['timestamp'],
                        source_sel  = from_slice_trains,
                        dest_sel    = to_slice_trains)

                    from_slice = (from_slice_trains, slice(0,1), slice(y0, y1), slice(x0, x1))
                    to_slice   = (to_slice_trains, slice(0,1), slice(0, y1-y0), slice(0, x1-x0))

                    h5_file[
                        detectors[roi['detector']]['datasets']['adc']
                    ].read_direct(
                        d[name]['adc'],
                        source_sel  = from_slice,
                        dest_sel    = to_slice)

                    h5_file[
                        detectors[roi['detector']]['datasets']['gain']
                    ].read_direct(
                        d[name]['gain'],
                        source_sel  = from_slice,
                        dest_sel    = to_slice)

            except Exception as e:
                Log.error(e)

    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
