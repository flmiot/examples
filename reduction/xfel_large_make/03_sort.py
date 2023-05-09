""" 03_sort: Read ROI data from 02_calibrate.h5 in the same directory and
    sort/integrate according to respective scanned motor positions

    Call as: python 03_sort.py /scratch/directory /config/directory

    Depends on the 03_sort.yml config file.
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

def get_motor_sorted(train_ids, motor_values, min_points, value_threshold):
    
    # Round motor_value to make value differences below *value_threshold* 
    # indistinguishable 
    motor_values = np.around(motor_values, value_threshold)
    values, counts = np.unique(motor_values, return_counts = True)
    
    sorted_values = []

    for v, c in zip(values, counts):
        if c < min_points:
            continue

        sorted_values.append(v)


    return sorted_values


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
        config = yaml.load(file.read(), Loader = yaml.SafeLoader)

    with open(configp('detectors.yml'), 'r') as file:
        detectors = yaml.load(file.read(), Loader = yaml.SafeLoader)

    with open(configp(yml_file_name), 'r') as file:
        sort = yaml.load(file.read(), Loader = yaml.SafeLoader)

    files = [f for f in sorted(os.listdir(config['directory'])) if sort['scan_motor']['da_name'] in f]
    files = [os.path.join(config['directory'], f) for f in files]

    h5_path = sort['scan_motor']['trainId']
    l, sequence = tools.get_dataset_length(files, h5_path)
    motor_trains = np.empty(l)
    motor_values = np.empty(l)

    # Read data into memory
    for idx, file in enumerate(files):
        try:
            with h5py.File(os.path.join(config['directory'], file), 'r') as h5_file:

                Log.info('Reading file: {}'.format(file))

                from_slice_trains  = slice(*sequence[idx][0])
                to_slice_trains    = slice(*sequence[idx][1])

                # Scan motor import
                h5_file[sort['scan_motor']['trainId']].read_direct(
                    motor_trains,
                    source_sel  = from_slice_trains,
                    dest_sel    = to_slice_trains)

                h5_file[sort['scan_motor']['dataset']].read_direct(
                    motor_values,
                    source_sel  = from_slice_trains,
                    dest_sel    = to_slice_trains)
        except Exception as e:
            Log.error(e)

    Log.info("Analyzing scan motor {}".format(sort['scan_motor']['dataset']))
    sorted_values = get_motor_sorted(
        train_ids = motor_trains,
        motor_values = motor_values,
        min_points = sort['filtering']['min_points'],
        value_threshold = sort['filtering']['value_threshold']
    )
    Log.info("Found {} scan points".format(len(sorted_values)))


    d = {}
    calibrate_d = tools.load_dict_from_hdf5(scratchp('02_calibrate.h5'))

    for idx, roi in enumerate(config['roi']):
        name = str(idx)
        d[name] = {}
        d[name]['motor_raw'] = motor_values
        d[name]['motor'] = sorted_values
        shape = calibrate_d[name]['adc'].shape
        d[name]['on'] = np.empty((len(sorted_values), ) + shape[1:])
        d[name]['off'] = np.empty((len(sorted_values), ) + shape[1:])

        norm_roi = str(sort['normalization']['roi'])

        n = calibrate_d[norm_roi]['adc']
        n = n - np.nanmin(n)
        n = np.nansum(n, axis = (1,2,3))
        adc = calibrate_d[name]['adc'] / n[:, None, None, None]
        trains = calibrate_d[name]['trains']
        
        # Align motor and adc trains
        b_adc = np.nonzero(np.in1d(trains, motor_trains))[0]
        b_motor = np.nonzero(np.in1d(motor_trains, trains))[0]
        Log.debug("{} aligned trains (of total {} train for scan motor & {} trains for adc)".format(
            len(b_adc), len(motor_trains), len(trains))) 

        t_adc = adc[b_adc]
        t_trains = trains[b_adc]
        t_motor_values = np.around(motor_values[b_motor], sort['filtering']['value_threshold'])

        for sidx, value in enumerate(sorted_values):
            b = np.nonzero(t_motor_values == value)[0]

            step_adc = t_adc[b]
            step_trains = t_trains[b]

            laser_on = step_adc[step_trains % 2 == 1]
            laser_off = step_adc[step_trains % 2 == 0]

            d[name]['on'][sidx] = np.nanmean(laser_on, axis = 0)
            d[name]['off'][sidx] = np.nanmean(laser_off, axis = 0)

            Log.info("Found {} trains belonging to step {} ({} laser on/ {} laser off)".format(
                step_adc.shape[0], value, laser_on.shape[0], laser_off.shape[0]))

    tools.write_dict_to_hdf5(d, scratchp(hdf_file_name), override = True)
