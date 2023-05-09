import os
import numpy as np
import h5py

def get_dataset_length(files, h5_path):
    l           = 0
    sequence    = []

    for file in files:
        with h5py.File(file, 'r') as h5_file:
            sh              = h5_file[h5_path].shape
            old_l, new_l    = l, l + sh[0]
            l               = new_l
            sequence.append( [[0, sh[0]], [old_l, new_l]] )

    return l, sequence

def write_dict_to_hdf5(
    d_dict,
    filename,
    file = None,
    path = '',
    override = False
):
    if file is None:
        if os.path.exists(filename) and override == False:
            raise FileExistsError("File already exists and *override* set to False.")

        with h5py.File(filename, 'w') as file:
            write_dict_to_hdf5(d_dict, filename, file, path, override)

    else:
        for key, value in d_dict.items():
            loc_path = '/'.join([path, str(key)])

            if isinstance(value, dict):
                # *value* will become a group in the file
                newGroup = file.create_group(loc_path)
                write_dict_to_hdf5(value, filename, file, loc_path, override)

            else:
                # *value* will become a dataset in the file
                if isinstance(value, str):
                    value = np.string_(value)

                file[path].create_dataset(str(key), data = value)
                

def load_dict_from_hdf5(
    filename,
    group = None,
):
    if group is None:

        with h5py.File(filename, 'r') as file:
            return load_dict_from_hdf5(filename, file)

    else:
        d = {}
        for key, value in group.items():
            
            if isinstance(value, h5py.Group):
                d[key] = load_dict_from_hdf5(filename, value)
            else:
                d[key] = value[()]
            
        return d
        
#         for key, value in d.items():
#             loc_path = '/'.join([path, str(key)])

#             if isinstance(value, dict):
#                 # *value* will become a group in the file
#                 newGroup = file.create_group(loc_path)
#                 write_dict_to_hdf5(value, filename, file, loc_path, override)

#             else:
#                 # *value* will become a dataset in the file
#                 if isinstance(value, str):
#                     value = np.string_(value)

#                 file[path].create_dataset(str(key), data = value)
