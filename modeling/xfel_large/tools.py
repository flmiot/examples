import os
import numpy as np
import h5py
from scipy.special import voigt_profile, beta

def lorentzian(x, params):
    return params[:, :,1, np.newaxis]/np.pi * params[:, :,2,np.newaxis] / ( params[:, :,2, np.newaxis]**2 + ( x[np.newaxis, np.newaxis, :] - params[:, :,0, np.newaxis] )**2)


def lorentzian_1d(x, params):
    return params[1]/np.pi * params[2] / ( params[2]**2 + ( x - params[0] )**2)


def split_lorentzian(x, params):
    mu = params[:, :,0,np.newaxis]
    a = params[:, :,1,np.newaxis]
    sigma = params[:, :,2,np.newaxis]
    sigma_r = params[:, :,3,np.newaxis]
    return 2*a/(np.pi*(sigma+sigma_r)) * (  sigma**2 / ((x-mu)**2 + sigma**2) * np.heaviside(mu-x, 1) + sigma_r**2/((x-mu)**2 + sigma_r**2) * np.heaviside(x-mu, 1))


def split_lorentzian_1d(x, params):
    mu = params[0]
    a = params[1]
    sigma = params[2]
    sigma_r = params[3]
    return 2*a/(np.pi*(sigma+sigma_r)) * (  sigma**2 / ((x-mu)**2 + sigma**2) * np.heaviside(mu-x, 1) + sigma_r**2/((x-mu)**2 + sigma_r**2) * np.heaviside(x-mu, 1))


def voigt(x, params):
    return params[:, :,1, np.newaxis] * voigt_profile(x[np.newaxis, np.newaxis, :] - params[:, :,0, np.newaxis], params[:, :,2, np.newaxis], params[:, :,3, np.newaxis])


def voigt_1d(x, params):
    return  params[1] * voigt_profile(x - params[0], params[2], params[3])

def pearson7(x, params):
    mu = params[:, :, 0, np.newaxis]
    a = params[:, :, 1, np.newaxis]
    sigma = params[:, :, 2, np.newaxis]
    m = params[:, :, 3, np.newaxis]
    
    return a / (sigma*beta(m-0.5, 0.5)) * (1+(x-mu)**2/sigma**2)**(-m)

def pearson7_1d(x, params):
    mu = params[0]
    a = params[1]
    sigma = params[2]
    m = params[3]
    
    return a / (sigma*beta(m-0.5, 0.5)) * (1+(x-mu)**2/sigma**2)**(-m)


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
