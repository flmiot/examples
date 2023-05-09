"""
HDFpreview logic module. Container file for all non-GUI-related program logic:
    - Reading HDF5 file
    - Extracting data from HDF5 files
    - Getting a tree of the file structure as dict
"""

import os
import h5py
import numpy as np

class Dataset(object):

    """
    Container class to hold a number of HDF5 files and functionality to extract
    data from them.
    """

    def __init__(self, files):
        """Specifiy a number of *files* as a list of string filenames."""
        self.files = files
        self.keys  = dict()


    def getData(self, path):
        """Extract data from this dataset's HDF5 files at data *path*."""
        with h5py.File(self.files[0], 'r') as file:
            shape = list()
            shape.extend(file[path].shape)
            dtype = file[path].dtype

        for ind, file in enumerate(self.files):
            if ind == 0:
                continue

            with h5py.File(file, 'r') as file:
                shape[0] += file[path].shape[0]

        data = np.empty(shape, dtype)

        index = 0
        for file in self.files:
            with h5py.File(file, 'r') as file:
                d = file[path]
                d.read_direct(data, None, np.s_[index : index + d.shape[0]])
                index += d.shape[0]

        return data


    def get_filenames(self):
        """Return the path-stripped filenames in this dataset."""
        return [os.path.split(f)[1] for f in self.files]


    def get_data_tree(self):
        """Return the datastructure of files in this dataset as nested dict."""
        for filename in self.files:
            with h5py.File(filename, 'r') as file:
                groups = list(file.values())
                for g in groups:
                    name = g.name
                    g.visititems(lambda n, i : self.fill_dict(name, n, i))

        return self.keys


    def fill_dict(self, group, name, item):
        """Callback function to be called from *get_data_tree*."""
        if isinstance(item, h5py.Group):
            self.add_node(self.keys, item.name)
        elif isinstance(item, h5py.Dataset):
            self.add_entry(self.keys, item.name, item.name)


    def add_node(self, dictionary, path):
        """
        Helper function to add a new node (parent for a group of subentries) to
        a nested *dictionary* of datasources. Calls itself recursivly to add the
        whole branch along *path* to the datasource *dictionary*.
        """

        segments = path.split('/')
        if len(segments) == 1:                      # Reached end of datapath
            dictionary[segments[0]] = dict()
        else:                                       # Path goes on...
            key = segments.pop(0)
            if key == '':
                key = segments.pop(0)
            path    = '/'.join(segments)
            if not key in dictionary.keys():
                dictionary[key] = dict()
            self.add_node(dictionary[key], path)    # Call itself with new path


    def add_entry(self, dictionary, path, value):
        """
        Helper function to add a new entry (tree leaf, selectable data source)
        to a nested *dictionary* of datasources. Calls itself recursivly to add the
        whole branch along *path* to the datasource *dictionary*.
        """

        segments = path.split('/')
        if len(segments) == 1:
            dictionary[segments[0]] = value
        else:
            key = segments.pop(0)
            if key == '':
                key = segments.pop(0)
            path    = '/'.join(segments)
            if not key in dictionary.keys():
                dictionary[key] = dict()
            self.add_entry(dictionary[key], path, value)
