#!/usr/bin/env python
import re, glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataLoader:

    def __init__(self, filename):
        self.bond_length = 1.2
        self.hbond_length = 2.5
        self.filename = filename
        self.atomlist = []
        self.coordinations = []
        self._load_gaussian_input()
        self.distances = np.zeros((len(self.atomlist), len(self.atomlist)))
        self.hydrogen_bonds = []
    
    def _load_gaussian_input(self):
        """Finds the atomlist and xyz coordinations of all atoms in the Gaussian input file.
        """
        with open(self.filename, 'r') as fo:
            lines = fo.readlines()
        pattern = re.compile(r'[A-Z][a-z]{,2}(\s*[\s|,|\\]+\s*[-]?\d+\.\d*){3}')
        for line in lines:
            line = line.strip()
            if bool(pattern.match(line)):
                xyz = re.split(r'\s*[\s|,|\\]+\s*', line)
                self.atomlist.append(xyz[0])
                self.coordinations.append([float(x) for x in xyz[1:]])
    
    def compute_distances(self):
        """Computes all distances in angstrom among all atoms.
        """
        for i, (x1, y1, z1) in enumerate(self.coordinations):
            for j, (x2, y2, z2) in enumerate(self.coordinations):
                self.distances[i][j] = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
    
    def find_hydrogen_bonds(self):
        """Returns all hydrogen bonds
        """
        NOF_indices = [i for i in range(len(self.atomlist)) if self.atomlist[i] in ['N', 'O', 'F']]
        for i, atom in enumerate(self.atomlist):
            if atom == 'H':
                bonded_atom_indices = []
                hbonded_atom_indices = []
                for j in NOF_indices:
                    if self.distances[i][j] <= self.bond_length:
                        bonded_atom_indices.append(j)
                    elif self.distances[i][j] <= self.hbond_length:
                        hbonded_atom_indices.append(j)
                for bonded_index in bonded_atom_indices:
                    for hbonded_index in hbonded_atom_indices:
                        self.hydrogen_bonds.append("{}{}-H{}-{}{}".format(self.atomlist[bonded_index],
                        bonded_index+1, i+1, self.atomlist[hbonded_index], hbonded_index+1))

    def find_nonxch_hydrogen(self):
        carbon_hydrogen_idx = []
        for i, atom in enumerate(self.atomlist):
            if atom == 'H':
                self.distances[i, i] = float("inf") 
                if self.atomlist[np.argmin(self.distances[i])] == 'C':
                    carbon_hydrogen_idx.append(i)
                self.distances[i, i] = 0
        return carbon_hydrogen_idx

def data_loader(filename):
    dataloader = DataLoader(filename)
    dataloader.compute_distances()
    dataloader.find_hydrogen_bonds()
    return dataloader

def data_preprocess(file_list):
    numerical_data = None
    categorical_data = []
    hydrogen_bonds = dict()
    carbon_hydrogen_idx = None

    for file in file_list:
        loader = data_loader(file)
        # Remove distances of carbon-hydrogens.
        if carbon_hydrogen_idx is None:
            carbon_hydrogen_idx = loader.find_nonxch_hydrogen()
            print("Carbon-hydrogen index:")
            print([idx+1 for idx in carbon_hydrogen_idx])
        distances = np.delete(loader.distances, carbon_hydrogen_idx, axis=0)
        distances = np.delete(distances, carbon_hydrogen_idx, axis=1)
        # Get the upper triangular distance matrix.
        distances = distances[np.triu_indices(len(distances))]
        distances = np.reshape(distances, (1, -1))
        if numerical_data is None:
            numerical_data = distances
        else:
            numerical_data = np.concatenate([numerical_data, distances])
        for hb in loader.hydrogen_bonds:
            if hb not in hydrogen_bonds:
                hydrogen_bonds[hb] = len(hydrogen_bonds)
        categorical_data.append(loader.hydrogen_bonds)
    onehot_data = np.zeros((len(categorical_data), len(hydrogen_bonds)))
    for i, hbonds in enumerate(categorical_data):
        for hb in hbonds:
            onehot_data[i][hydrogen_bonds[hb]] = 1
    # Filter out distances >= 2.5.
    min_distances = np.min(numerical_data, axis=0)
    numerical_data = numerical_data[:, min_distances<2.5]
    numerical_data = MinMaxScaler().fit_transform(numerical_data)
    print("%d distances are used." % np.shape(numerical_data)[1])
    print("%d different hydrogen bonds found in %d conformers." % (len(hydrogen_bonds), len(file_list)))
    data = np.concatenate([numerical_data, onehot_data], axis=1)
    return data, categorical_data

if __name__=='__main__':
    data, _ = data_preprocess(glob.glob("*.gjf"))
