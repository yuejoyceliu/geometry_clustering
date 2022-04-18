# geometry_clustering
Cluster conformers of the biomolecule

## data_loader.py
- Function data_loader:
  - Input: String of a Gaussian input file name.
  - Output: (distance, hbond). Distance is a numpy 2x2 matrix representing the inter-atomic distances. Hbond is a string list representing all hydrogen bonds in the Gaussian input.
- Function data_preprocess:
  - Input: String list of Gaussian input file names.
  - Output: (data, hbonds). Data only keep the distances that smaller than 2.5 angstrom. Data, with shape of (number-of-files, feature-embedding), is composed of normalized distances and one-hot encoded hydrogen bond information. Hbonds is a list of string list that indicate the hydrogen bonds for each file.

## kmeans_traj.py
- Function cluster_geometry:
  - Input: String list of Gaussian input file names.
  - Descriptions: Get input using data_preprocess function, run kmeans algorithms with different k values ranging from 1 to len(input)//3, find best k value from the  elbow point and cluster the structure using kmeans method with the best k value.
  - Output: (labels, hbonds).

## conformer_selector.py
- Usage: `python conformer_selection.py energy_file_name`
- Descriptions: Cluster structures in the energy file with a best k value and writes out the cluster label, hydrogen bonds and energy to the output file. Copy the lowest energetic conformer of each cluster to the folder named with "selected".
