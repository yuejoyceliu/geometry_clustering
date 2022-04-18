import os,sys, shutil
import pandas as pd
from kmeans_traj import cluster_geometry

def checkcommand():
    if len(sys.argv) == 2:
        energy_file = sys.argv[1]
        if not os.path.exists(energy_file):
            raise SystemExit("Error: Not found %s file!" % energy_file)
    else:
        raise SystemExit("Usage: python conformer_selection.py energy_file_name")
    return energy_file

def run_kmeans(energy_file):
    df = pd.read_csv(energy_file)
    filelist = df.iloc[:, 0]
    cluster_labels, hbonds = cluster_geometry(filelist)
    df.insert(1, "group id", cluster_labels)
    df["hydrogen bonds"] = [",".join(hb) for hb in hbonds]
    df["snap_no"] = df.index
    df.sort_values(by=["group id", "snap_no"], inplace=True)
    df.drop(columns=["snap_no"], inplace=True)

    # Select the lowest energetic conformer for each group.
    selected_idx = df.groupby("group id").apply(lambda df: df["energy/(kcal/mol)"].idxmin())
    selected_df = df.ix[selected_idx]
    output = "selected%dconformers.csv" % len(selected_df)
    selected_df.to_csv(output, index=False)
    with open(output, "a") as fo:
        fo.write("all conformers\n")
    df.to_csv(output, index=False, mode="a")
    return selected_df.iloc[:, 0]

def cp_conformers(file_list):
    os.mkdir("selected")
    for fl in file_list:
        shutil.copyfile(fl, "selected/"+fl)

def main():
    energy_file = checkcommand()
    selected_fl = run_kmeans(energy_file)
    cp_conformers(selected_fl)


if __name__ == "__main__":
    main()

    



    
    