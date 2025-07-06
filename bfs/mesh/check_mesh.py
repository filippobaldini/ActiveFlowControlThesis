import h5py

mesh_file = "two_jets.h5"
with h5py.File(mesh_file, "r") as f:
    print(f.keys())  # Check the keys in the .h5 file
