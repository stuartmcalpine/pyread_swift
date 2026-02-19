from pyread_swift import SwiftLightconeHealpix

"""
Example script for reading a SWIFT lightcone HEALPix shell.

Replace the path below with one of your lightcone HEALPix shell files.
The reader will automatically locate and combine all file parts for that
shell using the file numbering embedded in the filename
(e.g. lightcone0.shell_0001.0.hdf5 -> lightcone0.shell_0001.1.hdf5, ...).
"""

# Path to any one part of the HEALPix shell (e.g. part 0).
fname = "/path/to/lightcone/lightcone0.shell_0001.0.hdf5"

# Initialise the reader. This reads the Shell header.
healpix = SwiftLightconeHealpix(fname)

# Inspect the shell header (e.g. nr_files_per_shell, redshift bounds, nside).
print(healpix.header)

# Read and combine the HEALPix array across all file parts.
# The dataset name depends on your SWIFT configuration, e.g. "DarkMatterMass".
arr = healpix.read_lightcone("DarkMatterMass")

print(f"HEALPix array shape : {arr.shape}")
print(f"HEALPix array dtype : {arr.dtype}")
print(arr)
