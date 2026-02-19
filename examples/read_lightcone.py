from pyread_swift import SwiftParticleLightcone

"""
Example script for reading a SWIFT particle lightcone.

Replace the path and lightcone ID below with your own values.
The reader discovers all file parts in the directory that match
the given lightcone ID and reads them.

Each MPI rank will hold its own subset of the particles. To run
with MPI, see read_lightcone_mpi.py or pass a communicator via
the comm argument.
"""

# Directory containing the lightcone files.
lightcone_dir = "/path/to/lightcone/"

# Which lightcone to read (integer ID used in the filename, e.g. 0 for
# files named lightcone0_XXXX.hdf5).
lightcone_id = 0

# Initialise the reader. This scans the directory and reads the header.
lc = SwiftParticleLightcone(lightcone_dir, lightcone_id, verbose=True)

# Inspect the lightcone header.
print(lc.header)

# Read a dataset. parttype is a string, e.g. "DM" for dark matter.
coords = lc.read_dataset("Coordinates", parttype="DM")

print(f"Coordinates shape : {coords.shape}")
print(f"Coordinates dtype : {coords.dtype}")
print(coords)

# Read another dataset from the same lightcone.
masses = lc.read_dataset("Masses", parttype="DM")
print(f"Masses shape : {masses.shape}")
print(masses)
