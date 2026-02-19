from pyread_swift import SwiftParticleLightcone
from mpi4py import MPI

"""
Example MPI script for reading a SWIFT particle lightcone.

Run with, e.g.:
    mpirun -n 4 python read_lightcone_mpi.py

Replace the path and lightcone ID below with your own values.

Each rank reads a strided subset of the lightcone files
(rank reads files where file_index % num_ranks == rank), so no
parallel HDF5 is required. Each rank holds its own local portion
of the returned arrays.
"""

comm = MPI.COMM_WORLD

# Directory containing the lightcone files.
lightcone_dir = "/path/to/lightcone/"

# Which lightcone to read (integer ID used in the filename, e.g. 0 for
# files named lightcone0_XXXX.hdf5).
lightcone_id = 0

# Initialise the reader. Rank 0 reads the header and broadcasts it to
# all other ranks.
lc = SwiftParticleLightcone(lightcone_dir, lightcone_id, verbose=True, comm=comm)

# Inspect the lightcone header (same on all ranks after the broadcast).
if comm.Get_rank() == 0:
    print(lc.header)

# Read a dataset. Each rank reads its own subset of files and returns
# its local portion of the array.
coords = lc.read_dataset("Coordinates", parttype="DM")

print(f"[Rank {comm.Get_rank()}] Coordinates shape : {coords.shape}")
