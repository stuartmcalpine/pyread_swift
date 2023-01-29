from pyread_swift import SwiftSnapshot
from mpi4py import MPI

"""
Example read script.

If you have downloaded the test snapshot (get_ics.sh) you can run this example
directly. If not, replace the path with the location of your Swift snapshot.

Reads the snapshot in collective mode, i.e., all ranks will read each file at
once. You need parallel-hdf5 installed for this (see installation
instructions).

Each rank will load its own (roughly equal) subset of the particles.
"""

# Dark matter
parttype = 1

comm = MPI.COMM_WORLD

# Initiate the SwiftSnapshot object
swift = SwiftSnapshot("../tests/EAGLE_ICs_6.hdf5", verbose=True, comm=comm,
        mpi_read_format="collective")

# See whats in the header.
print(swift.header)

# Select the region of particles we want to load.
# In this case a 1 Mpc cube with its corner on 2,2,2.
swift.select_region(parttype, 2, 3, 2, 3, 2, 3)

# Split the selection over MPI ranks,
# (Need this even with no MPI).
swift.split_selection()

# Read the data.
coords = swift.read_dataset(parttype, "Coordinates")
print(coords)
