from pyread_swift import SwiftSnapshot
from mpi4py import MPI

"""
Example read script.

If you have downloaded the test snapshot (get_ics.sh) you can run this example
directly. If not, replace the path with the location of your Swift snapshot.
Note distributed MPI reading doesn't make sense for a single file, but this is
just an example. If you have more ranks than files, some cores will load
nothing.

Reads the snapshot in distributed mode, i.e., ranks will read their own files
separately (no collective reading). 

Each rank will load its own subset of the particles, though likely unbalanced
depending on the particle distribution in the snapshot parts.
"""

comm = MPI.COMM_WORLD

# Dark matter
parttype = 1

# Initiate the SwiftSnapshot object
swift = SwiftSnapshot("../tests/EAGLE_ICs_6.hdf5", verbose=True, comm=comm,
        mpi_read_format="distributed")

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
