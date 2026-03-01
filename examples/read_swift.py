from pyread_swift import SwiftSnapshot

"""
Example read script.

If you have downloaded the test snapshot (get_ics.sh) you can run this example
directly. If not, replace the path with the location of your Swift snapshot.
"""

# Dark matter
parttype = 1

# Initiate the SwiftSnapshot object
swift = SwiftSnapshot("../tests/EagleSingle.hdf5", verbose=True)

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
