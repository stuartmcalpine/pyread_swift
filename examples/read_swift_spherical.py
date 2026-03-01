import numpy as np
from pyread_swift import SwiftSnapshot

"""
Example read script using spherical region selection.

If you have downloaded the test snapshot (get_ics.sh) you can run this example
directly. If not, replace the path with the location of your Swift snapshot.

Note: select_spherical_region() selects all top-level cells that overlap the
requested sphere/shell. Cells on the boundary are included in full, so the
returned particles may extend slightly beyond the requested radius. Apply a
distance cut after reading coordinates if an exact sphere is needed.
"""

# Dark matter
parttype = 1

# Initiate the SwiftSnapshot object
swift = SwiftSnapshot("../tests/EagleSingle.hdf5", verbose=True)

# See what's in the header.
print(swift.header)

# --- Full sphere example ---
# Select all particles within 2 Mpc of the centre of the box.
centre = np.array([swift.header["BoxSize"] / 2.0] * 3)
r_max = 2.0

swift.select_spherical_region(parttype, *centre, r_min=0.0, r_max=r_max)
swift.split_selection()
coords = swift.read_dataset(parttype, "Coordinates")

# Post-hoc distance filter for an exact sphere (caller's responsibility).
r2 = np.sum((coords - centre) ** 2, axis=1)
coords = coords[r2 < r_max ** 2]
print(f"Sphere: {len(coords)} particles within r={r_max} Mpc of box centre")

# --- Shell example ---
# Select particles in a shell between 1 and 2 Mpc from the box centre.
r_min, r_max = 1.0, 2.0

swift.select_spherical_region(parttype, *centre, r_min=r_min, r_max=r_max)
swift.split_selection()
coords = swift.read_dataset(parttype, "Coordinates")

# Post-hoc distance filter for an exact shell (caller's responsibility).
r2 = np.sum((coords - centre) ** 2, axis=1)
coords = coords[(r2 >= r_min ** 2) & (r2 < r_max ** 2)]
print(f"Shell: {len(coords)} particles between r={r_min} and r={r_max} Mpc of box centre")
