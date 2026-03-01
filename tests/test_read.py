from pyread_swift import SwiftSnapshot
import os
import numpy as np
import pytest

SWIFT_FNAME = "EagleSingle.hdf5"
NPARTS = [817075, 830584, 0, 0, 13430, 47, 0]
SKIP_PARTS = [2, 3, 6]

# Centre and radii used for spherical region tests (box is 6.25 Mpc).
SPHERE_CENTRE = [3.125, 3.125, 3.125]
SPHERE_RMAX = 2.0
SHELL_RMIN = 1.5
SHELL_RMAX = 2.0


def _check_we_have_snapshot():
    exists = os.path.isfile(SWIFT_FNAME)
    if not exists:
        raise FileNotFoundError(
            "You have not downloaded the test snapshot, run 'get_ics.sh' first!"
        )


@pytest.mark.parametrize("att", ["Coordinates", "ParticleIDs"])
def test_read(att):

    # Check we have the example snapshot.
    _check_we_have_snapshot()

    # Swift read object.
    swift = SwiftSnapshot(SWIFT_FNAME)
    bs = swift.header["BoxSize"]

    for i in range(len(NPARTS)):
        if i in SKIP_PARTS:
            continue

        swift.select_region(i, 0, bs, 0, bs, 0, bs)
        swift.split_selection()

        # Load coordinates.
        data = swift.read_dataset(i, att)

        if att == "Coordinates":
            assert data.dtype == np.float64, f"Bad read parttype {i} (1)"
            assert data.shape == (NPARTS[i], 3), f"Bad read parttype {i} (2)"
        else:
            assert data.ndim == 1, f"Bad read parttype {i} (1)"
            assert data.shape == (NPARTS[i],), f"Bad read parttype {i} (2)"


def test_read_header():

    # Check we have the example snapshot.
    _check_we_have_snapshot()

    # Swift read object.
    swift = SwiftSnapshot(SWIFT_FNAME)

    assert isinstance(swift.header["BoxSize"], float)
    assert swift.header["BoxSize"] == 6.25
    assert np.array_equal(swift.header["NumPart_ThisFile"], swift.header["NumPart_Total"])
    for i in range(len(NPARTS)):
        if i in SKIP_PARTS:
            continue
        assert swift.header["NumPart_ThisFile"][i] == NPARTS[i]


@pytest.mark.parametrize("att", ["Coordinates", "ParticleIDs"])
def test_select_spherical_region(att):
    """Full sphere selection returns a subset of particles with correct dtype."""

    _check_we_have_snapshot()

    # Dark matter only — present in all cells and simple to reason about.
    parttype = 1
    swift = SwiftSnapshot(SWIFT_FNAME)

    swift.select_spherical_region(parttype, *SPHERE_CENTRE, r_min=0.0, r_max=SPHERE_RMAX)
    swift.split_selection()
    data = swift.read_dataset(parttype, att)

    if att == "Coordinates":
        assert data.dtype == np.float64
        assert data.ndim == 2 and data.shape[1] == 3
    else:
        assert data.ndim == 1

    # Should load some but not all particles (sphere doesn't cover the full box).
    assert 0 < len(data) < NPARTS[parttype]


@pytest.mark.parametrize("att", ["Coordinates", "ParticleIDs"])
def test_select_shell_region(att):
    """Shell selection returns fewer particles than the enclosing sphere."""

    _check_we_have_snapshot()

    parttype = 1
    swift = SwiftSnapshot(SWIFT_FNAME)

    # Full sphere up to SHELL_RMAX.
    swift.select_spherical_region(parttype, *SPHERE_CENTRE, r_min=0.0, r_max=SHELL_RMAX)
    swift.split_selection()
    sphere_data = swift.read_dataset(parttype, att)

    # Shell from SHELL_RMIN to SHELL_RMAX.
    swift.select_spherical_region(parttype, *SPHERE_CENTRE, r_min=SHELL_RMIN, r_max=SHELL_RMAX)
    swift.split_selection()
    shell_data = swift.read_dataset(parttype, att)

    if att == "Coordinates":
        assert shell_data.dtype == np.float64
        assert shell_data.ndim == 2 and shell_data.shape[1] == 3
    else:
        assert shell_data.ndim == 1

    assert len(shell_data) > 0
    # Shell (with inner void removed) should load fewer TL cells than the full sphere.
    assert len(shell_data) < len(sphere_data)
