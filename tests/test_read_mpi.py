from pyread_swift import SwiftSnapshot
import os
import numpy as np
import pytest
from mpi4py import MPI

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


@pytest.mark.mpi
@pytest.mark.parametrize("att", ["Coordinates", "ParticleIDs"])
@pytest.mark.parametrize("mpi_read_format", ["distributed", "collective"])
def test_read_mpi_distributed(att, mpi_read_format, min_size=2):

    comm = MPI.COMM_WORLD
    assert comm.size > 1

    # Check we have the example snapshot.
    _check_we_have_snapshot()

    # Swift read object.
    swift = SwiftSnapshot(SWIFT_FNAME, comm=comm, mpi_read_format=mpi_read_format)
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
        else:
            assert data.ndim == 1, f"Bad read parttype {i} (1)"

        ntot = len(data)
        ntot = comm.allreduce(ntot)

        assert ntot == NPARTS[i], f"Bad read parttype {i} (2)"


@pytest.mark.mpi
@pytest.mark.parametrize("att", ["Coordinates", "ParticleIDs"])
@pytest.mark.parametrize("mpi_read_format", ["distributed", "collective"])
def test_select_spherical_region_mpi(att, mpi_read_format):
    """Full sphere selection returns a subset of particles with correct dtype (MPI)."""

    comm = MPI.COMM_WORLD
    assert comm.size > 1

    _check_we_have_snapshot()

    parttype = 1
    swift = SwiftSnapshot(SWIFT_FNAME, comm=comm, mpi_read_format=mpi_read_format)

    swift.select_spherical_region(parttype, *SPHERE_CENTRE, r_min=0.0, r_max=SPHERE_RMAX)
    swift.split_selection()
    data = swift.read_dataset(parttype, att)

    if att == "Coordinates":
        assert data.dtype == np.float64
        assert data.ndim == 2 and data.shape[1] == 3
    else:
        assert data.ndim == 1

    ntot = comm.allreduce(len(data))
    assert 0 < ntot < NPARTS[parttype]


@pytest.mark.mpi
@pytest.mark.parametrize("mpi_read_format", ["distributed", "collective"])
def test_select_shell_region_mpi(mpi_read_format):
    """Shell selection returns fewer particles than the enclosing sphere (MPI)."""

    comm = MPI.COMM_WORLD
    assert comm.size > 1

    _check_we_have_snapshot()

    parttype = 1
    swift = SwiftSnapshot(SWIFT_FNAME, comm=comm, mpi_read_format=mpi_read_format)

    # Full sphere up to SHELL_RMAX.
    swift.select_spherical_region(parttype, *SPHERE_CENTRE, r_min=0.0, r_max=SHELL_RMAX)
    swift.split_selection()
    sphere_data = swift.read_dataset(parttype, "Coordinates")
    n_sphere = comm.allreduce(len(sphere_data))

    # Shell from SHELL_RMIN to SHELL_RMAX.
    swift.select_spherical_region(parttype, *SPHERE_CENTRE, r_min=SHELL_RMIN, r_max=SHELL_RMAX)
    swift.split_selection()
    shell_data = swift.read_dataset(parttype, "Coordinates")
    n_shell = comm.allreduce(len(shell_data))

    assert n_shell > 0
    assert n_shell < n_sphere
