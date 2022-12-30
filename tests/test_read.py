from read_swift import SwiftSnapshot
import os 
import numpy as np
import pytest

SWIFT_FNAME = "EAGLE_ICs_6.hdf5"
NPARTS = [800648, 830584, 0, 0, 29847, 53]
SKIP_PARTS = [2,3]

def _check_we_have_snapshot():
    exists = os.path.isfile(SWIFT_FNAME)
    if not exists:
        raise FileNotFoundError(
            "You have not downloaded the test snapshot, run 'get_ics.sh' first!"
        )

@pytest.mark.mpi_skip
def test_read():

    # Check we have the example snapshot.
    _check_we_have_snapshot()

    # Swift read object.
    swift = swift_snapshot("EAGLE_ICs_6.hdf5")
    bs = swift.HEADER["BoxSize"]

    for i in range(len(NPARTS)):
        if i in SKIP_PARTS: continue

        swift.select_region(i, 0, bs, 0, bs, 0, bs)
        swift.split_selection()
       
        # Load coordinates.
        coords = swift.read_dataset(i, "Coordinates")
    
        assert coords.dtype == np.float64, f"Bad read parttype {i} (1)"
        assert coords.shape == (NPARTS[i], 3), f"Bad read parttype {i} (2)"
    
@pytest.mark.mpi
def test_read_mpi():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.size > 1

    # Check we have the example snapshot.
    _check_we_have_snapshot()

    # Swift read object.
    swift = swift_snapshot("EAGLE_ICs_6.hdf5", comm=comm)
    bs = swift.HEADER["BoxSize"]

    for i in range(len(NPARTS)):
        if i in SKIP_PARTS: continue

        swift.select_region(i, 0, bs, 0, bs, 0, bs)
        swift.split_selection()
    
        # Load coordinates.
        coords = swift.read_dataset(i, "Coordinates")
   
        assert coords.dtype == np.float64, f"Bad read parttype {i} (1)"

        ntot = len(coords)
        ntot = comm.allreduce(ntot)

        assert ntot == NPARTS[i], f"Bad read parttype {i} (2)"

@pytest.mark.mpi_skip
def test_read_header():
    
    # Check we have the example snapshot.
    _check_we_have_snapshot()

    # Swift read object.
    swift = swift_snapshot("EAGLE_ICs_6.hdf5")

    assert swift.HEADER["BoxSize"].dtype == np.float64
    assert swift.HEADER["BoxSize"] == 4.235625
    assert np.array_equal(swift.HEADER["NumPart_ThisFile"], swift.HEADER["NumPart_Total"])
    for i in range(len(NPARTS)):
        if i in SKIP_PARTS: continue
        assert swift.HEADER["NumPart_ThisFile"][i] == NPARTS[i]
