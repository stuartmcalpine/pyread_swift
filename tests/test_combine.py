import os
import numpy as np
import h5py
import pytest
import tempfile
import shutil

# Skip the entire module if parallel h5py or mpi4py aren't available.
try:
    from mpi4py import MPI

    HAS_PARALLEL_HDF5 = h5py.get_config().mpi
except ImportError:
    HAS_PARALLEL_HDF5 = False

pytestmark = pytest.mark.skipif(
    not HAS_PARALLEL_HDF5, reason="Requires mpi4py and parallel HDF5"
)


def _create_snapshot_parts(tmpdir, num_parts=4, num_dm=100, num_boundary=20):
    """
    Create fake multi-part snapshot files for a DMO simulation.

    PartType1 = DM particles, PartType2 = DM boundary particles.
    Particles are split roughly evenly across parts.
    """

    basename = os.path.join(tmpdir, "snap")

    # Split particle counts across file parts.
    dm_per_part = np.zeros(num_parts, dtype=int)
    bd_per_part = np.zeros(num_parts, dtype=int)
    for i in range(num_dm):
        dm_per_part[i % num_parts] += 1
    for i in range(num_boundary):
        bd_per_part[i % num_parts] += 1

    coords_dm_all = np.random.rand(num_dm, 3).astype(np.float64)
    masses_dm_all = np.random.rand(num_dm).astype(np.float32)
    coords_bd_all = np.random.rand(num_boundary, 3).astype(np.float64)
    masses_bd_all = np.random.rand(num_boundary).astype(np.float32)

    dm_offset = 0
    bd_offset = 0
    for i in range(num_parts):
        fname = f"{basename}.{i}.hdf5"
        with h5py.File(fname, "w") as f:
            n_dm = int(dm_per_part[i])
            n_bd = int(bd_per_part[i])

            nparts_this = [0, n_dm, n_bd, 0, 0, 0]

            grp = f.create_group("Header")
            grp.attrs["NumPart_ThisFile"] = nparts_this
            grp.attrs["NumPart_Total"] = [0, num_dm, num_boundary, 0, 0, 0]
            grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
            grp.attrs["NumFilesPerSnapshot"] = num_parts
            grp.attrs["BoxSize"] = 100.0

            if n_dm > 0:
                pt1 = f.create_group("PartType1")
                pt1.create_dataset(
                    "Coordinates", data=coords_dm_all[dm_offset : dm_offset + n_dm]
                )
                pt1.create_dataset(
                    "Masses", data=masses_dm_all[dm_offset : dm_offset + n_dm]
                )
            dm_offset += n_dm

            if n_bd > 0:
                pt2 = f.create_group("PartType2")
                pt2.create_dataset(
                    "Coordinates", data=coords_bd_all[bd_offset : bd_offset + n_bd]
                )
                pt2.create_dataset(
                    "Masses", data=masses_bd_all[bd_offset : bd_offset + n_bd]
                )
            bd_offset += n_bd

    return basename, coords_dm_all, masses_dm_all, coords_bd_all, masses_bd_all


def test_combine_particle_counts():
    """Combined file should have the correct total particle counts."""

    from scripts.combine import combine

    tmpdir = tempfile.mkdtemp()
    try:
        num_dm, num_bd = 100, 20
        basename, *_ = _create_snapshot_parts(
            tmpdir, num_parts=4, num_dm=num_dm, num_boundary=num_bd
        )
        outfile = os.path.join(tmpdir, "combined.hdf5")

        combine(basename, outfile, parttypes=[1, 2])

        with h5py.File(outfile, "r") as f:
            nparts = f["Header"].attrs["NumPart_ThisFile"]
            assert nparts[1] == num_dm
            assert nparts[2] == num_bd
            assert f["Header"].attrs["NumFilesPerSnapshot"] == 1
    finally:
        shutil.rmtree(tmpdir)


def test_combine_data_integrity():
    """All particle data should be present in the combined file (order may differ)."""

    from scripts.combine import combine

    tmpdir = tempfile.mkdtemp()
    try:
        basename, coords_dm, masses_dm, coords_bd, masses_bd = _create_snapshot_parts(
            tmpdir, num_parts=4, num_dm=80, num_boundary=16
        )
        outfile = os.path.join(tmpdir, "combined.hdf5")

        combine(basename, outfile, parttypes=[1, 2])

        with h5py.File(outfile, "r") as f:
            # Check DM particles.
            out_coords_dm = f["PartType1/Coordinates"][:]
            out_masses_dm = f["PartType1/Masses"][:]
            assert out_coords_dm.shape == coords_dm.shape
            assert out_masses_dm.shape == masses_dm.shape

            # Sort both by coordinates to compare regardless of order.
            idx_orig = np.lexsort(coords_dm.T)
            idx_out = np.lexsort(out_coords_dm.T)
            np.testing.assert_allclose(coords_dm[idx_orig], out_coords_dm[idx_out])
            np.testing.assert_allclose(masses_dm[idx_orig], out_masses_dm[idx_out])

            # Check DM boundary particles.
            out_coords_bd = f["PartType2/Coordinates"][:]
            out_masses_bd = f["PartType2/Masses"][:]
            idx_orig = np.lexsort(coords_bd.T)
            idx_out = np.lexsort(out_coords_bd.T)
            np.testing.assert_allclose(coords_bd[idx_orig], out_coords_bd[idx_out])
            np.testing.assert_allclose(masses_bd[idx_orig], out_masses_bd[idx_out])
    finally:
        shutil.rmtree(tmpdir)


def test_combine_single_parttype():
    """Combining with only PartType1 should ignore PartType2."""

    from scripts.combine import combine

    tmpdir = tempfile.mkdtemp()
    try:
        num_dm = 50
        basename, *_ = _create_snapshot_parts(
            tmpdir, num_parts=2, num_dm=num_dm, num_boundary=10
        )
        outfile = os.path.join(tmpdir, "combined.hdf5")

        combine(basename, outfile, parttypes=[1])

        with h5py.File(outfile, "r") as f:
            assert f["Header"].attrs["NumPart_ThisFile"][1] == num_dm
            assert "PartType1" in f
            assert "PartType2" not in f
    finally:
        shutil.rmtree(tmpdir)


def test_combine_empty_parttype():
    """A particle type with zero particles should not create a group."""

    from scripts.combine import combine

    tmpdir = tempfile.mkdtemp()
    try:
        basename, *_ = _create_snapshot_parts(
            tmpdir, num_parts=2, num_dm=50, num_boundary=0
        )
        outfile = os.path.join(tmpdir, "combined.hdf5")

        combine(basename, outfile, parttypes=[1, 2])

        with h5py.File(outfile, "r") as f:
            assert "PartType1" in f
            assert "PartType2" not in f
            assert f["Header"].attrs["NumPart_ThisFile"][2] == 0
    finally:
        shutil.rmtree(tmpdir)
