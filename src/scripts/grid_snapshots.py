import h5py
from pyread_swift import SwiftSnapshot
import numpy as np
import sys
import time
from mpi4py import MPI
try:
    import virgo.mpi.parallel_sort as ps
except ImportError:
    pass

# MPI stuff.
comm = MPI.COMM_WORLD


class grid_particles:
    def __init__(
        self,
        fname,
        save_fname,
        region_min,
        region_max,
        do_checks=False,
        include_vels=False,
    ):
        """
        Add a "Cells" group to a SWIFT snapshot, gridding the particles so that
        they can be read with select region.

        Currently this only works for PartType1.

        Run using `pyread_swift grid`, see `pyread_swift grid --help`.

        This will make a new snapshot (`save_fname`) with only [coords, vdisps,
        hsmls, pids]. We need a new snapshot because the particles have to be
        rearranged by their cellids.

        Parameters
        ----------
        fname : str
            Path to snapshot to add grid to
        save_fname : str
            Output snapshot
        region_min/region_max : float
            For zooms, can restrict the extent of the grid (for higher
            resolution). By default the grid is done over the whole box.
        do_checks : bool
            Do extra validation checks
        include_vels : bool
            Include velocities in the output
        """

        assert comm.size > 1, "Need to run with at least 2 ranks"

        self.nbins = 64
        self.region_min = region_min
        self.region_max = region_max
        self.do_checks = do_checks
        self.include_vels = include_vels

        self.fname = fname
        self.save_fname = save_fname

        # Run.
        tic = time.time()
        self.load_particles()
        print(f"[{comm.rank}] Loading particles took {time.time()-tic}")
        tic = time.time()
        self.make_cell_tree()
        print(f"[{comm.rank}] Make cell tree took {time.time()-tic}")
        tic = time.time()
        self.bin_particles()
        print(f"[{comm.rank}] Binning took {time.time()-tic}")
        tic = time.time()
        self.sort_particles()
        print(f"[{comm.rank}] Sorting took {time.time()-tic}")
        tic = time.time()
        self.save()
        print(f"[{comm.rank}] Saving took {time.time()-tic}")

    def check_particle_pos(self, coords, cell_pos, width):
        """
        Check if all coords assigned to a cell, are actually in that cell.

        Parameters
        ----------
        coords : ndarray
            Subset of coords that have been assigned to a cell
        cell_pos : array
            Left corner of cell coords should belong to
        width : float
            Width of cell
        """

        print(f"Checking {len(coords)} particles that are meant to be in cell..")
        print(f"{cell_pos[0]}->{cell_pos[0]+width}")
        print(f"{cell_pos[1]}->{cell_pos[1]+width}")
        print(f"{cell_pos[2]}->{cell_pos[2]+width}")
        assert np.all(coords[:, 0] >= cell_pos[0])
        assert np.all(coords[:, 0] < cell_pos[0] + width)
        assert np.all(coords[:, 1] >= cell_pos[1])
        assert np.all(coords[:, 1] < cell_pos[1] + width)
        assert np.all(coords[:, 2] >= cell_pos[2])
        assert np.all(coords[:, 2] < cell_pos[2] + width)
        print("All good")

    def load_particles(self):
        """
        Load particle from snapshot.

        Only loads coordinates if we are doing it in place.
        """

        swift = SwiftSnapshot(self.fname, comm=comm)

        if swift.header["NumFilesPerSnapshot"] > 1:
            raise Exception("Only works on single file snapshots")

        bs = swift.header["BoxSize"]

        swift.select_region(1, 0, bs, 0, bs, 0, bs)
        swift.split_selection()

        self.data = {}
        if comm.rank == 0:
            print("Loading coords...")
        self.data["coords"] = swift.read_dataset(1, "Coordinates")
        try:
            if comm.rank == 0:
                print("Loading hsmls...")
            self.data["hsmls"] = swift.read_dataset(1, "SmoothingLengths")
        except:
            if comm.rank == 0:
                print("No hsmls skipping...")
        try:
            if comm.rank == 0:
                print("Loading vdisps...")
            self.data["vdisps"] = swift.read_dataset(1, "VelocityDispersions")
        except:
            if comm.rank == 0:
                print("No vdisps skipping...")
        if comm.rank == 0:
            print("Loading pids...")
        self.data["pids"] = swift.read_dataset(1, "ParticleIDs")
        if self.include_vels:
            if comm.rank == 0:
                print("Loading vels...")
            self.data["vels"] = swift.read_dataset(1, "Velocities")

        # The minimum and maximum bounds of the gridded area.
        # Generally this will always be the whole box, but for zooms you can
        # restrict it to get better cell resolution.
        if self.region_min is None:
            self.region_min = 0.0
        if self.region_max is None:
            self.region_max = bs
        self.region_size = self.region_max - self.region_min
        self.cell_width = np.true_divide(self.region_size, self.nbins)

    def make_cell_tree(self):
        """Create the grid structure."""

        if comm.rank == 0:
            # Make the cell tree.
            cellid_list = []
            cell_pos_list = []

            for i in range(self.nbins):
                for j in range(self.nbins):
                    for k in range(self.nbins):
                        cell_pos_list.append(
                            [
                                i * self.cell_width + self.region_min,
                                j * self.cell_width + self.region_min,
                                k * self.cell_width + self.region_min,
                            ]
                        )
                        cellid_list.append(k + self.nbins * (j + self.nbins * i))

            self.cellid_list = np.array(cellid_list, dtype="i4")
            self.cell_pos_list = np.vstack(cell_pos_list)
            self.cell_pos_list_centers = self.cell_pos_list + self.cell_width / 2.0
            assert np.all(self.cellid_list >= 0) and np.all(
                self.cellid_list < self.nbins**3
            ), "Cellid error"
        else:
            self.cellid_list = None
            self.cell_pos_list = None
            self.cell_pos_list_centers = None

        if comm.size > 1:
            self.cellid_list = comm.bcast(self.cellid_list)
            self.cell_pos_list = comm.bcast(self.cell_pos_list)
            self.cell_pos_list_centers = comm.bcast(self.cell_pos_list_centers)

    def bin_particles(self):
        """Bin the particles into their grid cells."""

        # Bin the particles
        bins = np.arange(
            self.region_min, self.region_max + self.cell_width, self.cell_width
        )

        i = np.digitize(self.data["coords"][:, 0], bins) - 1
        if self.do_checks:
            assert np.all(i) < self.nbins and np.all(i) >= 0
        j = np.digitize(self.data["coords"][:, 1], bins) - 1
        if self.do_checks:
            assert np.all(j) < self.nbins and np.all(j) >= 0
        k = np.digitize(self.data["coords"][:, 2], bins) - 1
        if self.do_checks:
            assert np.all(k) < self.nbins and np.all(k) >= 0

        self.data["cellid"] = np.array(
            k + self.nbins * (j + self.nbins * i), dtype="i4"
        )

        assert np.all(self.data["cellid"] >= 0) and np.all(
            self.data["cellid"] < self.nbins**3
        ), "Cellid error"

        # Check they went into the right bins.
        if self.do_checks:
            u_ids = np.unique(self.data["cellid"])
            for i, this_id in enumerate(u_ids):
                mask = np.where(self.cellid_list == this_id)[0]
                if len(mask) == 0:
                    raise Exception(f"Why not find cell {this_id}")
                mask2 = np.where(self.data["cellid"] == this_id)[0]
                if len(mask2) == 0:
                    raise Exception("Why not find any particles")
                self.check_particle_pos(
                    self.data["coords"][mask2],
                    self.cell_pos_list[mask][0],
                    self.cell_width,
                )

    def sort_particles(self):
        """
        Sort the particles by their cellid.

        So we can count the particles in each cell and compute their offset in
        the file.
        """

        # Sort the particles
        idx = ps.parallel_sort(self.data["cellid"], comm, return_index=True)

        for att in self.data.keys():
            if att == "cellid":
                continue
            self.data[att] = ps.fetch_elements(self.data[att], idx, comm=comm)

        if self.do_checks:
            assert np.array_equal(np.sort(self.data["cellid"]), self.data["cellid"])

        cell_ids = np.unique(self.data["cellid"])
        print(f"Rank {comm.rank} has {len(cell_ids)} cells with particles.")

        l = np.searchsorted(self.data["cellid"], cell_ids, side="left")
        r = np.searchsorted(self.data["cellid"], cell_ids, side="right")

        num = r - l
        assert np.sum(num) == len(self.data["cellid"]), "Dont add up"

        tot_per_rank = comm.allgather(len(self.data["cellid"]))
        offset_per_rank = np.cumsum(tot_per_rank) - tot_per_rank

        l += offset_per_rank[comm.rank]
        r += offset_per_rank[comm.rank]

        lefts = comm.gather(l)
        rights = comm.gather(r)
        c_list = comm.gather(cell_ids)

        if comm.rank == 0:
            lefts = np.concatenate(lefts)
            rights = np.concatenate(rights)
            c_list = np.concatenate(c_list)

            idx = np.lexsort((lefts, c_list))

            lefts = lefts[idx]
            rights = rights[idx]
            c_list = c_list[idx]

            assert np.array_equal(c_list, np.sort(c_list)), "cell list not sorted"
            assert len(lefts) == len(rights) == len(c_list), "Bad length"

            l = np.searchsorted(c_list, np.unique(c_list), side="left")
            r = np.searchsorted(c_list, np.unique(c_list), side="right")

            lefts = lefts[l]
            rights = rights[r - 1]

            num = rights - lefts
            assert np.sum(num) == np.sum(tot_per_rank), "Dont add up 2"

            self.cell_counts = np.zeros(len(self.cell_pos_list_centers), dtype="i8")
            mask = np.in1d(self.cellid_list, c_list)
            self.cell_counts[mask] = num
            self.lefts = np.zeros(len(self.cell_pos_list_centers), dtype="i8")
            self.lefts[mask] = lefts

    def save(self):
        """Save new snapshot to disk."""

        ntot = comm.allreduce(len(self.data["pids"]))

        comm.barrier()

        if comm.rank == 0:
            f = h5py.File(self.save_fname, "w")

            # Cell list
            g = f.create_group("Cells")

            gg = g.create_group("Counts")
            gg.create_dataset("PartType1", data=self.cell_counts)

            gg = g.create_group("OffsetsInFile")
            gg.create_dataset("PartType1", data=self.lefts)

            gg = g.create_group("Files")
            gg.create_dataset(
                "PartType1", data=np.zeros(len(self.cell_counts), dtype="i4")
            )

            g.create_dataset("Centres", data=self.cell_pos_list_centers)

            gg = g.create_group("Meta-data")
            gg.attrs.create("dimension", [self.nbins, self.nbins, self.nbins])
            gg.attrs.create("nr_cells", self.nbins**3)
            gg.attrs.create("size", [self.cell_width, self.cell_width, self.cell_width])

            # Header info.
            oldf = h5py.File(self.fname, "r")
            if "Cosmology" in oldf:
                f.copy(oldf["Cosmology"], f)
            if "Parameters" in oldf:
                f.copy(oldf["Parameters"], f)
            if "Header" in oldf:
                f.copy(oldf["Header"], f)
            oldf.close()

            if "Header" in f:
                f["Header"].attrs["NumFilesPerSnapshot"] = 1
                f["Header"].attrs["NumPart_ThisFile"] = f["Header"].attrs[
                    "NumPart_Total"
                ]
                f["Header"].attrs["CoordinatesOffset"] = [0.0, 0.0, 0.0]

            f.close()

        comm.barrier()

        # Make a new communicator for tasks with data only due to a h5py bug
        io_comm = comm.Split((0 if len(self.data["pids"]) > 0 else 1), comm.rank)

        if len(self.data["pids"]) > 0:
            f = h5py.File(self.save_fname, "a", driver="mpio", comm=io_comm)
            g = f.create_group("PartType1")

            self.write_collective_dataset(
                g, "ParticleIDs", (ntot,), np.int64, self.data["pids"], io_comm
            )
            self.write_collective_dataset(
                g, "Coordinates", (ntot, 3), np.float32, self.data["coords"], io_comm
            )
            if "hsmls" in self.data.keys():
                self.write_collective_dataset(
                    g,
                    "SmoothingLengths",
                    (ntot,),
                    np.float32,
                    self.data["hsmls"],
                    io_comm,
                )
            if "vdisps" in self.data.keys():
                self.write_collective_dataset(
                    g,
                    "VelocityDispersions",
                    (ntot,),
                    np.float32,
                    self.data["vdisps"],
                    io_comm,
                )
            if self.include_vels:
                self.write_collective_dataset(
                    g, "Velocities", (ntot, 3), np.float32, self.data["vels"], io_comm
                )

            f.close()

        io_comm.Free()

    def write_collective_dataset(self, g, name, shape, dtype, data, io_comm):
        # Determine how many elements to write on each task
        num_on_task = np.asarray(io_comm.allgather(len(self.data["cellid"])))
        ntot = np.sum(num_on_task)

        # Determine offsets at which to write data from each task
        offset_on_task = np.cumsum(num_on_task) - num_on_task

        dataset = g.create_dataset(name, shape=shape, dtype=dtype)

        left = offset_on_task[io_comm.rank]
        right = offset_on_task[io_comm.rank] + num_on_task[io_comm.rank]

        # How many chunks will we right over.
        max_write = 2 * 1024.0 * 1024 * 1024  # 2 Gb in bytes
        byte_size = np.dtype(dtype).itemsize
        if len(shape) > 1:
            byte_size *= shape[1]
        num_chunks = int(np.ceil((num_on_task[io_comm.rank] * byte_size) / max_write))

        # Work out slices.
        num_per_cycle = np.tile(num_on_task[io_comm.rank] // num_chunks, num_chunks)
        mini_lefts = np.cumsum(num_per_cycle) - num_per_cycle[0]
        mini_rights = np.cumsum(num_per_cycle)
        num_per_cycle[-1] += num_on_task[io_comm.rank] % num_chunks
        mini_rights[-1] += num_on_task[io_comm.rank] % num_chunks
        assert (
            np.sum(mini_rights - mini_lefts) == num_on_task[io_comm.rank]
        ), "Minis dont add up"

        count = 0
        for i in range(num_chunks):
            this_l = left + mini_lefts[i]
            this_r = left + mini_rights[i]
            count += this_r - this_l

            slice_to_write = np.s_[this_l:this_r, ...]

            with dataset.collective:
                dataset[slice_to_write] = data[mini_lefts[i] : mini_rights[i]]

        assert count == num_on_task[io_comm.rank], "Final count not add up"
