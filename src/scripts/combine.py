import os
import numpy as np
import sys
import h5py

from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.rank
comm_size = MPI.COMM_WORLD.size
comm = MPI.COMM_WORLD


class combine:
    def __init__(self, fname, outfile, parttypes=[1, 2]):
        """
        Combine multiple snapshot parts into a single snapshot file.

        For DMO (dark matter only) simulations where PartType1 is the main DM
        particles and PartType2 is the DM boundary particles.

        File parts are distributed round-robin across MPI ranks. Uses parallel
        collective HDF5 I/O (mpio driver), so compression is not supported.

        Run using ``pyread_swift combine``, see ``pyread_swift combine --help``.

        Parameters
        ----------
        fname : str
            Path to snapshot parts (excluding the ``.x.hdf5`` suffix)
        outfile : str
            Path to output combined snapshot
        parttypes : list
            Particle types to combine (default: [1, 2] for DM + DM boundary)
        """

        if isinstance(parttypes, int):
            parttypes = [parttypes]
        self.parttypes = parttypes
        self.fname = fname
        self.outfile = outfile

        self.read_header()

        lefts_dict, ntot_dict = self.index_files()

        self.create_new_file(ntot_dict)
        self.create_datasets(ntot_dict)
        self.copy_particles(ntot_dict, lefts_dict)

        self.new_f.close()

    def copy_particles(self, ntot_dict, lefts_dict, max_write_num=170000000):
        """
        Copy particles from each snapshot part into the combined file.

        Each rank processes its assigned file parts (round-robin). Large writes
        are split into chunks of ``max_write_num`` to manage memory.

        Parameters
        ----------
        ntot_dict : dict
            Total particle count per particle type
        lefts_dict : dict
            Global write offset for each rank per particle type
        max_write_num : int
            Max particles to write in a single operation (default: 170M)
        """

        # Track how many particles this rank has written so far per type,
        # so we can offset correctly when processing multiple file parts.
        particles_processed = {parttype: 0 for parttype in self.parttypes}

        for j in range(self.num_files_per_snapshot):
            if j % comm_size != comm_rank:
                continue

            this_fname = self.fname + ".%i.hdf5" % j
            if not os.path.isfile(this_fname):
                raise ValueError(f"Snapshot part not found: {this_fname}")

            this_f = h5py.File(this_fname, "r")

            for parttype in self.parttypes:
                parttype_group = f"PartType{parttype}"
                if parttype_group not in this_f or self.attribute_list[parttype] is None:
                    continue

                this_num = this_f["Header"].attrs["NumPart_ThisFile"][parttype]

                if this_num > 0:
                    # Global write position = this rank's base offset + particles
                    # already written by this rank for this type.
                    tmp_left = lefts_dict[parttype][comm_rank] + particles_processed[parttype]
                    tmp_right = tmp_left + this_num
                    particles_processed[parttype] += this_num

                    print(
                        f"[Rank {comm_rank}] PartType{parttype}: writing {this_num} "
                        f"particles from file {j} to [{tmp_left}:{tmp_right}]"
                    )

                    # Split into chunks to avoid writing too much at once.
                    num_chunks = int(np.ceil(this_num / max_write_num))
                    num_per_cycle = np.tile(this_num // num_chunks, num_chunks)
                    mini_lefts = np.cumsum(num_per_cycle) - num_per_cycle[0]
                    mini_rights = np.cumsum(num_per_cycle)
                    num_per_cycle[-1] += this_num % num_chunks
                    mini_rights[-1] += this_num % num_chunks
                    assert np.sum(mini_rights - mini_lefts) == this_num, (
                        f"Chunk sizes don't sum to total: "
                        f"{np.sum(mini_rights - mini_lefts)} != {this_num}"
                    )

                    # Copy each attribute in chunks.
                    for att in self.attribute_list[parttype].keys():
                        num_read = 0
                        for i in range(num_chunks):
                            this_l_read = mini_lefts[i]
                            this_r_read = mini_rights[i]

                            this_l_write = tmp_left + mini_lefts[i]
                            this_r_write = this_l_write + (mini_rights[i] - mini_lefts[i])

                            tmp_data = this_f[f"{parttype_group}/{att}"][
                                this_l_read:this_r_read
                            ]
                            num_read += len(tmp_data)
                            with self.new_f[
                                f"{parttype_group}/{att}"
                            ].collective:
                                self.new_f[f"{parttype_group}/{att}"][
                                    this_l_write:this_r_write
                                ] = tmp_data

                        assert tmp_right - tmp_left == num_read, "Indexing error"

            this_f.close()
            print(
                "[Rank %i] Done %i/%i" % (comm_rank, j + 1, self.num_files_per_snapshot)
            )

    def create_datasets(self, ntot_dict):
        """Create empty datasets in the output file for each particle type."""

        for parttype in self.parttypes:
            ntot = ntot_dict[parttype]
            if ntot > 0 and self.attribute_list[parttype] is not None:
                parttype_group = f"PartType{parttype}"
                grp = self.new_f.create_group(parttype_group)

                for att in self.attribute_list[parttype].keys():
                    shape = self.attribute_list[parttype][att][1]
                    if len(shape) == 1:
                        grp.create_dataset(att, (ntot,), dtype=self.attribute_list[parttype][att][0])
                    else:
                        grp.create_dataset(att, (ntot, shape[1]), dtype=self.attribute_list[parttype][att][0])

    def create_new_file(self, ntot_dict):
        """
        Create the output HDF5 file with header metadata.

        Opens the file with the MPI-IO driver for parallel writes. Copies
        Header, Cosmology, and Parameters groups from the source, updating
        particle counts to reflect the combined totals.
        """

        self.new_f = h5py.File(self.outfile, "w", driver="mpio", comm=comm)
        self.new_f.atomic = True
        if comm_rank == 0:
            print(f"Creating {self.outfile}...")

        grp = self.new_f.create_group("Header")

        # Build combined particle count arrays (6 SWIFT particle types).
        ntots = np.zeros(6, dtype=np.int64)
        highwords = np.zeros(6, dtype=np.int64)
        num_this_file = np.zeros(6, dtype=np.int64)

        for parttype in self.parttypes:
            ntot = int(ntot_dict[parttype])
            ntots[parttype] = ntot
            highwords[parttype] = 0
            num_this_file[parttype] = ntot

        # Write header attributes, overriding count-related fields.
        if self.header is not None:
            for att in self.header.keys():
                if att == "NumPart_Total":
                    grp.attrs.create(att, ntots)
                elif att == "NumPart_Total_HighWord":
                    grp.attrs.create(att, highwords)
                elif att == "NumFilesPerSnapshot":
                    grp.attrs.create(att, 1)
                elif att == "NumPart_ThisFile":
                    grp.attrs.create(att, num_this_file)
                else:
                    grp.attrs.create(att, self.header[att])

        if self.cosmo is not None:
            grp = self.new_f.create_group("Cosmology")
            for att in self.cosmo.keys():
                grp.attrs.create(att, self.cosmo[att])

        if self.params is not None:
            grp = self.new_f.create_group("Parameters")
            for att in self.params.keys():
                if att == "Snapshots:distributed":
                    grp.attrs.create(att, 0)
                else:
                    grp.attrs.create(att, self.params[att])

    def index_files(self):
        """
        Count particles per rank and compute global write offsets.

        Each rank opens its assigned file parts (round-robin) and tallies
        particle counts. An MPI allgather + cumulative sum determines the
        global starting write position for each rank per particle type.

        Returns
        -------
        lefts_dict : dict
            Global write offset per rank per particle type
        ntot_dict : dict
            Total particle count per particle type
        """

        ntot_dict = {parttype: 0 for parttype in self.parttypes}
        ntot_this_rank_dict = {parttype: 0 for parttype in self.parttypes}

        for i in range(self.num_files_per_snapshot):
            if i % comm_size != comm_rank:
                continue

            this_fname = self.fname + f".{i}.hdf5"
            if not os.path.isfile(this_fname):
                continue

            f = h5py.File(this_fname, "r")

            for parttype in self.parttypes:
                n = f["Header"].attrs["NumPart_ThisFile"][parttype]
                ntot_this_rank_dict[parttype] += n

            f.close()

        # Compute global offsets via allgather + cumsum.
        lefts_dict = {}
        if comm_size > 1:
            for parttype in self.parttypes:
                ntot_dict[parttype] = comm.allreduce(ntot_this_rank_dict[parttype])
                gather_counts = comm.allgather(ntot_this_rank_dict[parttype])
                rights = np.cumsum(gather_counts)
                lefts = rights - gather_counts
                assert np.array_equal(gather_counts, rights - lefts), "Particle counts don't add up"
                lefts_dict[parttype] = lefts
        else:
            for parttype in self.parttypes:
                ntot_dict[parttype] = ntot_this_rank_dict[parttype]
                lefts_dict[parttype] = [0]

        if comm_rank == 0:
            for parttype in self.parttypes:
                print(f"PartType{parttype}: {ntot_dict[parttype]} total particles")

        return lefts_dict, ntot_dict

    def read_header(self):
        """
        Read header metadata and discover dataset attributes.

        Rank 0 reads Header, Cosmology, and Parameters from file part 0.
        For each particle type, searches through file parts to find one
        containing that type, in order to record dataset dtypes and shapes.
        Results are broadcast to all ranks.
        """

        self.header = None
        self.cosmo = None
        self.params = None
        self.attribute_list = {parttype: None for parttype in self.parttypes}

        if comm_rank == 0:
            f = h5py.File(self.fname + ".0.hdf5", "r")

            if "Header" in f:
                self.header = {}
                for att in f["Header"].attrs.keys():
                    self.header[att] = f["Header"].attrs[att]
            if "Cosmology" in f:
                self.cosmo = {}
                for att in f["Cosmology"].attrs.keys():
                    self.cosmo[att] = f["Cosmology"].attrs[att]
            if "Parameters" in f:
                self.params = {}
                for att in f["Parameters"].attrs.keys():
                    self.params[att] = f["Parameters"].attrs[att]

            try:
                num_files = self.header["NumFilesPerSnapshot"][0]
            except (TypeError, IndexError):
                num_files = self.header["NumFilesPerSnapshot"]

            # For each particle type, find a file part that contains it so we
            # can record dataset names, dtypes, and shapes.
            for parttype in self.parttypes:
                parttype_group = f"PartType{parttype}"

                for i in range(num_files):
                    file_path = self.fname + f".{i}.hdf5" if i > 0 else self.fname + ".0.hdf5"
                    try:
                        check_f = h5py.File(file_path, "r") if i > 0 else f
                        if (
                            parttype_group in check_f
                            and check_f["Header"].attrs["NumPart_ThisFile"][parttype] > 0
                        ):
                            self.attribute_list[parttype] = {}
                            for att in check_f[parttype_group].keys():
                                self.attribute_list[parttype][att] = [
                                    check_f[f"{parttype_group}/{att}"].dtype,
                                    check_f[f"{parttype_group}/{att}"].shape,
                                    check_f[f"{parttype_group}/{att}"].ndim,
                                ]
                            if i > 0:
                                check_f.close()
                            break
                        if i > 0:
                            check_f.close()
                    except Exception as e:
                        print(f"Warning: error reading file part {i}: {e}")
                        continue
                else:
                    print(f"Warning: PartType{parttype} not found in any file part")

            f.close()

        if comm_size > 1:
            self.header = comm.bcast(self.header)
            self.cosmo = comm.bcast(self.cosmo)
            self.params = comm.bcast(self.params)
            self.attribute_list = comm.bcast(self.attribute_list)

        # NumFilesPerSnapshot may be stored as scalar or length-1 array.
        try:
            self.num_files_per_snapshot = self.header["NumFilesPerSnapshot"][0]
        except (TypeError, IndexError):
            self.num_files_per_snapshot = self.header["NumFilesPerSnapshot"]
