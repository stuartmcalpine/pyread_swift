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
        Combine multiple snapshots parts into a single snapshot.

        Currently only works for PartType=1 (DM).

        Is run using `pyread_swift combine` see `pyread_swift combine --help`.

        In MPI each rank copies its own file part. Because it uses collective
        HDF5, it can't use compression.

        Parameters
        ----------
        fname : str
            Path to snapshot parts (excluding the `.x.hdf5`)
        outfile : str
            Path to output combined snapshot
        parttypes : list[int]
            Must be 1,2 or both (DM)
        """

        self.parttypes = parttypes
        self.fname = fname
        self.outfile = outfile

        # Read header.
        self.read_header()

        # Index files.
        lefts, ntot = self.index_files()

        # Create new output joined file.
        self.create_new_file(ntot)

        # Copy particles into new file.
        self.copy_particles(lefts)

        # Close file.
        self.new_f.close()

    def copy_particles(self, lefts, max_write_num=170000000):
        """
        Copy particles from this snapshot part into new single snapshot.

        Parameters
        ----------
        lefts : dict
            Entry indexes for each snapshot part in the combined snapshot
        max_write_num : int
            Max number of particles a core can write at once
        """

        left = {}

        for parttype in self.parttypes:
            left[parttype] = 0

        # Loop over each snapshot part and copy them over.
        for j in range(self.num_files_per_snapshot):
            if j % comm_size != comm_rank:
                continue

            # This snapshot part.
            this_fname = self.fname + ".%i.hdf5" % j
            if not os.path.isfile(this_fname):
                raise ValueError("Bad snapshot part asked for")

            # Open the file
            this_f = h5py.File(this_fname, "r")

            # Loop over each parttype
            for parttype in self.parttypes:

                this_num = this_f["Header"].attrs["NumPart_ThisFile"][parttype]
                if this_num > 0:
                    tmp_left = lefts[parttype][comm_rank] + left[parttype]
                    tmp_right = lefts[parttype][comm_rank] + left[parttype] + this_num

                    num_chunks = int(np.ceil(this_num / max_write_num))
                    num_per_cycle = np.tile(this_num // num_chunks, num_chunks)
                    mini_lefts = np.cumsum(num_per_cycle) - num_per_cycle[0]
                    mini_rights = np.cumsum(num_per_cycle)
                    num_per_cycle[-1] += this_num % num_chunks
                    mini_rights[-1] += this_num % num_chunks
                    assert (
                        np.sum(mini_rights - mini_lefts) == this_num
                    ), "Minis dont add up %i != this_num=%i" % (
                        np.sum(mini_rights - mini_lefts),
                        this_num,
                    )

                    # Loop over each attribute and copy.
                    for att in self.attribute_list.keys():
                        if f"PartType{parttype}" not in att:
                            continue

                        print(f"[Rank {comm_rank}] Copying {att}...")
                        num_read = 0
                        for i in range(num_chunks):
                            this_l_read = mini_lefts[i]
                            this_r_read = mini_rights[i]

                            this_l_write = tmp_left + mini_lefts[i]
                            this_r_write = this_l_write + (mini_rights[i] - mini_lefts[i])

                            tmp_data = this_f[att][this_l_read:this_r_read]
                            num_read += len(tmp_data)
                            with self.new_f[att].collective:
                                self.new_f[att][this_l_write:this_r_write] = tmp_data

                        assert tmp_right - tmp_left == num_read, "Indexing error"
                    left[parttype] += this_num
            this_f.close()
            print(
                "[Rank %i] Done %i/%i" % (comm_rank, j + 1, self.num_files_per_snapshot)
            )

    def create_new_file(self, ntot):
        """
        Create the placeholder output file.

        This initiates the file and its headers, then creates placeholder
        datasets of the correct shape and dtype.

        Parameters
        ----------
        ntot : dict
            Total number of particles for each parttype
        """

        # Create HDF5 file.
        # Keep it open so we can add to it later.
        self.new_f = h5py.File(self.outfile, "w", driver="mpio", comm=comm)
        self.new_f.atomic = True
        if comm_rank == 0:
            print(f"creating {self.outfile}...")

        # Create header group
        grp = self.new_f.create_group("Header")

        # Updated counts.
        ntots = [0, 0, 0, 0, 0, 0]
        highwords = [0, 0, 0, 0, 0, 0]
        num_this_file = [0, 0, 0, 0, 0, 0]

        for parttype in self.parttypes:
            ntots[parttype] = ntot[parttype] % 2**32
            highwords[parttype] = ntot[parttype] >> 32
            num_this_file[parttype] = ntot[parttype]

        # Create Header group
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

        # Create Cosmology group (if exists)
        if self.cosmo is not None:
            grp = self.new_f.create_group("Cosmology")
            for att in self.cosmo.keys():
                grp.attrs.create(att, self.cosmo[att])

        # Create Parameters group (if exists)
        if self.params is not None:
            grp = self.new_f.create_group("Parameters")
            for att in self.params.keys():
                if att == "Snapshots:distributed":
                    grp.attrs.create(att, 0)
                else:
                    grp.attrs.create(att, self.params[att])

        # Create the dataset placeholders.
        for parttype in self.parttypes:
            grp = self.new_f.create_group(f"PartType{parttype}")

            for att in self.attribute_list.keys():
                if f"PartType{parttype}" in att:

                    shape = self.attribute_list[att][1]
                    if len(shape) == 1:
                        grp.create_dataset(
                            att.split("/")[1],
                            (ntot[parttype],),
                            dtype=self.attribute_list[att][0],
                        )
                    else:
                        grp.create_dataset(
                            att.split("/")[1],
                            (ntot[parttype], shape[1]),
                            dtype=self.attribute_list[att][0],
                        )

    def index_files(self):
        """
        Index the files.

        Count up how many particles each rank will load in, and where it will
        put those particles in the final combined snapshot.

        Returns
        -------
        lefts : list
            Left index this ranks partices will go in the final snapshot
        ntot : list
            Total number of particles this rank will deal with
        """
        ntot_this_rank = [0, 0, 0, 0, 0, 0]

        # Loop over each snap part, count the particles from the header.
        for i in range(self.num_files_per_snapshot):
            if i % comm_size != comm_rank:
                continue
            this_fname = self.fname + f".{i}.hdf5"
            if not os.path.isfile(this_fname):
                raise FileNotFoundError()

            with h5py.File(this_fname, "r") as f:
                for parttype in self.parttypes:
                    ntot_this_rank[parttype] += f["Header"].attrs["NumPart_ThisFile"][
                        parttype
                    ]

        ntot = {}
        lefts = {}
        rights = {}

        if comm_size > 1:

            ntot_all_ranks = np.vstack(comm.allgather(ntot_this_rank))

            for parttype in self.parttypes:
                ntot[parttype] = np.sum(ntot_all_ranks[:, parttype])
                rights[parttype] = np.cumsum(ntot_all_ranks[:, parttype], axis=0)
                lefts[parttype] = rights[parttype] - ntot_all_ranks[:, parttype]

        else:
            ntot_all_ranks = ntot_this_rank

            for parttype in self.parttypes:
                ntot[parttype] = ntot_all_ranks[parttype]
                lefts[parttype] = [0]

        if comm_rank == 0:
            for parttype in self.parttypes:
                print(f"{ntot[parttype]} total particles of type {parttype}")

        return lefts, ntot

    def read_header(self):
        """
        Read header information from 0th snapshot part.

        Also gets information about dtypes and shapes of each attribute.

        Attributes
        ----------
        header/cosmo/params : dict
            Header information from the 0th snapshot
        attribute_list : dict
            dtype and shape information for each attribute
        """

        self.header = None
        self.cosmo = None
        self.params = None
        self.attribute_list = None

        if comm_rank == 0:
            with h5py.File(self.fname + ".0.hdf5", "r") as f:
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

            # See what attributes the file has.
            self.attribute_list = {}
            for i_file in range(self.header["NumFilesPerSnapshot"]):
                with h5py.File(self.fname + f".{i_file}.hdf5", "r") as f:

                    for parttype in self.parttypes:
                        if f"PartType{parttype}" not in f.keys():
                            continue

                        for att in f[f"PartType{parttype}"].keys():
                            this_att = f"PartType{parttype}/{att}"

                            if this_att in self.attribute_list.keys():
                                continue

                            self.attribute_list[this_att] = [
                                f[this_att].dtype,
                                f[this_att].shape,
                                f[this_att].ndim,
                            ]

        if comm_size > 1:
            self.header = comm.bcast(self.header)
            self.cosmo = comm.bcast(self.cosmo)
            self.params = comm.bcast(self.params)
            self.attribute_list = comm.bcast(self.attribute_list)

        # Might be an array in some cases, might be scalar, so pull it out.
        try:
            self.num_files_per_snapshot = self.header["NumFilesPerSnapshot"][0]
        except:
            self.num_files_per_snapshot = self.header["NumFilesPerSnapshot"]
