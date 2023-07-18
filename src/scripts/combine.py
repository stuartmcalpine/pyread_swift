import os
import numpy as np
import sys
import h5py

from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.rank
comm_size = MPI.COMM_WORLD.size
comm = MPI.COMM_WORLD


class combine:
    def __init__(self, fname, outfile, parttype=1):
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
        parttype : int
            Must be 1 (DM)
        """

        self.parttype = parttype
        self.fname = fname
        self.outfile = outfile

        # Read header.
        self.read_header()

        # Index files.
        lefts, ntot = self.index_files()

        # Create new output joined file.
        self.create_new_file(ntot)

        # Set up datasets.
        self.create_datasets(ntot)

        # Copy particles into new file.
        self.copy_particles(ntot, lefts)

        # Close file.
        self.new_f.close()

    def copy_particles(
        self, ntot, lefts, max_write_num=170000000
    ):
        """
        Copy particles from this snapshot part into new single snapshot.

        Parameters
        ----------
        ntot : int
            Total number of particles in combined snapshot
        lefts : array
            Entry indexes for each snapshot part in the combined snapshot
        max_write_num : int
            Max number of particles a core can write at once
        """

        left = 0
        total_num = 0

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

            this_num = this_f["Header"].attrs["NumPart_ThisFile"][self.parttype]
            total_num += this_num
            if this_num > 0:
                tmp_left = lefts[comm_rank] + left
                tmp_right = lefts[comm_rank] + left + this_num

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
                    print(f"[Rank {comm_rank}] Copying {att}...")
                    num_read = 0
                    for i in range(num_chunks):
                        this_l_read = mini_lefts[i]
                        this_r_read = mini_rights[i]

                        this_l_write = tmp_left + mini_lefts[i]
                        this_r_write = this_l_write + (mini_rights[i] - mini_lefts[i])

                        tmp_data = this_f["PartType%i/%s" % (self.parttype, att)][
                            this_l_read:this_r_read
                        ]
                        num_read += len(tmp_data)
                        with self.new_f[
                            "PartType%i/%s" % (self.parttype, att)
                        ].collective:
                            self.new_f["PartType%i/%s" % (self.parttype, att)][
                                this_l_write:this_r_write
                            ] = tmp_data

                    assert tmp_right - tmp_left == num_read, "Indexing error"
                left += this_num
            this_f.close()
            print(
                "[Rank %i] Done %i/%i"
                % (comm_rank, j + 1, self.num_files_per_snapshot)
            )

    def create_datasets(self, ntot):
        grp = self.new_f.create_group(f"PartType{self.parttype}")

        for att in self.attribute_list.keys():
            shape = self.attribute_list[att][1]
            if len(shape) == 1:
                grp.create_dataset(att, (ntot,), dtype=self.attribute_list[att][0])
            else:
                grp.create_dataset(att, (ntot,shape[1]), dtype=self.attribute_list[att][0])

    def create_new_file(self, ntot):
        self.new_f = h5py.File(self.outfile, "w", driver="mpio", comm=comm)
        self.new_f.atomic = True
        if comm_rank == 0:
            print(f"creating {self.outfile}...")
        grp = self.new_f.create_group("Header")

        # Updated counts.
        ntots = [0, 0, 0, 0, 0, 0]
        highwords = [0, 0, 0, 0, 0, 0]
        ntots[self.parttype] = ntot % 2**32
        highwords[self.parttype] = ntot >> 32
        num_this_file = [0, 0, 0, 0, 0, 0]
        num_this_file[self.parttype] = ntot

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
        ntot_this_rank = 0

        for i in range(self.num_files_per_snapshot):
            if i % comm_size != comm_rank:
                continue
            this_fname = self.fname + f".{i}.hdf5"
            if not os.path.isfile(this_fname):
                continue
            f = h5py.File(this_fname, "r")
            n = f["Header"].attrs["NumPart_ThisFile"][self.parttype]
            f.close()

            ntot_this_rank += n

        if comm_size > 1:
            ntot = comm.allreduce(ntot_this_rank)
            gather_counts = comm.allgather(ntot_this_rank)
            rights = np.cumsum(gather_counts)
            lefts = rights - gather_counts
            counts = rights - lefts
            assert np.array_equal(gather_counts, counts), "Not add up"
        else:
            ntot = ntot_this_rank
            lefts = [0]

        if comm_rank == 0:
            print(f"{ntot} total particles of type {self.parttype}")

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

            # See what attributes the file has.
            self.attribute_list = {}
            for att in f[f"PartType{self.parttype}"].keys():
                self.attribute_list[att] = [
                    f[f"PartType{self.parttype}/{att}"].dtype,
                    f[f"PartType{self.parttype}/{att}"].shape,
                    f[f"PartType{self.parttype}/{att}"].ndim,
                ]

            f.close()

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
