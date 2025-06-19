import os
import numpy as np
import sys
import h5py

from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.rank
comm_size = MPI.COMM_WORLD.size
comm = MPI.COMM_WORLD


class combine:
    def __init__(self, fname, outfile, parttypes=[1,2]):
        """
        Combine multiple snapshots parts into a single snapshot.

        Works for multiple particle types by specifying a list.

        Is run using `pyread_swift combine` see `pyread_swift combine --help`.

        In MPI each rank copies its own file part. Because it uses collective
        HDF5, it can't use compression.

        Parameters
        ----------
        fname : str
            Path to snapshot parts (excluding the `.x.hdf5`)
        outfile : str
            Path to output combined snapshot
        parttypes : list
            List of particle types to combine (default: [1] for DM only)
        """

        # Convert single integer to list if needed
        if isinstance(parttypes, int):
            parttypes = [parttypes]
        self.parttypes = parttypes
        self.fname = fname
        self.outfile = outfile

        # Read header.
        self.read_header()

        # Index files.
        lefts_dict, ntot_dict = self.index_files()

        # Create new output joined file.
        self.create_new_file(ntot_dict)

        # Set up datasets.
        self.create_datasets(ntot_dict)

        # Copy particles into new file.
        self.copy_particles(ntot_dict, lefts_dict)

        # Close file.
        self.new_f.close()

    def copy_particles(
        self, ntot_dict, lefts_dict, max_write_num=170000000
    ):
        # Initialize counters to track particles processed per type per rank
        particles_processed = {parttype: 0 for parttype in self.parttypes}
        """
        Copy particles from this snapshot part into new single snapshot.

        Parameters
        ----------
        ntot_dict : dict
            Dictionary of total number of particles per particle type
        lefts_dict : dict
            Dictionary of entry indexes for each snapshot part per particle type
        max_write_num : int
            Max number of particles a core can write at once
        """

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

            # Process each particle type
            for parttype in self.parttypes:
                # Skip if this particle type doesn't exist in this file
                # or if we couldn't find attributes for it in any file
                parttype_group = f"PartType{parttype}"
                if parttype_group not in this_f or self.attribute_list[parttype] is None:
                    continue

                left = 0
                this_num = this_f["Header"].attrs["NumPart_ThisFile"][parttype]
                
                if this_num > 0:
                    # Track particles per part per rank per type
                    file_part_index = j // comm_size
                    
                    # Calculate the starting position in the final array for this part's data
                    tmp_left = lefts_dict[parttype][comm_rank]
                    # Update the left index based on particles already processed in previous files
                    tmp_left += particles_processed[parttype]
                    tmp_right = tmp_left + this_num
                    
                    # Update the count of processed particles for this type
                    particles_processed[parttype] += this_num
                    
                    print(f"[Rank {comm_rank}] Part {parttype}: writing {this_num} particles from file {j} to positions {tmp_left}-{tmp_right}")
                    
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
                    for att in self.attribute_list[parttype].keys():
                        print(f"[Rank {comm_rank}] Copying {parttype_group}/{att}...")
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
        
        # Set counts for all particle types being processed
        for parttype in self.parttypes:
            ntot = ntot_dict[parttype]
            ntots[parttype] = ntot % 2**32
            highwords[parttype] = ntot >> 32
            num_this_file[parttype] = ntot

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

    def index_files(self):
        ntot_dict = {parttype: 0 for parttype in self.parttypes}
        ntot_this_rank_dict = {parttype: 0 for parttype in self.parttypes}
        
        # Count particles per file to prepare for copying
        files_per_rank = {}
        for parttype in self.parttypes:
            files_per_rank[parttype] = []
            
        for i in range(self.num_files_per_snapshot):
            if i % comm_size != comm_rank:
                continue
                
            this_fname = self.fname + f".{i}.hdf5"
            if not os.path.isfile(this_fname):
                continue
                
            f = h5py.File(this_fname, "r")
            
            for parttype in self.parttypes:
                n = f["Header"].attrs["NumPart_ThisFile"][parttype]
                if n > 0:
                    files_per_rank[parttype].append((i, n))
                ntot_this_rank_dict[parttype] += n
            
            f.close()
            
        # Print file and particle distribution for debugging
        if comm_rank == 0:
            print("===== File and particle distribution per rank =====")
        for r in range(comm_size):
            if r == comm_rank:
                for parttype in self.parttypes:
                    if files_per_rank[parttype]:
                        file_info = [f"file {f[0]}: {f[1]} particles" for f in files_per_rank[parttype]]
                        print(f"Rank {comm_rank}, PartType{parttype}: {file_info}")
            comm.Barrier()

        lefts_dict = {}
        if comm_size > 1:
            for parttype in self.parttypes:
                ntot_dict[parttype] = comm.allreduce(ntot_this_rank_dict[parttype])
                gather_counts = comm.allgather(ntot_this_rank_dict[parttype])
                rights = np.cumsum(gather_counts)
                lefts = rights - gather_counts
                counts = rights - lefts
                assert np.array_equal(gather_counts, counts), "Not add up"
                lefts_dict[parttype] = lefts
                
                # Print debugging info about particle distribution
                if comm_rank == 0:
                    print(f"PartType{parttype} distribution across ranks: {gather_counts}")
                    print(f"PartType{parttype} lefts: {lefts}")
                    print(f"PartType{parttype} rights: {rights}")
        else:
            for parttype in self.parttypes:
                ntot_dict[parttype] = ntot_this_rank_dict[parttype]
                lefts_dict[parttype] = [0]

        if comm_rank == 0:
            for parttype in self.parttypes:
                print(f"{ntot_dict[parttype]} total particles of type {parttype}")

        return lefts_dict, ntot_dict

    def read_header(self):
        """
        Read header information from snapshot parts.

        Reads the header from the 0th snapshot part.
        For each particle type, search through file parts until we find one
        that contains that particle type to get attribute information.

        Attributes
        ----------
        header/cosmo/params : dict
            Header information from the 0th snapshot
        attribute_list : dict
            Dictionary of dtypes and shapes for each attribute per particle type
        """

        self.header = None
        self.cosmo = None
        self.params = None
        self.attribute_list = {parttype: None for parttype in self.parttypes}

        if comm_rank == 0:
            # Read header info from file part 0
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
            
            # Get the number of files per snapshot
            try:
                num_files = self.header["NumFilesPerSnapshot"][0]
            except:
                num_files = self.header["NumFilesPerSnapshot"]
            
            # See what attributes each particle type has
            # For each particle type, search through files until we find one with that type
            for parttype in self.parttypes:
                parttype_group = f"PartType{parttype}"
                
                # First check file 0
                found_parttype = False
                if parttype_group in f and f["Header"].attrs["NumPart_ThisFile"][parttype] > 0:
                    # Only use this file if it actually has particles of this type
                    self.attribute_list[parttype] = {}
                    for att in f[parttype_group].keys():
                        self.attribute_list[parttype][att] = [
                            f[f"{parttype_group}/{att}"].dtype,
                            f[f"{parttype_group}/{att}"].shape,
                            f[f"{parttype_group}/{att}"].ndim,
                        ]
                    found_parttype = True
                    print(f"Found attributes for {parttype_group} in file part 0 ({f['Header'].attrs['NumPart_ThisFile'][parttype]} particles)")
                
                # If not in file 0, search through subsequent files
                if not found_parttype:
                    for i in range(1, num_files):
                        file_path = f"{self.fname}.{i}.hdf5"
                        if not os.path.isfile(file_path):
                            continue
                            
                        try:
                            next_f = h5py.File(file_path, "r")
                            if parttype_group in next_f and next_f["Header"].attrs["NumPart_ThisFile"][parttype] > 0:
                                # Only use this file if it actually has particles of this type
                                self.attribute_list[parttype] = {}
                                for att in next_f[parttype_group].keys():
                                    self.attribute_list[parttype][att] = [
                                        next_f[f"{parttype_group}/{att}"].dtype,
                                        next_f[f"{parttype_group}/{att}"].shape,
                                        next_f[f"{parttype_group}/{att}"].ndim,
                                    ]
                                found_parttype = True
                                print(f"Found attributes for {parttype_group} in file part {i} ({next_f['Header'].attrs['NumPart_ThisFile'][parttype]} particles)")
                                next_f.close()
                                break
                            next_f.close()
                        except Exception as e:
                            print(f"Error reading file part {i}: {e}")
                            continue
                
                if not found_parttype:
                    print(f"Warning: Could not find {parttype_group} in any file part")
            
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
