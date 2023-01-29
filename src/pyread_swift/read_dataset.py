import h5py
import numpy as np

from .select_region import _get_filename


def read_dataset_collective(parttype, att, params, header, region_data, index_data):
    """
    Read particles from snapshot in collective mode (each rank reads each file
    collectivley).

    Parameters
    ----------
    parttype : int
        Parttype we are reading
    att : string
        The dataset are we reading
    params : _SwiftSnapshotParams object
    header : dict
        Header information from the snapshot
    region_data : dict
        Stores the indexes for the particles in the selected region
    index_data : dict
        File indexing data from split_selection()

    Returns
    -------
    return_array : ndarray
        The particle data
    """

    file_offset_count = 0

    # Loop over each file part.
    for j, fileno in enumerate(index_data["files"]):

        # When there is so few particles in the file to read, only core 0 will read.
        if (
            params.comm_rank > 0
            and region_data["total_num_to_load"] <= params.min_in_tot_collective
        ):
            continue

        # How many particles are we loading (on this core) from this file part?
        if (
            params.comm_size > 1
            and region_data["total_num_to_load"] > params.min_in_tot_collective
        ):
            tot = params.comm.allreduce(np.sum(index_data["num_to_load"][j]))
            params.message(
                "Collective loading %i particles from file %i." % (tot, fileno)
            )
        else:
            tot = np.sum(index_data["num_to_load"][j])
            params.message("Serial loading %i particles from file %i." % (tot, fileno))

        # Open the hdf5 file.
        if (
            params.comm_size > 1
            and region_data["total_num_to_load"] > 100 * params.comm_size
        ):
            f = h5py.File(
                _get_filename(params.fname, header["NumFilesPerSnapshot"], fileno),
                "r",
                driver="mpio",
                comm=params.comm,
            )
            f.atomic = True
        else:
            f = h5py.File(
                _get_filename(params.fname, header["NumFilesPerSnapshot"], fileno), "r"
            )

        # First time round we setup the return array.
        if j == 0:
            shape = f["PartType%i/%s" % (parttype, att)].shape
            dtype = f["PartType%i/%s" % (parttype, att)].dtype

            if len(shape) > 1:
                return_array = np.empty(
                    (np.sum(index_data["total_num_to_load"]), shape[1]),
                    dtype=dtype,
                )
                byte_size = dtype.itemsize * shape[1]
            else:
                byte_size = dtype.itemsize
                return_array = np.empty(
                    np.sum(index_data["total_num_to_load"]), dtype=dtype
                )
            return_array = return_array.astype(return_array.dtype.newbyteorder("="))

        count = 0

        # Loop over each left and right block for this file.
        for l, r in zip(index_data["lefts"][j], index_data["rights"][j]):
            this_count = r - l

            # Can't read more than <max_size_to_read_at_once> at once.
            # Need to chunk it.
            num_chunks = int(
                np.ceil((this_count * byte_size) / params.max_size_to_read_at_once)
            )
            num_per_cycle = np.tile(this_count // num_chunks, num_chunks)
            mini_lefts = np.cumsum(num_per_cycle) - num_per_cycle[0]
            mini_rights = np.cumsum(num_per_cycle)
            num_per_cycle[-1] += this_count % num_chunks
            mini_rights[-1] += this_count % num_chunks
            assert (
                np.sum(mini_rights - mini_lefts) == this_count
            ), "Minis dont add up %i != this_count=%i" % (
                np.sum(mini_rights - mini_lefts),
                this_count,
            )

            # Loop over each 2 Gb chunk.
            for i in range(num_chunks):

                this_l_return = file_offset_count + count + mini_lefts[i]
                this_r_return = file_offset_count + count + mini_rights[i]

                this_l_read = l + mini_lefts[i]
                this_r_read = l + mini_rights[i]

                return_array[this_l_return:this_r_return] = f[
                    "PartType%i/%s" % (parttype, att)
                ][this_l_read:this_r_read]

            count += this_count

        f.close()

        # Keep track of offset by reading multiple files in return array.
        file_offset_count += count

    # Get return array dtype for low particle case.
    if params.comm_rank == 0:
        return_array_dtype = return_array.dtype
        return_array_shape = return_array.shape
    else:
        return_array_dtype = None
        return_array_shape = None
    if params.comm_size > 1:
        return_array_dtype = params.comm.bcast(return_array_dtype)
        return_array_shape = params.comm.bcast(return_array_shape)

    # When there is so few particles, only core 0 will read.
    # So other ranks have to return empty array.
    if (
        params.comm_rank > 0
        and region_data["total_num_to_load"] <= params.min_in_tot_collective
    ):
        if len(return_array_shape) == 2:
            return np.zeros([0, return_array_shape[1]], dtype=return_array_dtype)
        else:
            return np.zeros(
                [
                    0,
                ],
                dtype=return_array_dtype,
            )
    else:
        return return_array


def _get_dtype(fname, parttype, att, comm):
    """
    Quickly scan the reference snapshot part to establish the shape and dtype
    of the desired array.

    This is so we can build a read array to host the eventual particle data,
    before reading the data itself.

    Parameters
    ----------
    fname : string
        Path of the reference swift snapshot
    parttype : int
        Parttype of attribute
    att : string
        Name of attribute
    comm : mpi4py communicator

    Retuns
    ------
    shape : list
        Shape of array (only interested in the dimension)
    dtype : np.dtype
        Datatype of array
    """

    if comm.rank == 0:
        with h5py.File(fname, "r") as f:
            shape = f["PartType%i/%s" % (parttype, att)].shape
            dtype = f["PartType%i/%s" % (parttype, att)].dtype
    else:
        shape = None
        dtype = None

    if comm.size > 1:
        shape = comm.bcast(shape)
        dtype = comm.bcast(dtype)

    return shape, dtype


def read_dataset_distributed(parttype, att, params, header, index_data):
    """
    Read particles from snapshot in distributed mode (only one rank ever reads
    a single file).

    Parameters
    ----------
    parttype : int
        Parttype we are reading
    att : string
        The dataset are we reading
    params : _SwiftSnapshotParams object
    header : dict
        Header information from the snapshot
    index_data : dict
        File indexing data from split_selection()

    Returns
    -------
    return_array : ndarray
        The particle data
    """

    # Get shape and dtype of array.
    shape, dtype = _get_dtype(params.fname, parttype, att, params.comm)

    # New communicator
    # Because we might have more ranks than files.
    new_comm = params.comm.Split(
        (0 if len(index_data["num_to_load"]) > 0 else 1), params.comm_rank
    )

    # No particles for this rank to load. Return empty array.
    if len(index_data["num_to_load"]) == 0:
        if len(shape) == 2:
            return np.zeros([0, shape[1]], dtype=dtype)
        else:
            return np.zeros(
                [
                    0,
                ],
                dtype=dtype,
            )

    # Number of particles this core is loading.
    tot = np.sum(np.concatenate(index_data["num_to_load"]))

    # Total number of files the snapshot is distributed over.
    num_files = len(index_data["num_to_load"])

    params.message(f"Loading {tot} particles from {num_files} files.")

    # Set up return array.
    if len(shape) > 1:
        return_array = np.empty((tot, shape[1]), dtype=dtype)
        byte_size = dtype.itemsize * shape[1]
    else:
        byte_size = dtype.itemsize
        return_array = np.empty(tot, dtype=dtype)
    return_array = return_array.astype(return_array.dtype.newbyteorder("="))

    # Loop over each file and load the particles.
    # One rank per file.
    count = 0
    lo_ranks = np.arange(0, new_comm.size + params.max_concur_io, params.max_concur_io)[
        :-1
    ]
    hi_ranks = np.arange(0, new_comm.size + params.max_concur_io, params.max_concur_io)[
        1:
    ]

    # Making sure not to go over max_concur_io.
    for lo, hi in zip(lo_ranks, hi_ranks):

        # Loop over each file.
        for j, fileno in enumerate(index_data["files"]):

            if not (new_comm.rank >= lo and new_comm.rank < hi):
                continue

            params.message(f"Loading file {fileno}.")

            f = h5py.File(
                _get_filename(params.fname, header["NumFilesPerSnapshot"], fileno), "r"
            )

            # Loop over each left and right block for this file.
            for l, r in zip(index_data["lefts"][j], index_data["rights"][j]):
                this_count = r - l

                # Can't read more than <max_size_to_read_at_once> at once.
                # Need to chunk it.
                num_chunks = int(
                    np.ceil((this_count * byte_size) / params.max_size_to_read_at_once)
                )
                num_per_cycle = np.tile(this_count // num_chunks, num_chunks)
                mini_lefts = np.cumsum(num_per_cycle) - num_per_cycle[0]
                mini_rights = np.cumsum(num_per_cycle)
                num_per_cycle[-1] += this_count % num_chunks
                mini_rights[-1] += this_count % num_chunks
                assert (
                    np.sum(mini_rights - mini_lefts) == this_count
                ), "Minis dont add up %i != this_count=%i" % (
                    np.sum(mini_rights - mini_lefts),
                    this_count,
                )

                # Loop over each 2 Gb chunk.
                for i in range(num_chunks):

                    this_l_return = count + mini_lefts[i]
                    this_r_return = count + mini_rights[i]

                    this_l_read = l + mini_lefts[i]
                    this_r_read = l + mini_rights[i]

                    return_array[this_l_return:this_r_return] = f[
                        "PartType%i/%s" % (parttype, att)
                    ][this_l_read:this_r_read]

                count += this_count

            f.close()

        # Need to wait as to not go over max_concur_io.
        if new_comm.size > 1:
            new_comm.barrier()

    return return_array
