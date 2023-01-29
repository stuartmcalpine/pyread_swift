import h5py
import numpy as np


def read_header(params):
    """
    Get information from the "Header", "Cosmology", "RuntimePars", and
    "Parameters" groups.

    Parameters
    ----------
    params : _SwiftSnapshotParams object

    Returns
    -------
    header : dict
        Loaded header information
    """

    # Rank 0 reads the header from the file.
    if params.comm_rank == 0:
        params.message(f"Reading {params.fname} header...")

        with h5py.File(params.fname, "r") as f:

            # Dict to store values.
            header = {}

            # Load attributes from groups.
            for grp in ["Header", "Cosmology", "Parameters", "RuntimePars"]:
                if grp not in f:
                    continue

                # Loop over each attribute in the group.
                for att in f[grp].attrs.keys():
                    if att in header.keys():
                        print(f"*WARNING* Duplicate header attribute {att}..")
                        continue
                    header[att] = f[grp].attrs.get(att)
    else:
        header = None
    if params.comm_size > 1:
        header = params.comm.bcast(header)

    # Check we have the minimum values.
    _required = ["NumFilesPerSnapshot", "NumPart_ThisFile", "NumPart_Total", "BoxSize"]
    for att in _required:
        assert att in header.keys(), f"Missing {att} from header"

    # Convert total number of particles to unsigned I64.
    header["NumPart_Total"] = header["NumPart_Total"].astype(np.uint64)

    # Compute total number of particles using highword.
    if "NumPart_Total_HighWord" in header.keys():
        for i in range(len(header["NumPart_Total"])):
            header["NumPart_Total"][i] = header["NumPart_Total"][i] + (
                header["NumPart_Total_HighWord"][i] * 2**32
            )
            header["NumPart_Total_HighWord"][i] = 0

    # Assume boxsize is equal in all dimensions.
    if type(header["BoxSize"]) == list or type(header["BoxSize"]) == np.ndarray:
        if len(header["BoxSize"]) >= 1:
            header["BoxSize"] = float(header["BoxSize"][0])

    return header
