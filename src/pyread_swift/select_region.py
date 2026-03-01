import h5py
import numpy as np
import os


def _get_filename(fname, num_files_per_snapshot, fileno):
    """
    For a given partnumber, what's the filename?

    Parameters
    ----------
    fname : string
        The base filename passed to pyread_swift
    num_files_per_snapshot : int
        From the header, number of files a snapshot is split over
    fileno : int
        The file part number we want the filename for

    Returns
    -------
    - : string
        The desired filename
    """
    if num_files_per_snapshot > 1:
        dirname = os.path.dirname(fname)
        basename = os.path.basename(fname)         # 'snap_0077.0.hdf5'
        base = basename.rsplit(".", 2)[0]          # 'snap_0077'
        return os.path.join(dirname, f"{base}.{fileno}.hdf5")
    else:
        return fname


def _mask_selected_region(
    params, x_min, x_max, y_min, y_max, z_min, z_max, boxsize, eps=1e-4
):
    """
    Find all the top level cells that fit into the selected region.

    If the Swift snapshot doesn't have any top level cell information, we will
    not be able to select on a region, and will have to load all the particles.

    Parameters
    ----------
    params : _SwiftSnapshotParams object
        Contains the pyread_swift parameters
    x_min/x_max : float
        Minimum and maximum bounds in x-dim to select
    y_min/y_max : float
        Minimum and maximum bounds in y-dim to select
    z_min/z_max : float
        Minimum and maximum bounds in z-dim to select
    boxsize : float
        Boxsize of simulation
    eps : float
        Buffer to add to select region

    Returns
    -------
    mask : ndarray
        Mask of TL cells to load (same for all ranks)
    """

    # Placeholders
    mask = None

    # Rank 0 loads the snapshot and sees if a "Cells" group exists.
    if params.comm_rank == 0:

        with h5py.File(params.fname, "r") as f:

            # Do we have top level cell information?
            if "Cells" in f:
                centres = f["/Cells/Centres"][...]
                size = f["/Cells/Meta-data"].attrs["size"]

                # Coordinates to load around.
                coords = np.array(
                    [
                        x_min + (x_max - x_min) / 2.0,
                        y_min + (y_max - y_min) / 2.0,
                        z_min + (z_max - z_min) / 2.0,
                    ]
                )

                # Wrap to given coordinates.
                centres = (
                    np.mod(centres - coords + 0.5 * boxsize, boxsize)
                    + coords
                    - 0.5 * boxsize
                )

                # Find what cells fall within boundary.
                dx_over_2 = (x_max - x_min) / 2.0 + eps
                dy_over_2 = (y_max - y_min) / 2.0 + eps
                dz_over_2 = (z_max - z_min) / 2.0 + eps
                half_size = size / 2.0  # Half a cell size.

                mask = np.where(
                    (centres[:, 0] + half_size[0] >= coords[0] - dx_over_2)
                    & (centres[:, 0] - half_size[0] <= coords[0] + dx_over_2)
                    & (centres[:, 1] + half_size[1] >= coords[1] - dy_over_2)
                    & (centres[:, 1] - half_size[1] <= coords[1] + dy_over_2)
                    & (centres[:, 2] + half_size[2] >= coords[2] - dz_over_2)
                    & (centres[:, 2] - half_size[2] <= coords[2] + dz_over_2)
                )[0]

    if params.comm_size > 1:
        mask = params.comm.bcast(mask)

    return mask


def _mask_selected_shell(params, cx, cy, cz, r_min, r_max, boxsize):
    """
    Find all the top level cells that overlap a spherical shell region.

    A cell is selected if it overlaps the outer sphere (r_max) and is not
    entirely contained within the inner sphere (r_min). For a full sphere
    pass r_min=0.

    If the Swift snapshot doesn't have any top level cell information, we will
    not be able to select on a region, and will have to load all the particles.

    Parameters
    ----------
    params : _SwiftSnapshotParams object
        Contains the pyread_swift parameters
    cx/cy/cz : float
        Centre of the sphere/shell
    r_min : float
        Inner radius of the shell (0 for a full sphere)
    r_max : float
        Outer radius of the shell
    boxsize : float
        Boxsize of simulation

    Returns
    -------
    mask : ndarray
        Mask of TL cells to load (same for all ranks)
    """

    mask = None

    if params.comm_rank == 0:

        with h5py.File(params.fname, "r") as f:

            if "Cells" in f:
                centres = f["/Cells/Centres"][...]
                half_size = f["/Cells/Meta-data"].attrs["size"] / 2.0

                query_centre = np.array([cx, cy, cz])

                # Wrap cell centres relative to sphere centre (periodic BC).
                centres = (
                    np.mod(centres - query_centre + 0.5 * boxsize, boxsize)
                    + query_centre
                    - 0.5 * boxsize
                )

                delta = np.abs(centres - query_centre)

                # Squared distance from sphere centre to nearest point on each cell.
                # nearest point per axis = clamp(centre, cell_min, cell_max)
                d2 = np.sum(np.maximum(0.0, delta - half_size) ** 2, axis=1)

                # Squared distance from sphere centre to farthest point on each cell.
                D2 = np.sum((delta + half_size) ** 2, axis=1)

                # A cell overlaps the shell if it intersects the outer sphere
                # and is not entirely inside the inner sphere.
                overlaps_outer = d2 < r_max ** 2
                fully_inside_inner = D2 <= r_min ** 2

                mask = np.where(overlaps_outer & ~fully_inside_inner)[0]

    if params.comm_size > 1:
        mask = params.comm.bcast(mask)

    return mask


def _build_region_data(params, header, part_type, mask):
    """
    Build the region_data dict from a top-level cell mask.

    Given a mask of TL cell indices (or None if no cell information exists),
    reads the per-cell offsets/counts/files and merges contiguous cells into
    the minimum number of HDF5 slice reads.

    Parameters
    ----------
    params : _SwiftSnapshotParams object
        Contains the pyread_swift parameters
    header : dict
        Snapshot header information
    part_type : int
        Parttype to select on
    mask : ndarray or None
        Indices of TL cells to load. None means no cell info; load everything.

    Returns
    -------
    region_data : dict
        Stores the particle index information for each file (which particles to
        load for the selected region)
    """

    region_data = {}

    # ---------------------------------------------------
    # No top level cell information, just load everything
    # ---------------------------------------------------
    if mask is None:
        params.message(f"No TL cell information, just loading all cells...")

        region_data["lefts"] = []
        region_data["rights"] = []
        region_data["files"] = []
        region_data["num_to_load"] = []

        # Loop over each file part to index them.
        lo_ranks = np.arange(
            0, params.comm_size + params.max_concur_io, params.max_concur_io
        )[:-1]
        hi_ranks = np.arange(
            0, params.comm_size + params.max_concur_io, params.max_concur_io
        )[1:]
        for lo, hi in zip(lo_ranks, hi_ranks):
            for this_file_i in range(header["NumFilesPerSnapshot"]):
                if not (params.comm_rank >= lo and params.comm_rank < hi):
                    continue

                if this_file_i % params.comm_size != params.comm_rank:
                    continue
                tmp_f = h5py.File(
                    _get_filename(
                        params.fname, header["NumFilesPerSnapshot"], this_file_i
                    ),
                    "r",
                )
                tmp_num_this_file = tmp_f["Header"].attrs["NumPart_ThisFile"][part_type]
                if tmp_num_this_file == 0:
                    continue
                region_data["num_to_load"].append(tmp_num_this_file)
                region_data["lefts"].append(0)
                region_data["rights"].append(tmp_num_this_file)
                region_data["files"].append(this_file_i)
                tmp_f.close()

            if params.comm_size > 1:
                params.comm.barrier()

        region_data["total_num_to_load"] = header["NumPart_Total"][part_type]
        for att in ["lefts", "rights", "num_to_load", "files"]:
            if params.comm_size > 1:
                region_data[att] = np.concatenate(
                    params.comm.allgather(region_data[att])
                )
            region_data[att] = np.array(region_data[att], dtype="i8")

        # Make sure we found all the particles we intended to.
        tmp_tot = np.sum(region_data["num_to_load"])
        assert (
            tmp_tot == region_data["total_num_to_load"]
        ), "Error loading region, count err"

        # How many did we find?
        params.message(
            f"{tmp_tot} particles selected from {len(region_data['files'])} files"
        )

    # -----------------------------------------------------------------
    # Yes we have top level cell information, load only selected region
    # -----------------------------------------------------------------

    else:
        # Make sure we found some particles.
        if len(mask) == 0:
            raise Exception("Found no cells in selected region")

        if params.comm_rank == 0:

            # Read in the top level cell information.
            with h5py.File(params.fname, "r") as f:

                if "Cells/OffsetsInFile/" in f:
                    offsets = f["Cells/OffsetsInFile/PartType%i" % part_type][mask]
                else:
                    offsets = f["Cells/Offset/PartType%i" % part_type][mask]
                counts = f["Cells/Counts/PartType%i" % part_type][mask]
                files = f["Cells/Files/PartType%i" % part_type][mask]

            # Only interested in cells with at least 1 particle.
            keep = np.where(counts > 0)
            offsets = offsets[keep]
            counts = counts[keep]
            files = files[keep]

            # Sort by file number then by offsets.
            sort_mask = np.lexsort((offsets, files))
            offsets = offsets[sort_mask]
            counts = counts[sort_mask]
            files = files[sort_mask]

            # Case of one cell.
            if len(offsets) == 1:
                region_data["files"] = [files[0]]
                region_data["lefts"] = [offsets[0]]
                region_data["rights"] = [offsets[0] + counts[0]]
            # Case of multiple cells.
            else:
                region_data["lefts"] = []
                region_data["rights"] = []
                region_data["files"] = []

                buff = 0
                # Loop over each cell.
                for i in range(len(offsets) - 1):
                    if offsets[i] + counts[i] == offsets[i + 1]:
                        buff += counts[i]

                        if i == len(offsets) - 2:
                            region_data["lefts"].append(offsets[i + 1] - buff)
                            region_data["rights"].append(offsets[i + 1] + counts[i + 1])
                            region_data["files"].append(files[i + 1])
                    else:
                        region_data["lefts"].append(offsets[i] - buff)
                        region_data["rights"].append(offsets[i] + counts[i])
                        region_data["files"].append(files[i])
                        buff = 0

                        if i == len(offsets) - 2:
                            region_data["lefts"].append(offsets[i + 1] - buff)
                            region_data["rights"].append(offsets[i + 1] + counts[i + 1])
                            region_data["files"].append(files[i + 1])

            # Convert to numpy arrays.
            for tmp_att in region_data.keys():
                region_data[tmp_att] = np.array(region_data[tmp_att])

            # Add up the particles.
            region_data["total_num_to_load"] = np.sum(counts)
            region_data["num_to_load"] = region_data["rights"] - region_data["lefts"]

            # Make sure we found all the particles we intended to.
            tmp_tot = np.sum(region_data["num_to_load"])
            assert (
                tmp_tot == region_data["total_num_to_load"]
            ), "Error loading region, count err"

            params.message(
                "%i cells and %i particles selected from %i file(s)."
                % (len(offsets), tmp_tot, len(np.unique(files)))
            )

        if params.comm_size > 1:
            region_data = params.comm.bcast(region_data)

    return region_data


def select_region(
    params,
    header,
    part_type,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
):
    """
    Find the snapshot files and top level cells that contain particles
    of a given type within a selected cuboidal region.

    Selection is based off the position of the top level cells. Any top
    level cells that intercet the load region will be selected, and their
    particles indexed for loading.

    Generates a "region_data" dict that stores the HDF5 array indexs to
    load from each file. This is the same for all ranks, until
    split_selection is called.

    Parameters
    ----------
        params : _SwiftSnapshotParams object
                Contains the pyread_swift parameters
        header : dict
                Snapshot header information
    part_type : int
        Parttype to select on
    x_min/x_max : float
        Minimum and maximum bounds in x-dim to select
    y_min/y_max : float
        Minimum and maximum bounds in y-dim to select
    z_min/z_max : float
        Minimum and maximum bounds in z-dim to select

        Returns
        -------
        region_data : dict
        Stores the particle index information for each file (which particles to
        load for the selected region)
    """

    mask = _mask_selected_region(
        params,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        header["BoxSize"],
    )

    return _build_region_data(params, header, part_type, mask)


def select_spherical_region(
    params,
    header,
    part_type,
    cx,
    cy,
    cz,
    r_min,
    r_max,
):
    """
    Find the snapshot files and top level cells that contain particles
    of a given type within a spherical shell region.

    Selection is based off the position of the top level cells. Any top
    level cells that overlap the shell [r_min, r_max] will be selected,
    and their particles indexed for loading. Use r_min=0 for a full sphere.

    Note: the selection is conservative — TL cells on the boundary of the
    sphere will be included in full even if only partially inside. The
    caller is responsible for filtering loaded particles by distance.

    Generates a "region_data" dict that stores the HDF5 array indexs to
    load from each file. This is the same for all ranks, until
    split_selection is called.

    Parameters
    ----------
    params : _SwiftSnapshotParams object
        Contains the pyread_swift parameters
    header : dict
        Snapshot header information
    part_type : int
        Parttype to select on
    cx/cy/cz : float
        Centre of the sphere/shell
    r_min : float
        Inner radius of the shell (0 for a full sphere)
    r_max : float
        Outer radius of the shell

    Returns
    -------
    region_data : dict
        Stores the particle index information for each file (which particles to
        load for the selected region)
    """

    mask = _mask_selected_shell(
        params,
        cx,
        cy,
        cz,
        r_min,
        r_max,
        header["BoxSize"],
    )

    return _build_region_data(params, header, part_type, mask)
