## An MPI read routine for Swift simulation snapshots

``pyread_swift`` is an MPI read routine for [``swiftsim``](https://github.com/SWIFTSIM/swiftsim) snapshots, very similar in style to John Helly's [``read_eagle``](https://gitlab.cosma.dur.ac.uk/jch/Read_Eagle) code to read EAGLE snapshots.

The package can read ``swiftsim`` snapshots both in "collective" (i.e., multiple MPI ranks read from a single file simultaneously) and "distributed" (i.e., each MPI reads an individual snapshot file part in isolation) modes. 

Can also read in pure serial, if no MPI libraries are installed.

## Installation

### Requirements

* `OpenMPI` or other MPI library
* `python>=3.8`

Recommended modules when working on COSMA7:

* `module load gnu_comp/11.1.0 openmpi/4.1.4 parallel_hdf5/1.12.0 python/3.9.1-C7`

### Installation from source

Given the need for a parallel HDF5 installation, it is recommended you install ``pyread_swift`` within a virtual/conda environment. However you can ofcourse also install directly into your base Python environment if you prefer.

First make sure your `pip` is up-to-date:

* `python3 -m pip install --upgrade pip`

Then you can install the `pyread_swift` package by typing the following in
the root git directory: 

* `python3 -m pip install .[mpi]`

which will install `pyread_swift` and any dependencies (omit ``[mpi]`` for pure serial version).

### MPI installation for collective reading

If you are using `pyread_swift` to load large snapshots over MPI collectively
(i.e., multiple cores read in parallel from the same file), a bit of additional
setup is required.

Make sure you have `hdf5` installed with **parallel** compatibility ([see here for details](https://docs.h5py.org/en/stable/mpi.html)).

Then, uninstall any installed versions of `mpi4py` and `h5py`:

* `python3 -m pip uninstall mpi4py h5py`

and reinstall then from source, with MPI flags:

* `MPICC=mpicc CC=mpicc HDF5_MPI="ON" python3 -m pip install --no-binary=mpi4py mpi4py`

* `MPICC=mpicc CC=mpicc HDF5_MPI="ON" python3 -m pip install --no-binary=h5py h5py`

If `pip` struggles to find your `HDF5` libraries automatically, e.g., `error: libhdf5.so: cannot open shared object file: No such file or directory`. You may have to specify the path to the HDF5 installation manually, i.e., `HDF5_DIR=/path/to/hdf5/lib` (see [here](https://docs.h5py.org/en/stable/build.html#building-against-parallel-hdf5) for more details).

For our COSMA7 setup, that would be:

`HDF5DIR="/cosma/local/parallel-hdf5//gnu_11.1.0_ompi_4.1.4/1.12.0/"`

## Usage

``pyread_swift`` is build around a primary read wrapper, called ``SwiftSnapshot``. The snapshot particles are loaded into, stored, and manipulated by this object.

Reading follows the same four steps (see also the examples below):

* Initialize a ``SwiftSnapshot`` object pointing to the location of the HDF5 file.

* Select the spatial region you want to extract the particles from using the ``select_region()`` routine.

* Split the selection over the MPI ranks using the ``split_selection()`` routine.

* Read a selected property of the particles using the ``read_dataset()`` routine.

### Input parameters to SwiftSnapshot

| Input | Description | Default option |
| ----- | ----------- | --------- |
| fname | Full path to HDF5 snapshot file. If the snapshot is split over multiple files, this can just be one of the file parts | - |
| comm= | MPI4PY communicator (if reading in MPI) | None |
| verbose= | True for more a more verbose output | False |
| mpi_read_format= | How to read the snapshot in MPI mode ("collective" or "distributed") <br><br>"collective": Do a collective read of each file, i.e., all ranks read a single file at one. Recommended for single, or few large snapshot file(s). Requires parallel-hdf5 to be installed. <br><br>"distributed": Each rank reads its own file part. Recommended for multiple smaller files. | "collective" |
| max_concur_io= | When reading in MPI, how many HDF5 files can be open at once | 64 |

### Example usage (No MPI case)

```python
from pyread_swift import SwiftSnapshot

# Set up pyread_swift object pointing at HDF5 snapshot file (or a file part). 
snapshot = "/path/to/snap/part.0.hdf5"
swift = SwiftSnapshot(snapshot)

# Select region to load from.
parttype = 1 # Dark matter
region = [0,100,0,100,0,100] # [xlo,xhi,ylo,yhi,zlo,zhi]
swift.select_region(parttype, *region)

# Divide selection between ranks (needs to be invoked even for non-mpi case).
swift.split_selection()

# Read data.
ids = swift.read_dataset(parttype, "ParticleIDs")
```

### Example usage (MPI case)

```python
from mpi4py import MPI
from pyread_swift import SwiftSnapshot

# MPI communicator.
comm = MPI.COMM_WORLD

# Set up read_swift object pointing at HDF5 snapshot file (or a file part). 
snapshot = "/path/to/snap/part.0.hdf5"
swift = SwiftSnapshot(snapshot, comm=comm)

# Select region to load from.
parttype = 1 # Dark matter
region = [0,100,0,100,0,100] # [xlo,xhi,ylo,yhi,zlo,zhi]
swift.select_region(parttype, *region)

# Divide selection between ranks.
swift.split_selection()

# Read data.
ids = swift.read_dataset(parttype, "ParticleIDs")
```


