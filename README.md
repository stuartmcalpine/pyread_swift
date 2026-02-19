## An MPI read routine for Swift simulation snapshots

``pyread_swift`` is an MPI read routine for [``swiftsim``](https://github.com/SWIFTSIM/swiftsim) snapshots, very similar in style to John Helly's [``read_eagle``](https://gitlab.cosma.dur.ac.uk/jch/Read_Eagle) code to read EAGLE snapshots.

The package can read ``swiftsim`` snapshots both in "collective" (i.e., multiple MPI ranks read from a single file simultaneously) and "distributed" (i.e., each MPI reads an individual snapshot file part in isolation) modes. 

## Installation

### Requirements

* `OpenMPI` or other MPI library
* `python>=3.10`
* `virgodc`

Recommended modules when working on COSMA7:

```bash
module load gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.14.4 fftw/3.3.10
module load python/3.12.4
```

Given the need for a parallel HDF5 installation, it is recommended you install ``pyread_swift`` within a virtual/conda environment. However you can ofcourse also install directly into your base Python environment if you prefer.

First make sure your `pip` is up-to-date:

```bash
python3 -m pip install --upgrade pip
```

### Method 1) Installation from PyPi

The easiest method is to install from [``PyPI``](https://pypi.org/project/pyread-swift/)

```bash
python3 -m pip install pyread-swift
```

### Method 2) Installation from source

Or, you can install directly from source.

First clone the repo, then you can install the `pyread_swift` package by typing the following in
the root git directory: 

```bash
git clone https://github.com/stuartmcalpine/pyread_swift.git
cd pyread_swift
python3 -m pip install .
```

which will install `pyread_swift` and any dependencies.

### MPI installation for collective reading

If you are using `pyread_swift` to load large snapshots over MPI collectively
(i.e., multiple cores read in parallel from the same file), a bit of additional
setup is required.

Make sure you have `hdf5` installed with **parallel** compatibility ([see here for details](https://docs.h5py.org/en/stable/mpi.html)).

Then, uninstall any versions of `h5py` and reinstall from source:

```bash
python3 -m pip uninstall h5py
MPICC=mpicc CC=mpicc HDF5_MPI="ON" python3 -m pip install --no-binary=h5py h5py
```

If `pip` struggles to find your `HDF5` libraries automatically, e.g., `error: libhdf5.so: cannot open shared object file: No such file or directory`. You may have to specify the path to the HDF5 installation manually, i.e., `HDF5_DIR=/path/to/hdf5/lib` (see [here](https://docs.h5py.org/en/stable/build.html#building-against-parallel-hdf5) for more details).

For our COSMA7 setup, that would be:

`HDF5_DIR="/cosma/local/parallel-hdf5//gnu_14.1.0_ompi_5.0.3/1.14.4/"`

## Usage

``pyread_swift`` is build around a primary read wrapper, called ``SwiftSnapshot``. The snapshot particles are loaded into, stored, and manipulated by this object.

Reading follows these four steps (see also the examples below):

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

## Lightcone particle reading

``pyread_swift`` also provides ``SwiftParticleLightcone`` for reading particle lightcone outputs. Files are discovered automatically from a directory, and MPI is supported in distributed mode (no parallel HDF5 required).

### Input parameters to SwiftParticleLightcone

| Input | Description | Default |
| ----- | ----------- | ------- |
| lightcone_dir | Path to directory containing lightcone files | - |
| lightcone_id | Integer lightcone ID (e.g. `0` for files named `lightcone0_XXXX.hdf5`) | - |
| verbose= | True for more verbose output | False |
| comm= | MPI4PY communicator (if reading in MPI) | None |

### Example usage (No MPI case)

```python
from pyread_swift import SwiftParticleLightcone

lc = SwiftParticleLightcone("/path/to/lightcone/", lightcone_id=0)

# Inspect header.
print(lc.header)

# Read a dataset (each call can request a different attribute or particle type).
coords = lc.read_dataset("Coordinates", parttype="DM")
```

### Example usage (MPI case)

```python
from mpi4py import MPI
from pyread_swift import SwiftParticleLightcone

comm = MPI.COMM_WORLD

lc = SwiftParticleLightcone("/path/to/lightcone/", lightcone_id=0, comm=comm)

# Each rank reads a strided subset of files and holds its own portion.
coords = lc.read_dataset("Coordinates", parttype="DM")
```

## Lightcone HEALPix reading

``SwiftLightconeHealpix`` reads HEALPix map shells from SWIFT lightcone outputs. Shells are split across multiple file parts, which are located and combined automatically.

### Input parameters to SwiftLightconeHealpix

| Input | Description |
| ----- | ----------- |
| fname | Path to any one file part of the shell (e.g. `lightcone0.shell_0001.0.hdf5`) |

### Example usage

```python
from pyread_swift import SwiftLightconeHealpix

healpix = SwiftLightconeHealpix("/path/to/lightcone0.shell_0001.0.hdf5")

# Inspect header (e.g. nside, redshift bounds, nr_files_per_shell).
print(healpix.header)

# Read and combine HEALPix array across all file parts.
arr = healpix.read_lightcone("DarkMatterMass")
```

### Combining snapshot parts

`pyread_swift` provides a CLI tool to combine multiple snapshot file parts into a single file (designed for DMO simulations):

```bash
mpirun -np 4 pyread_swift combine /path/to/snapshot_0000 /path/to/combined_snapshot.hdf5
```
