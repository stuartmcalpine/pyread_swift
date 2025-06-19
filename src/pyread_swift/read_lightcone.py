import numpy as np
import os
import h5py
import logging
import time
from typing import Dict, List, Union, Optional, Any, Tuple


class SwiftParticleLightcone:
    def __init__(
        self,
        lightcone_dir: str,
        lightcone_id: int,
        verbose: bool = False,
        log_level: int = logging.INFO,
        comm=None,
    ):
        """
        Initialize a SwiftParticleLightcone object.

        Parameters:
        -----------
        lightcone_dir : str
            Path to lightcone files
        lightcone_id : int
            Which lightcone to load
        verbose : bool, optional
            Whether to output detailed information. Defaults to True.
        log_level : int, optional
            Logging level to use. Defaults to logging.INFO.

        Raises:
        -------
        FileNotFoundError
            If the specified file does not exist.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(log_level if verbose else logging.WARNING)

        # Check file exists
        if not os.path.isdir(lightcone_dir):
            self.logger.error(f"Directory not found: {lightcone_dir}")
            raise FileNotFoundError(f"{lightcone_dir} not found")

        self.lightcone_dir = lightcone_dir
        self.lightcone_id = lightcone_id
        self.verbose = verbose

        # MPI communicator
        if comm is None:
            self.comm = None
            self.rank = 0
            self.size = 1
        else:
            self.comm = comm
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

        # Performance tracking
        self.timings = {}

        # Get lightcone files
        self._get_lightcone_files()

        # Get information from the header
        self.read_header()

    def _get_lightcone_files(self):
        """Look in the lightcone directory and see what files there are"""

        files = [
            x
            for x in os.listdir(self.lightcone_dir)
            if f"lightcone{self.lightcone_id}_" in x
        ]
        if len(files) == 0:
            self.logger.error(f"No files found in : {self.lightcone_dir}")
            raise FileNotFoundError(f"No lightcone files found")

        self.logger.info(f"{len(files)} lightcone files found")
        self.lightcone_files = np.sort(files)

    def read_header(self) -> None:
        """
        Read information from the "Lightcone" group in the HDF5 file.

        This method populates the self.header dictionary with attributes from the file.
        """
        start_time = time.time()

        if self.rank == 0:
            self.logger.info(f"Reading header")
            self.header = {}
            try:
                fname = os.path.join(self.lightcone_dir, self.lightcone_files[0])
                with h5py.File(fname, "r") as f:
                    if "Lightcone" not in f:
                        self.logger.error("No 'Lightcone' group found in file")
                        raise KeyError("No 'Lightcone' group found in file")

                    for att in f["Lightcone"].attrs.keys():
                        tmp = f["Lightcone"].attrs.get(att)
                        if len(tmp) == 1:
                            self.header[att] = tmp[0]
                        else:
                            self.header[att] = tmp
            except Exception as e:
                self.logger.error(f"Error reading header: {str(e)}")
                raise
        else:
            self.header = None

        if self.size > 1:
            self.header = self.comm.bcast(self.header, root=0)

        if self.rank == 0 and self.verbose:
            self.logger.info("Header information:")
            for att, v in self.header.items():
                self.logger.info(f"  {att}: {v}")

        self.timings["read_header"] = time.time() - start_time

    def read_dataset(self, att: str, parttype: str = "DM") -> np.ndarray:
        """
        Read a dataset from the HDF5 files across MPI ranks.

        Parameters:
        -----------
        att : str
            Attribute name to read from the dataset.
        parttype : str, optional
            Particle type to read. Defaults to "DM" (Dark Matter).

        Returns:
        --------
        np.ndarray
            Combined dataset from all files.

        Notes:
        ------
        Each MPI rank reads a subset of the files and the results are combined.
        """
        start_time = time.time()
        self.logger.debug(
            f"[Rank {self.rank}] Starting to read dataset {att} for {parttype}"
        )

        tmp_data = []
        files_processed = 0
        files_with_errors = 0
        total_particles = 0
        data_shape = None

        nr_files = self.header.get("nr_mpi_ranks", 0)
        if nr_files == 0:
            self.logger.warning(
                "Header does not contain 'nr_mpi_ranks', cannot determine file count"
            )
            return np.array([])

        for i, lightcone_file in enumerate(self.lightcone_files):
            # Assign each file to a specific rank using a deterministic pattern
            # This ensures even distribution of work across ranks
            if i % self.size != self.rank:
                continue

            file_path = os.path.join(self.lightcone_dir, lightcone_file)

            if not os.path.isfile(file_path):
                self.logger.debug(
                    f"[Rank {self.rank}] {file_path} doesn't exist, skipping"
                )
                continue

            self.logger.debug(f"[Rank {self.rank}] Loading: {file_path}")

            try:
                with h5py.File(file_path, "r") as f:
                    dataset_path = f"{parttype}/{att}"

                    if parttype not in f:
                        self.logger.warning(
                            f"[Rank {self.rank}] Particle type '{parttype}' not found in file {i}"
                        )
                        files_with_errors += 1
                        continue

                    if att not in f[parttype]:
                        self.logger.warning(
                            f"[Rank {self.rank}] Attribute '{att}' not found under '{parttype}' for file {i}"
                        )
                        files_with_errors += 1
                        continue

                    data = f[dataset_path][...]
                    data_shape = data.shape
                    total_particles += data.shape[0]
                    tmp_data.append(data)
                    files_processed += 1
            except Exception as e:
                self.logger.error(
                    f"[Rank {self.rank}] Error reading file {file_path}: {str(e)}"
                )
                files_with_errors += 1

        # Collect statistics from all ranks
        if self.size > 1:
            all_processed = self.comm.gather(files_processed, root=0)
            all_errors = self.comm.gather(files_with_errors, root=0)
            all_particles = self.comm.gather(total_particles, root=0)
        else:
            all_processed = [files_processed]
            all_errors = [files_with_errors]
            all_particles = [total_particles]

        # Join results
        result = None
        if tmp_data:
            if len(data_shape) > 1 and data_shape[1] == 3:
                result = np.vstack(tmp_data)
            else:
                result = np.concatenate(tmp_data)

            self.logger.debug(f"[Rank {self.rank}] Combined data shape: {result.shape}")
        else:
            result = np.array([])
            self.logger.warning(f"[Rank {self.rank}] No data collected for {att}")

        read_time = time.time() - start_time
        self.timings[f"read_dataset_{parttype}_{att}"] = read_time

        # Print summary if verbose and on rank 0
        if self.rank == 0 and self.verbose:
            total_processed = sum(all_processed)
            total_errors = sum(all_errors)
            total_parts = sum(all_particles)

            self.logger.info("=" * 60)
            self.logger.info(f"Dataset read summary for {parttype}/{att}:")
            self.logger.info(
                f"  Total files processed: {total_processed} of {nr_files}"
            )
            self.logger.info(f"  Files with errors: {total_errors}")
            self.logger.info(f"  Total particles loaded: {total_parts}")
            self.logger.info(f"  Time taken: {read_time:.2f} seconds")
            self.logger.info("=" * 60)

        return result

    def get_performance_report(self) -> Dict[str, float]:
        """
        Get a dictionary of performance timings for the various operations.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping operation names to execution times in seconds.
        """
        return self.timings
