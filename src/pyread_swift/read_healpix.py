import numpy as np
import os
import h5py

class SwiftLightconeHealpix:

    def __init__(self, fname):

        # Make sure file exists
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Passed file {fname} doesn't exist")

        self.fname = fname

        # Start by loading the shell header
        self._read_header()

    def _read_header(self):
        """
        Read the shell header information from one of the file parts

        Attributes
        ----------
        self.header : dict
        """

        self.header = {}
        with h5py.File(self.fname, "r") as f:
            for att in f["Shell"].attrs:
                self.header[att] = f["Shell"].attrs.get(att)[0]

    def _get_filename(self, i):
        """
        Get the path for a given filepart using a base filename for reference

        Parameters
        ----------
        i : int
            Which file part

        Return
        ------
        - : str
            Full path to file part i
        """
        
        file_name = os.path.basename(self.fname)
        base_name = os.path.dirname(self.fname)

        parts = file_name.split(".")
        parts[2] = str(i)

        return os.path.join(base_name, ".".join(parts))
        
    
    def read_lightcone(self, which_lightcone):
        """
        Loop over each lightcone file and read in the healpix array.

        It also appends the attribute information of the array to the header in
        a sub-dict `self.header[which_lightcone]`.

        Parameters
        ----------
        which_lightcone : str
            Which lightcone array to load (e.g., "DarkMatterMass")

        Return
        ------
        arr : ndarray
            Combined Healpix array
        """

        for i in range(self.header["nr_files_per_shell"]):
            
            this_fname = self._get_filename(i)
            with h5py.File(this_fname, "r") as f:
                if i == 0:
                    arr = f[which_lightcone][...]

                    # Append header
                    for att in f[which_lightcone].attrs:
                        self.header[att] = f[which_lightcone].attrs.get(att)[0]

                else:
                    arr = np.concatenate((arr, f[which_lightcone][...]))

        return arr
