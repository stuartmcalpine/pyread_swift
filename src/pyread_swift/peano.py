import numpy as np

"""
Routine to calculate Peano-Hilbert keys from a set of coordinates.

Taken from John Helly's VirgoDC package
https://github.com/jchelly/VirgoDC/blob/master/python/virgo/util/peano.py
"""

blocksize = 10000

quadrants = np.asarray([
  0, 7, 1, 6, 3, 4, 2, 5,
  7, 4, 6, 5, 0, 3, 1, 2,
  4, 3, 5, 2, 7, 0, 6, 1,
  3, 0, 2, 1, 4, 7, 5, 6,
  1, 0, 6, 7, 2, 3, 5, 4,
  0, 3, 7, 4, 1, 2, 6, 5,
  3, 2, 4, 5, 0, 1, 7, 6,
  2, 1, 5, 6, 3, 0, 4, 7,
  6, 1, 7, 0, 5, 2, 4, 3,
  1, 2, 0, 3, 6, 5, 7, 4,
  2, 5, 3, 4, 1, 6, 0, 7,
  5, 6, 4, 7, 2, 1, 3, 0,
  7, 6, 0, 1, 4, 5, 3, 2,
  6, 5, 1, 2, 7, 4, 0, 3,
  5, 4, 2, 3, 6, 7, 1, 0,
  4, 7, 3, 0, 5, 6, 2, 1,
  6, 7, 5, 4, 1, 0, 2, 3,
  7, 0, 4, 3, 6, 1, 5, 2,
  0, 1, 3, 2, 7, 6, 4, 5,
  1, 6, 2, 5, 0, 7, 3, 4,
  2, 3, 1, 0, 5, 4, 6, 7,
  3, 4, 0, 7, 2, 5, 1, 6,
  4, 5, 7, 6, 3, 2, 0, 1,
  5, 2, 6, 1, 4, 3, 7, 0], dtype=np.int32).reshape((24,2,2,2))

rotxmap_table = np.asarray([4, 5, 6, 7, 8, 9, 10, 11,
                         12, 13, 14, 15, 0, 1, 2,
                         3, 17, 18, 19, 16, 23, 20,
                         21, 22], dtype=np.int32)

rotymap_table = np.asarray([1, 2, 3, 0, 16, 17, 18, 19,
                         11, 8, 9, 10, 22, 23, 20,
                         21, 14, 15, 12, 13, 4, 5,
                         6, 7], dtype=np.int32)

rotx_table = np.asarray([3, 0, 0, 2, 2, 0, 0, 1], dtype=np.int32)
roty_table = np.asarray([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)
sense_table = np.asarray([-1, -1, -1, +1, +1, -1, -1, -1], dtype=np.int32)

def peano_hilbert_keys_block(ix, iy, iz, bits):
    """
    Function to calculate Peano-Hilbert keys

    ix, iy, iz : integer coordinates in the grid
    bits       : number of bits per dimension in each key

    Returns the P-H key corresponding to coordinates (ix,iy,iz).
    If ix, iy and iz are sequences, key will be a numpy array
    with the same number of elemnts.
    """
    x = np.asarray(ix, dtype=np.int32)
    y = np.asarray(iy, dtype=np.int32)
    z = np.asarray(iz, dtype=np.int32)

    mask     = 1 << (bits - 1)
    key      = np.zeros(x.shape, dtype=np.int64)
    rotation = np.zeros(key.shape, dtype=np.int32)
    sense    = np.ones(key.shape, dtype=np.int32)

    for i in range(bits):

        bitx = np.where(x & mask, 1, 0)
        bity = np.where(y & mask, 1, 0)
        bitz = np.where(z & mask, 1, 0)
        quad = quadrants[rotation,bitx,bity,bitz]

        key <<= 3
        key += np.where(sense==1, quad, 7-quad)

        rotx = rotx_table[quad]
        roty = roty_table[quad]
        sense *= sense_table[quad]

        if len(x.shape) > 0:
            while np.any(rotx>0):
                ind = rotx > 0
                rotation[ind] = rotxmap_table[rotation[ind]]
                rotx[ind] -= 1

            while np.any(roty>0):
                ind = roty > 0
                rotation[ind] = rotymap_table[rotation[ind]]
                roty[ind] -= 1
        else:
            while rotx>0:
                rotation = rotxmap_table[rotation]
                rotx -= 1
            while roty>0:
                rotation = rotymap_table[rotation]
                roty -= 1

        mask >>= 1

    return key

def peano_hilbert_keys_from_coords(pos, boxsize, bits):
    """
    Calculate PH keys given particle coordinates,
    size of the simulation box, and number of bits
    per dimension.
    
    Divides the calculation into blocks to minimize
    memory usage.
    """
    
    cellsize = np.float(boxsize) / np.float(2**bits)
    n = pos.shape[0]
    key = np.ndarray(n, dtype=np.int64)
    for i1 in np.arange(0, n, blocksize):

        # Get section to do
        i2 = i1 + blocksize
        if i2 > n:
            i2 = n
        
        # Calculate coordinates
        ix = np.floor(pos[i1:i2,0]/cellsize).astype(np.int32)
        iy = np.floor(pos[i1:i2,1]/cellsize).astype(np.int32)
        iz = np.floor(pos[i1:i2,2]/cellsize).astype(np.int32)

        # Calculate keys
        key[i1:i2] = peano_hilbert_keys_block(ix, iy, iz, bits)

    return key
