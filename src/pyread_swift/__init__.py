import importlib.metadata

from .io import SwiftSnapshot
from .read_healpix import SwiftLightconeHealpix

__version__ = importlib.metadata.version("pyread_swift")
