from spheral.config import spheral_config
from spheral.sphere_models import Sphere

# Cleaner API
set_option = spheral_config.set_option
get_option = spheral_config.get_option

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spheral")  # Reads version from installed package metadata
except PackageNotFoundError:
    __version__ = "unknown"  # Fallback when running locally without installation

