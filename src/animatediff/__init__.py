try:
    from ._version import (
        version as __version__,
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

import os
import sys
from functools import lru_cache
from os import getenv
from pathlib import Path
from warnings import filterwarnings

if sys.platform == "win32":
    # huggingface_hub < 0.19 probes for symlink support (hf_hub_download(local_dir=...)) by creating a temp
    # dir in the common parent of the HF cache and data/ — usually C:\Users, which isn't writable. Python's
    # mkdtemp then retries ~238k times and the download appears to hang forever. Copying files instead of
    # symlinking skips the probe. Must be set before huggingface_hub is imported.
    os.environ.setdefault("HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD", str(1 << 62))

from rich.console import Console
from tqdm import TqdmExperimentalWarning

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent.parent

HF_HOME = Path(getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
HF_HUB_CACHE = Path(getenv("HUGGINGFACE_HUB_CACHE", HF_HOME.joinpath("hub")))

HF_LIB_NAME = "animatediff-cli"
HF_LIB_VER = __version__
HF_MODULE_REPO = "neggles/animatediff-modules"

console = Console(highlight=True)
err_console = Console(stderr=True)

# shhh torch, don't worry about it it's fine
filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
# you too tqdm
filterwarnings("ignore", category=TqdmExperimentalWarning)


@lru_cache(maxsize=4)
def get_dir(dirname: str = "data") -> Path:
    if PACKAGE_ROOT.name == "src":
        # we're installed in editable mode from within the repo
        dirpath = PACKAGE_ROOT.parent.joinpath(dirname)
    else:
        # we're installed normally, so we just use the current working directory
        dirpath = Path.cwd().joinpath(dirname)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath.absolute()


__all__ = [
    "__version__",
    "version_tuple",
    "PACKAGE",
    "PACKAGE_ROOT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "console",
    "err_console",
    "get_dir",
    "models",
    "pipelines",
    "rife",
    "utils",
    "cli",
    "generate",
    "schedulers",
    "settings",
]
