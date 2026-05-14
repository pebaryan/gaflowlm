"""Model package exports.

Keep imports tolerant so lightweight modules can be used without pulling in
optional training-time dependencies such as flash_attn or lightning.
"""

from . import ema

try:
    from . import dit
except Exception:
    dit = None

try:
    from . import sphere_arch
except Exception:
    sphere_arch = None

try:
    from . import sphere_dit
except Exception:
    sphere_dit = None

try:
    from . import care
except Exception:
    care = None

try:
    from . import cfs_arch
except Exception:
    cfs_arch = None

try:
    from . import cfs_model
except Exception:
    cfs_model = None
