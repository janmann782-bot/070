"""All recognition/physics knobs. Changing physics requires a new session."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ENGINE_VERSION = '1.0'
TERRAIN_MODE = 'brightness'  # supplied map: black plains, white mountains
TERRAIN_MIN = 1.0
TERRAIN_MAX = 1.7
TERRAIN_STRENGTH = 0.21
ROAD_COST = 0.55
ROAD_FALLOFF = 12.0
CITY_COST = 2.6
CITY_RADIUS = 9.0
CITY_HOLD_DAYS = 18
MAX_DAYS = 1460
MAX_WIDTH = 1200
MAX_UPLOAD_BYTES = 19 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_EXPORT_WIDTH = 4096
MAX_FPS = 60
MAX_EXPORT_BYTES = 49 * 1024 * 1024
# Exact source colors only: blue rivers (0,0,255) remain nonterritory.
ORDER_TOLERANCE = 65
REFERENCE_TOLERANCE = 3
REFERENCE_CHANGED_FRACTION = 0.00005
REFERENCE_GUARD_PX = 4
REFERENCE_MAX_COVERAGE = 0.10

PHYSICS_KEYS = ["TERRAIN_MODE","TERRAIN_MIN","TERRAIN_MAX","TERRAIN_STRENGTH","ROAD_COST","ROAD_FALLOFF","CITY_COST","CITY_RADIUS","CITY_HOLD_DAYS"]
def physics(): return {k:globals()[k] for k in PHYSICS_KEYS}
