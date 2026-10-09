"""Match supplied names to native marker coordinates, never to component IDs."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree

CATALOG_PATH = Path(__file__).resolve().with_name('settlements.json')
MATCH_RADIUS = 8.0


class CityCatalog:
    def __init__(self, data):
        self.size = (int(data['width']), int(data['height']))
        self.places = data['all_places']
        points = np.array([(float(p['x']), float(p['y'])) for p in self.places], dtype=float).reshape(-1, 2)
        if self.size[0] <= 0 or self.size[1] <= 0 or not np.isfinite(points).all():
            raise ValueError('Некорректные координаты каталога городов')
        if any(not isinstance(p['name'], str) or not p['name'].strip() for p in self.places):
            raise ValueError('Пустое название в каталоге городов')
        self.tree = cKDTree(points) if len(points) else None

    def find(self, x, y, source_size):
        # Custom maps must supply their own native-size registry. No guessed scaling.
        if tuple(source_size) != self.size or self.tree is None:
            return None
        indices = self.tree.query_ball_point((x, y), MATCH_RADIUS)
        if len(indices) != 1:
            return None  # Ambiguous overlapping markers are not assigned an arbitrary name.
        return self.places[indices[0]]


@lru_cache(maxsize=4)
def _read(path, modified_ns, size):
    return CityCatalog(json.loads(Path(path).read_text(encoding='utf-8')))


def catalog():
    try:
        stat = CATALOG_PATH.stat()
    except FileNotFoundError:
        return None
    return _read(str(CATALOG_PATH), stat.st_mtime_ns, stat.st_size)


def name_event(event, source_size):
    result = dict(event)
    source = catalog()
    place = source.find(result['x'], result['y'], source_size) if source else None
    result['name'] = place['name'] if place else 'Без названия'
    result['named'] = place is not None
    return result


def present_info(info, recent=()):
    """Refresh presentation metadata without modifying stored deltas or checksums."""
    result = dict(info)
    result['captures'] = [name_event(e, info['source_size']) for e in info.get('captures', [])]
    for event in result['captures']:
        event['label'] = info['label']
    result['recent'] = (list(recent) + result['captures'])[-4:]
    return result
