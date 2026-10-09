"""Extract markers from the supplied alpha layer, never invent city names."""
import hashlib
import cv2
import numpy as np
from city_catalog import name_event


class Settlements:
    def __init__(self, world):
        self.signature = self.layer_signature(world)
        self.items = []
        cities = world.layers.cities
        if cities is None:
            return
        mask = (cities[:,:,3] > 128).astype(np.uint8)
        n, labels, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
        # Work only on labelled pixels. No per-city full-map scans.
        flat = np.flatnonzero(labels)
        order = np.argsort(labels.ravel()[flat], kind='stable')
        flat = flat[order]
        ids, start, counts = np.unique(labels.ravel()[flat], return_index=True, return_counts=True)
        for marker_id, a, count in zip(ids,start,counts):
            idx = flat[a:a+count]
            idx = idx[world.homeland.ravel()[idx] != 0]
            if len(idx) < 3:
                continue
            x, y = centres[marker_id]
            self.items.append(name_event(dict(id=int(marker_id), x=float(x), y=float(y), pixels=idx),
                                         (world.control.shape[1], world.control.shape[0])))

    @staticmethod
    def layer_signature(world):
        if world.layers.cities is None:
            return None
        return hashlib.sha256(world.layers.cities.tobytes()).hexdigest()

    def owners(self, world):
        result = {}
        for marker in self.items:
            idx = marker['pixels']
            if np.count_nonzero(world.disputed.ravel()[idx]) > len(idx)/2:
                owner = 0
            else:
                votes = np.bincount(world.control.ravel()[idx], minlength=3)
                owner = int(np.argmax(votes[1:])+1)
            result[marker['id']] = owner
        return result

    def describe(self, world, previous):
        owners = self.owners(world)
        counts = {side:sum(v == side for v in owners.values()) for side in [0,1,2]}
        events = []
        for marker in self.items:
            key = marker['id']; side = owners[key]
            before = previous.get(key, side)
            if side and before and side != before:
                home = np.bincount(world.homeland.ravel()[marker['pixels']], minlength=3).argmax()
                events.append({k:marker[k] for k in ['id','x','y','name']} | dict(
                    side=side, action='Возвращен под родной контроль' if home==side else 'Установлен контроль', day=world.elapsed))
        return owners, counts, events
