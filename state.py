from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
from layers import Layers
from palette import decode

@dataclass
class World:
    background: np.ndarray
    homeland: np.ndarray
    initial_control: np.ndarray
    control: np.ndarray
    battle_age: np.ndarray
    encirclement_age: np.ndarray
    disputed: np.ndarray
    layers: Layers
    start_date: str
    elapsed: int = 0
    fresh_capture: np.ndarray | None = None
    meeting_age: np.ndarray | None = None

    @classmethod
    def create(cls, rgba, start_date, layers=None):
        date.fromisoformat(start_date)
        home,control=decode(rgba)
        layer=layers or Layers(); layer.rebuild(home.shape)
        initial=control.copy(); initial.flags.writeable=False
        return cls(rgba,home,initial,control,np.zeros_like(home,np.uint16),np.zeros_like(home,np.uint16),np.zeros_like(home,bool),layer,start_date)

    @property
    def date(self): return date.fromisoformat(self.start_date)+timedelta(days=self.elapsed)

    @property
    def territory(self): return self.homeland != 0

    def stats(self):
        valid=~self.disputed
        return {'date':self.date.isoformat(),'days':self.elapsed,
            'kefir':int(np.count_nonzero((self.control==1)&valid)),
            'yogurt':int(np.count_nonzero((self.control==2)&valid)),
            'kefir_occupation':int(np.count_nonzero((self.control==1)&(self.homeland==2)&valid)),
            'yogurt_occupation':int(np.count_nonzero((self.control==2)&(self.homeland==1)&valid)),
            'contested':int(np.count_nonzero(self.disputed))}
