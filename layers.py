import io
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt
import config

def read_png(data, shape=None):
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Нужен PNG как документ. JPEG и фотографии не принимаются')
    with Image.open(io.BytesIO(data)) as im:
        if im.width * im.height > config.MAX_PIXELS:
            raise ValueError('Карта слишком большая')
        if shape and (im.height, im.width) != tuple(shape):
            raise ValueError(f'Неправильный размер: нужен {shape[1]}×{shape[0]} PNG')
        return np.array(im.convert('RGBA'))

def png_bytes(array):
    out=io.BytesIO(); Image.fromarray(array).save(out, format='PNG', compress_level=3)
    return out.getvalue()

@dataclass
class Layers:
    settings: dict | None = None
    terrain: np.ndarray | None = None
    cities: np.ndarray | None = None
    roads: np.ndarray | None = None
    terrain_cost: np.ndarray | None = None
    road_distance: np.ndarray | None = None
    road_mask: np.ndarray | None = None
    road_nodes: np.ndarray | None = None
    city_mask: np.ndarray | None = None
    city_influence: np.ndarray | None = None

    def rebuild(self, shape):
        if self.settings is None: self.settings=config.physics()
        settings=self.settings
        self.terrain_cost = np.ones(shape, np.float32)
        if self.terrain is not None:
            rgb=self.terrain[:,:,:3].astype(np.float32)
            lum=(rgb[:,:,0]*.2126+rgb[:,:,1]*.7152+rgb[:,:,2]*.0722)/255
            if self.settings['TERRAIN_MODE'] == 'brightness':
                height=lum
            elif self.settings['TERRAIN_MODE'] == 'inverse_brightness':
                height=1-lum
            else:
                raise ValueError('TERRAIN_MODE: brightness или inverse_brightness')
            self.terrain_cost = (self.settings['TERRAIN_MIN']+(self.settings['TERRAIN_MAX']-self.settings['TERRAIN_MIN'])*height**1.7).astype(np.float32)
        road=np.zeros(shape, bool)
        if self.roads is not None:
            a=self.roads[:,:,3]; rgb=self.roads[:,:,:3]
            if np.any(a==0): road=a>32
            else: road=rgb.max(axis=2)>128  # opaque black/white binary mask
        density=cv2.boxFilter(road.astype(np.float32),-1,(11,11),normalize=True)
        self.road_mask=road
        self.road_nodes=cv2.dilate((road&(density>.28)).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))).astype(bool)
        self.road_distance=cv2.distanceTransform((~road).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if road.any() else np.full(shape, 1e5,np.float32)
        self.city_mask=np.zeros(shape,bool)
        if self.cities is not None:
            if not np.any(self.cities[:,:,3]<255):
                raise ValueError('Слой городов должен иметь прозрачный фон')
            self.city_mask=self.cities[:,:,3]>32
        if self.city_mask.any():
            dist=cv2.distanceTransform((~self.city_mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            self.city_influence=np.exp(-dist/self.settings['CITY_RADIUS']).astype(np.float32)
        else: self.city_influence=np.zeros(shape,np.float32)

    def movement_cost(self, sl):
        # Roads partly compensate terrain; supply improves continuously near road.
        advantage=np.exp(-self.road_distance[sl]/self.settings['ROAD_FALLOFF'])
        compensation=.82 if self.settings.get('FRONT_MODEL',1)>=2 else .65
        terrain=1+(self.terrain_cost[sl]-1)*(1-compensation*advantage)
        return (terrain*(1-(1-self.settings['ROAD_COST'])*advantage)*(1+(self.settings['CITY_COST']-1)*self.city_influence[sl])*(1+.20*self.road_nodes[sl])).astype(np.float32)
