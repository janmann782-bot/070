"""Extract orders, validate fresh native PNG, orient arbitrary curved lines."""
import hashlib
import numpy as np
import cv2
from layers import read_png
from palette import ORDERS
import config

def order_masks(rgba):
    rgb=rgba[:,:,:3].astype(np.int16)
    masks={s:np.max(np.abs(rgb-np.array(c,dtype=np.int16)),axis=2)<=config.ORDER_TOLERANCE for s,c in ORDERS.items()}
    return masks

def validate_markup(data, reference, permitted=(1,2)):
    rgba=read_png(data,reference.shape[:2]); masks=order_masks(rgba)
    drawn=masks[1]|masks[2]
    if not drawn.any(): raise ValueError('Нет линий приказа: #FF00FF / #00FF66')
    if any(masks[s].any() for s in (1,2) if s not in permitted):
        raise ValueError('На карте есть приказ другой стороны: используйте /turn')
    guard=cv2.dilate(drawn.astype(np.uint8),np.ones((2*config.REFERENCE_GUARD_PX+1,)*2,np.uint8)).astype(bool)
    if guard.mean()>config.REFERENCE_MAX_COVERAGE:
        raise ValueError('Разметка закрывает слишком много карты; рисуйте тонкие линии')
    delta=np.max(np.abs(rgba[:,:,:3].astype(np.int16)-reference[:,:,:3].astype(np.int16)),axis=2)
    changed=(delta>config.REFERENCE_TOLERANCE)&~guard
    if changed.mean()>config.REFERENCE_CHANGED_FRACTION:
        raise ValueError('Карта устарела или изменена. Возьмите свежую /map после команды хода')
    return {s:m for s,m in masks.items() if s in permitted and m.any()}

def reference_hash(rgb): return hashlib.sha256(rgb.tobytes()).hexdigest()
