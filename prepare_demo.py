"""Explicit offline correction of a thin black cartographic BORDER, never water.
Runtime /newmap decoding remains strict: all unknown colors are impassable.
The original uploaded map is preserved as yogurtstan_base_control.png.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
import cv2
from layers import read_png,png_bytes
from palette import decode,COLORS
import config

def prepare(rgba):
    home,control=decode(rgba)
    black=np.all(rgba[:,:,:3]==0,axis=2)&(rgba[:,:,3]==255)
    d1=cv2.distanceTransform((control!=1).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    d2=cv2.distanceTransform((control!=2).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    border=black&(d1<=3)&(d2<=3)
    result=rgba.copy()
    result[border&(d1<=d2),:3]=COLORS[1,1]
    result[border&(d1>d2),:3]=COLORS[2,2]
    return result,{'changed_border_pixels':int(border.sum()),'source_territory_pixels':int((control!=0).sum()),'water_changed':0,'rule':'Only opaque exact-black pixels within 3 px of BOTH homeland colors; nearest side, ties kefir. This preprocessing occurs before a new demo, never during simulation.'}

if __name__=='__main__':
    source=config.ROOT/'yogurtstan_base_control.png'
    output=config.ROOT/'yogurtstan_demo_control.png'
    prepared,report=prepare(read_png(source.read_bytes()))
    output.write_bytes(png_bytes(prepared))
    (config.ROOT/'demo_preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)
