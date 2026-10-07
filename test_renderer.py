import io
import numpy as np
import pytest
import cv2
from PIL import Image
from test_engine import world,line
from renderer import render,Curve
from layers import read_png,png_bytes
from orders import validate_markup
from palette import COLORS,CONTESTED

def test_occupation_and_homeland_render():
    w=world(); w.control[50:60,110:120]=1; w.control[70:80,50:60]=2
    rgb=render(w)
    assert tuple(rgb[55,115])==COLORS[2,1]
    assert tuple(rgb[75,55])==COLORS[1,2]
    assert w.homeland[55,115]==2

def test_city_above_contested_and_terrain_hue():
    w=world(cities=True); w.disputed[110,150]=True
    assert tuple(render(w)[110,150])==(255,255,255)
    w.layers.terrain=np.full(w.background.shape,128,np.uint8); w.layers.rebuild(w.control.shape)
    rgb=render(w)
    assert rgb[40,30,0]>rgb[40,30,2]
    assert rgb[40,130,2]>rgb[40,130,0]

def markup(rgb,points=[(55,110),(200,110)]):
    im=rgb.copy(); cv2.polylines(im,[np.array(points,np.int32)],False,(255,0,255),4)
    return png_bytes(im)

def test_fresh_markup_and_old_map():
    w=world(); reference=render(w,stamp='r1 / fresh')
    masks=validate_markup(markup(reference),reference)
    assert masks[1].sum()>100
    old=render(w,stamp='r0 / old')
    with pytest.raises(ValueError,match='устарела'): validate_markup(markup(old),reference)

def test_stale_front_rejected_without_stamp():
    w=world(); old=render(w); w.control[80:140,80:130]=1
    with pytest.raises(ValueError,match='устарела'): validate_markup(markup(old),render(w))

def test_jpeg_and_wrong_size():
    w=world(); ref=render(w); b=io.BytesIO(); Image.fromarray(ref).save(b,format='JPEG')
    with pytest.raises(ValueError,match='PNG'): validate_markup(b.getvalue(),ref)
    with pytest.raises(ValueError,match='размер'): read_png(png_bytes(ref[:100]),ref.shape[:2])

def test_wrong_side_rejected():
    ref=render(world()); im=ref.copy(); cv2.line(im,(100,80),(30,80),(0,255,102),4)
    with pytest.raises(ValueError,match='другой'): validate_markup(png_bytes(im),ref,(1,))

def test_4k_and_curve_visual_only():
    w=world(); control=w.control.copy()
    frame=render(w,3840); assert frame.shape==(2640,3840,3)
    curved=Curve(frame.shape).apply(frame)
    assert curved.shape==frame.shape and not np.array_equal(curved,frame)
    assert np.array_equal(w.control,control)

def test_demo_preparation_preserves_water_and_strict_decoder():
    from prepare_demo import prepare
    im=world().background.copy()
    im[:,79:81,:3]=0
    im[90:92,:,:3]=(0,0,255)
    corrected,report=prepare(im)
    water=np.all(im[:,:,:3]==(0,0,255),axis=2)
    assert np.array_equal(im[water],corrected[water])
    assert report['changed_border_pixels']>100
    assert read_png(png_bytes(corrected)).shape==im.shape
