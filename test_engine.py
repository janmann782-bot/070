import os
os.environ.setdefault('NUMBA_NUM_THREADS','2')
import numpy as np
import cv2
import pytest
from layers import Layers
from state import World
from engine import simulate,prepare,update_contested
from palette import COLORS

def world(shape=(220,320),axis='horizontal',cities=False,roads=False):
    h,w=shape; yy,xx=np.mgrid[:h,:w]
    own=xx<80 if axis=='horizontal' else yy<65 if axis=='vertical' else xx+yy<150
    rgba=np.empty((h,w,4),np.uint8); rgba[:,:,:3]=COLORS[2,2]; rgba[:,:,3]=255
    rgba[own,:3]=COLORS[1,1]
    layer=Layers()
    if roads:
        layer.roads=np.zeros((h,w,4),np.uint8)
        layer.roads[h//2,:,:3]=255; layer.roads[h//2,:,3]=255
    if cities:
        layer.cities=np.zeros((h,w,4),np.uint8)
        cv2.circle(layer.cities,(150,h//2),3,(255,255,255,255),-1)
    return World.create(rgba,'2057-06-28',layer)

def line(shape,points):
    mask=np.zeros(shape,np.uint8)
    cv2.polylines(mask,[np.array(points,np.int32)],False,1,3)
    return mask.astype(bool)

@pytest.mark.parametrize('axis,points',[
 ('horizontal',[(55,110),(245,110)]),
 ('vertical',[(160,45),(160,185)]),
 ('diagonal',[(55,55),(210,165)]),
 ('horizontal',[(55,110),(105,108),(145,75),(190,82),(235,135)]),
 ('horizontal',[(55,110),(100,80),(135,75),(165,100),(180,150),(215,155),(250,115)]),
])
def test_arbitrary_geometry_advances(axis,points):
    w=world(axis=axis); initial=w.control.copy(); home=w.homeland.copy()
    simulate(w,{1:line(w.control.shape,points)},20,48,42)
    assert ((w.control==1)&(initial==2)).sum()>100
    assert np.array_equal(w.homeland,home) and not w.homeland.flags.writeable
    assert w.elapsed==20 and w.date.isoformat()=='2057-07-18'

def test_multiple_lines():
    w=world(); a=line(w.control.shape,[(55,50),(230,50)])|line(w.control.shape,[(55,180),(235,180)])
    simulate(w,{1:a},15,40,9)
    assert (w.control[20:80,100:]==1).sum()>200
    assert (w.control[155:205,100:]==1).sum()>200
    assert w.control[110,160]==2

def test_yogurt_and_simultaneous():
    w=world(); old=w.control.copy()
    simulate(w,{2:line(w.control.shape,[(115,110),(20,110)])},15,45,111)
    # After the canon-border fix Yogurtstan cannot occupy homeland Kefirstan.
    assert ((w.control==2)&(w.homeland==1)).sum()==0
    w=world()
    simulate(w,{1:line(w.control.shape,[(55,95),(220,105)]),2:line(w.control.shape,[(130,105),(20,95)])},20,60,12)
    assert w.battle_age.max()>=2 and w.disputed.any()
    assert set(np.unique(w.control))<={1,2}

def test_road_speed_and_flanks():
    a=world(); b=world(roads=True)
    mask=line(a.control.shape,[(55,110),(270,110)])
    simulate(a,{1:mask},16,60,43); simulate(b,{1:mask},16,60,43)
    xp=lambda w:int(np.where(w.control[110]==1)[0].max())
    assert xp(b)>xp(a)+20
    assert xp(a)>100  # off-road motion really occurs
    assert xp(b)>int(np.where(b.control[130]==1)[0].max())+15
    assert (b.control[103:108,100:]==1).sum()>100  # not glued to 1 px road

def test_water_barrier_and_no_diagonal_corner_cut():
    w=world(); w.background[:,135:140,:3]=(0,0,255)
    w=World.create(w.background,'2057-06-28')
    simulate(w,{1:line(w.control.shape,[(55,110),(270,110)])},30,60,44)
    assert np.all(w.control[:,135:140]==0)
    assert not np.any(w.control[:,140:]==1)

def pocket(city):
    w=world(); w.control[:,:280]=1; w.control[85:125,160:200]=2
    if city:
        layer=np.zeros(w.background.shape,np.uint8); cv2.circle(layer,(180,105),3,(255,255,255,255),-1)
        w.layers.cities=layer; w.layers.rebuild(w.control.shape)
    return w

def test_empty_pocket_surrenders():
    w=pocket(False); simulate(w,{},1,30,0)
    assert np.all(w.control[85:125,160:200]==1)

def test_city_pocket_holds_then_falls():
    w=pocket(True); simulate(w,{},1,30,0)
    assert w.control[105,180]==2 and w.encirclement_age[105,180]==1
    simulate(w,{},40,30,0)
    assert w.control[105,180]==1

def test_deblockade_resets_timer():
    w=pocket(True); simulate(w,{},8,30,0)
    w.control[100:110,190:290]=2
    simulate(w,{},1,30,0)
    assert w.encirclement_age[105,180]==0

def test_ordinary_contested_and_local_prolonged():
    w=world(); simulate(w,{},1,20,0)
    assert np.all(w.disputed.sum(axis=1)==1)
    cities=np.zeros(w.background.shape,np.uint8); cv2.circle(cities,(80,110),3,(255,255,255,255),-1)
    w.layers.cities=cities; w.layers.rebuild(w.control.shape)
    simulate(w,{},28,20,0)
    counts=w.disputed.sum(axis=1)
    assert counts[110]>=5 and counts[110]<=11
    assert np.all(counts[:65]==1) and (counts>4).mean()<.2

def test_terrain_is_physics():
    a=world(); b=world()
    b.layers.terrain=np.full(b.background.shape,255,np.uint8); b.layers.rebuild(b.control.shape)
    mask=line(a.control.shape,[(55,110),(270,110)])
    simulate(a,{1:mask},15,55,4); simulate(b,{1:mask},15,55,4)
    assert ((a.control==1)&(a.homeland==2)).sum()>((b.control==1)&(b.homeland==2)).sum()*1.2

def test_detached_and_loop_orders_rejected():
    w=world()
    with pytest.raises(ValueError,match='своей'): prepare(w,{1:line(w.control.shape,[(170,80),(270,80)])},20,50,9)
    mask=np.zeros(w.control.shape,np.uint8); cv2.circle(mask,(80,110),35,1,3)
    with pytest.raises(ValueError,match='Замкнутая'): prepare(w,{1:mask.astype(bool)},20,50,9)

def test_deterministic_daily_frames():
    a=world(roads=True,cities=True); b=world(roads=True,cities=True)
    mask=line(a.control.shape,[(55,110),(250,125)])
    frames=[]; other=[]
    simulate(a,{1:mask},12,50,908,on_day=lambda w:frames.append(w.control.copy()))
    simulate(b,{1:mask},12,50,908,on_day=lambda w:other.append(w.control.copy()))
    assert all(np.array_equal(x,y) for x,y in zip(frames,other))
    assert np.array_equal(a.battle_age,b.battle_age)
    assert np.array_equal(a.disputed,b.disputed)
    # Daily gains fluctuate rather than constant-rate expansion.
    areas=np.array([np.count_nonzero(f==1) for f in frames]); assert np.std(np.diff(areas))>10

def test_nonpalette_pixels_remain_impassable():
    w=world(); w.background[:,135:138,:3]=(0,0,0)
    w=World.create(w.background,'2057-06-28')
    simulate(w,{1:line(w.control.shape,[(55,110),(250,110)])},12,60,19)
    assert np.all(w.control[:,135:138]==0) and not np.any(w.control[:,138:]==1)

def test_road_junctions_are_detected_and_local():
    w=world(); roads=np.zeros(w.background.shape,np.uint8)
    cv2.line(roads,(30,110),(290,110),(255,255,255,255),2)
    cv2.line(roads,(80,30),(80,200),(255,255,255,255),2)
    w.layers.roads=roads; w.layers.rebuild(w.control.shape)
    assert w.layers.road_nodes[110,80]
    assert not w.layers.road_nodes[110,200]
    simulate(w,{},24,20,0)
    assert w.disputed[110].sum()>3 and w.disputed[40].sum()==1
