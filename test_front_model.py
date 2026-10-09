"""Observable road priority, irregular sectors, rare prolonged battle zones and migration."""
import numpy as np
import cv2
import config
from test_engine import world,line
from test_storage import store,assert_same
from engine import simulate,road_guidance,update_contested
from storage import pack_masks,pack_world


def legacy(w):
    settings={k:v for k,v in w.layers.settings.items() if k not in ['FRONT_MODEL','ROAD_PRIORITY','WIDTH_VARIATION','WIDE_BATTLE_DAYS','WIDE_BATTLE_CHANCE']}
    settings.update(ROAD_COST=.55,ROAD_FALLOFF=12.)
    w.layers.settings=settings;w.layers.rebuild(w.control.shape)
    return w


def test_nearest_connected_road_is_selected():
    shape=(240,360);road=np.zeros(shape,bool);road[70,:]=True;road[150,:]=True
    along=np.broadcast_to(np.arange(shape[1],dtype=np.float32),shape)
    guide=road_guidance(road,np.ones(shape,bool),along,np.array([100,55]),300,80)
    assert guide[70,250]==0 and guide[150,250]>70


def test_long_operation_uses_nearby_offset_road_with_offroad_flanks():
    new=world(shape=(300,420));roads=np.zeros(new.background.shape,np.uint8)
    cv2.line(roads,(0,130),(419,130),(255,255,255,255),2)
    new.layers.roads=roads;new.layers.rebuild(new.control.shape)
    old=legacy(world(shape=(300,420)));old.layers.roads=roads;old.layers.rebuild(old.control.shape)
    mask=line(new.control.shape,[(55,95),(360,95)])
    newframes={};oldframes={}
    simulate(new,{1:mask},30,85,823,on_day=lambda w:newframes.update({w.elapsed:w.control.copy()}))
    simulate(old,{1:mask},30,85,823,on_day=lambda w:oldframes.update({w.elapsed:w.control.copy()}))
    depth=lambda a,y:int(np.where(a[y]==1)[0].max())
    assert depth(newframes[10],130)>depth(oldframes[10],130)+25
    assert depth(newframes[10],130)>depth(newframes[10],80)+25
    assert depth(newframes[20],115)>depth(newframes[10],115)+15
    assert np.count_nonzero(new.control[112:120,120:]==1)>100


def test_width_varies_between_supplied_sectors_and_tip_does_not_make_a_bulb():
    w=world(shape=(400,650));mask=line(w.control.shape,[(55,200),(570,200)])
    simulate(w,{1:mask},40,90,293)
    widths=(w.control[:,120:400]==1).sum(axis=0)
    assert np.quantile(widths,.9)>np.quantile(widths,.1)*1.35
    assert widths.max()-widths.min()>25
    assert not np.any(w.control[:,585:]==1)


def test_u_turn_keeps_its_proximal_route():
    w=world(shape=(300,420))
    mask=line(w.control.shape,[(55,80),(170,80),(240,120),(230,185),(150,200),(105,180)])
    simulate(w,{1:mask},30,45,42)
    assert (w.control[160:195,215:240]==1).sum()>50


def test_seven_pixel_meeting_zones_are_rare_local_and_require_time():
    w=world(shape=(1536,320));collision=np.zeros(w.control.shape,bool);collision[100:1400,78:82]=True
    for _ in range(12):update_contested(w,collision)
    assert w.disputed.sum(axis=1).max()<=6
    for _ in range(28):update_contested(w,collision)
    widths=w.disputed.sum(axis=1)
    assert widths.max()>=7 and widths.max()<=10
    assert 0<(widths>=7).mean()<.3
    assert np.all(widths[:80]==1) and np.all(widths[1420:]==1)
    assert w.meeting_age.max()==40
    for _ in range(25):update_contested(w)
    assert w.meeting_age.max()==0 and w.disputed.sum(axis=1).max()==1


def test_legacy_turns_and_automatic_next_turn_upgrade_survive_undo(store):
    old=legacy(world());store.create(1,old)
    params={'days':8,'width':50,'seed':42};payload=pack_masks({1:line(old.control.shape,[(55,110),(245,110)])})
    # Historical event made by the previous release; no settings switch in params.
    StoreApply=type(store).apply
    StoreApply(old,'turn',params,payload)
    import json
    with store.connect() as db:
        db.execute('INSERT INTO operations(chat_id,kind,params,payload) VALUES(?,?,?,?)',(1,'turn',json.dumps(params),payload))
        db.execute('UPDATE sessions SET cache=? WHERE chat_id=1',(pack_world(old),))
    assert_same(old,store.replay(1));assert store.replay(1).meeting_age is None
    upgraded=store.commit(1,'turn',{'days':3,'width':20,'seed':0},pack_masks({}))
    assert upgraded.layers.settings['FRONT_MODEL']==2
    assert 'physics' in store.events(1)[-1][1]
    assert_same(upgraded,store.replay(1));assert store.doctor(1)[0]
    restored=store.undo(1)
    assert_same(restored,old) and restored.layers.settings.get('FRONT_MODEL',1)==1
    assert restored.meeting_age is None


def test_meeting_age_is_cached_and_replayed_pixel_perfect(store):
    w,_=store.load(1)
    masks={1:line(w.control.shape,[(55,95),(220,105)]),2:line(w.control.shape,[(130,105),(20,95)])}
    result=store.commit(1,'turn',{'days':30,'width':60,'seed':12},pack_masks(masks))
    assert result.meeting_age.max()>0
    assert np.array_equal(result.meeting_age,store.load(1)[0].meeting_age)
    assert np.array_equal(result.meeting_age,store.replay(1).meeting_age)
    assert np.array_equal(result.disputed,store.replay(1).disputed)
    assert store.doctor(1)[0]
