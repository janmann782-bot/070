import os
import tempfile
import numpy as np
import pytest
from storage import Store,pack_masks,pack_world,unpack_world
from test_engine import world,line
from layers import png_bytes

@pytest.fixture
def store():
    fd,path=tempfile.mkstemp(suffix='.sqlite3'); os.close(fd)
    s=Store(path); s.create(1,world())
    yield s
    os.unlink(path)

def attack(store,seed=9):
    w,_=store.load(1)
    return store.commit(1,'turn',{'days':6,'width':45,'seed':seed},pack_masks({1:line(w.control.shape,[(55,110),(240,110)])}))

def assert_same(a,b):
    assert a.elapsed==b.elapsed
    for k in ['control','homeland','battle_age','encirclement_age','disputed']:
        assert np.array_equal(getattr(a,k),getattr(b,k)),k

def test_replay_and_undo(store):
    first=attack(store); store.commit(1,'turn',{'days':5,'width':30,'seed':0},pack_masks({}))
    cached,_=store.load(1); assert_same(cached,store.replay(1))
    undone=store.undo(1); assert_same(first,undone)
    assert_same(first,store.replay(1))

def test_doctor_repairs_corrupt_cache(store):
    good=attack(store)
    bad,_=store.load(1); bad.control[110,120]=2 if bad.control[110,120]==1 else 1
    with store.connect() as db: db.execute('UPDATE sessions SET cache=? WHERE chat_id=1',(pack_world(bad),))
    same,repaired=store.doctor(1); assert not same; assert_same(good,repaired)
    with store.connect() as db: db.execute("UPDATE sessions SET cache=X'123456' WHERE chat_id=1")
    same,repaired=store.doctor(1); assert not same; assert_same(good,repaired)
    assert store.doctor(1)[0]

def test_layer_history_is_authoritative(store):
    w,_=store.load(1); roads=np.zeros(w.background.shape,np.uint8); roads[110,:,3]=255
    store.commit(1,'layer',{'name':'roads'},png_bytes(roads))
    with_road=attack(store)
    store.commit(1,'layer',{'name':'roads'})
    replay=store.replay(1)
    assert_same(with_road,replay) and replay.layers.roads is None
    restored=store.undo(1); assert restored.layers.roads is not None
    assert_same(with_road,restored)

def test_failed_operation_is_atomic(store):
    before,_=store.load(1)
    with pytest.raises(ValueError):
        store.commit(1,'turn',{'days':10,'width':40,'seed':9},pack_masks({1:line(before.control.shape,[(200,80),(280,80)])}))
    assert store.events(1)==[]; assert_same(before,store.load(1)[0])
    with pytest.raises(ValueError,match='размер'): store.commit(1,'layer',{'name':'terrain'},png_bytes(np.zeros((30,40,4),np.uint8)))
    assert store.events(1)==[]

def test_pending_survives_restart_and_newmap_no_session(store):
    from renderer import render
    w,rev=store.load(1); ref=render(w)
    store.set_pending(1,'turn',{'days':2},ref,rev)
    assert Store(store.path).pending(1)[1]['days']==2
    store.set_pending(99,'newmap',{'date':'2057-06-28'})
    assert store.pending(99)[0]=='newmap'
    store.cancel(99); assert store.pending(99) is None

def test_physics_settings_are_saved_not_read_from_future_config(store):
    import config
    initial=attack(store)
    old=config.ROAD_COST
    try:
        config.ROAD_COST=.02
        assert_same(initial,store.replay(1))
        assert store.load(1)[0].layers.settings['ROAD_COST']==old
    finally: config.ROAD_COST=old
