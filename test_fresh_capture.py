"""Daily capture highlight: transient, deterministic and backward compatible."""
import io
import numpy as np
import pytest
from test_engine import world,line,pocket
from test_storage import store,attack
from engine import simulate
from renderer import render
from palette import FRESH_CAPTURE,COLORS,CONTESTED
from storage import pack_world,unpack_world,pack_masks
from orders import validate_markup
from test_renderer import markup

@pytest.mark.parametrize('side',[1,2])
def test_only_that_days_gains_are_highlighted(side):
    w=world(); prev=w.control.copy(); home=w.homeland.copy()
    points=[(55,110),(240,110)] if side==1 else [(120,110),(20,110)]
    frames=[]; visible_counts=[]
    def day(current):
        nonlocal prev
        delta=current.control!=prev
        assert np.array_equal(current.fresh_capture,delta)
        rgb=render(current)
        visible=delta&~current.disputed&(current.control==side)
        visible_counts.append(int(visible.sum()))
        assert np.all(rgb[visible]==FRESH_CAPTURE[side])
        frames.append(delta.copy());prev=current.control.copy()
    simulate(w,{side:line(w.control.shape,points)},8,45,991,on_day=day)
    assert np.array_equal(w.homeland,home)
    assert sum(int(mask.sum()) for mask in frames)>0
    assert sum(visible_counts)>0
    assert np.any(np.logical_or.reduce(frames[:-1])&~frames[-1])  # earlier gains no longer highlighted; pauses allowed
    simulate(w,{},1,45,0)
    assert not w.fresh_capture.any()
    normal=(w.control==side)&(w.homeland==3-side)&~w.disputed
    assert normal.any() and np.all(render(w)[normal]==COLORS[3-side,side])


def test_grey_and_cities_remain_above_highlight():
    w=world(cities=True);w.control[90:130,140:170]=1
    w.fresh_capture=np.zeros_like(w.control,bool);w.fresh_capture[90:130,140:170]=True
    w.disputed[100,150]=True
    rgb=render(w)
    assert tuple(rgb[100,150])==CONTESTED
    assert tuple(rgb[110,150])==(255,255,255)
    assert tuple(rgb[120,160])==FRESH_CAPTURE[1]


def test_pocket_fall_is_a_capture_on_that_day():
    w=pocket(False);simulate(w,{},1,20,0)
    assert np.all(w.fresh_capture[85:125,160:200])
    simulate(w,{},1,20,0)
    assert not w.fresh_capture.any()


def test_fresh_colors_do_not_become_order_lines():
    w=world();w.fresh_capture=np.ones_like(w.control,bool)
    reference=render(w)
    masks=validate_markup(markup(reference),reference,(1,))
    assert set(masks)=={1} and masks[1].sum()<1000


def test_restart_replay_undo_and_doctor_preserve_daily_highlight(store):
    first=attack(store)
    assert first.fresh_capture.any()
    cached=store.load(1)[0];replayed=store.replay(1)
    assert np.array_equal(first.fresh_capture,cached.fresh_capture)
    assert np.array_equal(first.fresh_capture,replayed.fresh_capture)
    store.commit(1,'turn',{'days':1,'width':20,'seed':0},pack_masks({}))
    assert not store.load(1)[0].fresh_capture.any()
    assert np.array_equal(store.undo(1).fresh_capture,first.fresh_capture)
    corrupted=store.load(1)[0];corrupted.fresh_capture[:]=False
    with store.connect() as db:db.execute('UPDATE sessions SET cache=? WHERE chat_id=1',(pack_world(corrupted),))
    same,repaired=store.doctor(1)
    assert not same and np.array_equal(repaired.fresh_capture,first.fresh_capture)


def test_legacy_blob_loads_and_render_stays_unchanged():
    w=world()
    legacy=pack_world(w)
    with np.load(io.BytesIO(legacy),allow_pickle=False) as a:
        assert 'fresh_capture' not in a
    loaded=unpack_world(legacy)
    assert loaded.fresh_capture is None
    assert np.array_equal(render(loaded),render(w))
