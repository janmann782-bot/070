"""Regression checks for source coordinates and metadata-only name updates."""
import hashlib
import json
import numpy as np
import pytest
from city_catalog import CityCatalog, name_event, present_info, catalog
from test_storage import store
from test_engine import pocket
from storage import pack_masks
from render_timeline import Timeline


def data(size=(4096,3072),places=None):
    return dict(width=size[0],height=size[1],all_places=places or [dict(name='Новомир',x=180,y=105)])


def test_native_coordinates_not_component_id_and_no_distant_or_scaled_guess():
    c=CityCatalog(data())
    assert c.find(182,104,(4096,3072))['name']=='Новомир'
    assert c.find(188,105,(4096,3072))['name']=='Новомир'
    assert c.find(189,105,(4096,3072)) is None
    assert c.find(90,52.5,(2048,1536)) is None
    assert c.find(3000,1000,(4096,3072)) is None
    overlapping=CityCatalog(data(places=[dict(name='А',x=180,y=105),dict(name='Б',x=182,y=105)]))
    assert overlapping.find(181,105,(4096,3072)) is None


def test_real_supplied_catalog_has_names_and_coordinates():
    c=catalog()
    assert c.size==(4096,3072) and len(c.places)==369
    assert c.find(2420,2413,c.size)['name']=='Разам'
    assert c.find(3475,1595,c.size)['name']=='Дестар'
    # Northern Kefir marker has no local catalog record: no invented nearest town.
    unknown=name_event(dict(id=1,x=3904,y=32),c.size)
    assert unknown['name']=='Без названия' and not unknown['named']


def test_invalid_catalog_rejected():
    with pytest.raises(ValueError):CityCatalog(data(places=[dict(name='А',x=float('nan'),y=105)]))
    with pytest.raises(ValueError):CityCatalog(data(places=[dict(name='',x=180,y=105)]))


def test_existing_cached_captures_gain_names_without_physics_or_db_changes(store,tmp_path,monkeypatch):
    import city_catalog
    import storage
    path=tmp_path/'registry.json';path.write_text(json.dumps(data((320,220))),encoding='utf-8')
    monkeypatch.setattr(city_catalog,'CATALOG_PATH',path)
    w=pocket(True);store.create(1,w)
    store.commit(1,'turn',dict(days=40,width=30,seed=0),pack_masks({}))
    t=Timeline(store,1);t.ensure()
    # Simulate the existing v1.4 presentation cache with generic number-based names.
    with store.connect() as db:
        rows=db.execute('SELECT day,delta,info FROM render_frames WHERE chat_id=1').fetchall()
        for day,delta,text in rows:
            info=json.loads(text)
            for event in info['captures']:event['name']='Населенный пункт 001'
            text=json.dumps(info)
            db.execute('UPDATE render_frames SET info=?,checksum=? WHERE chat_id=1 AND day=?',
                       (text,hashlib.sha256(delta+text.encode()).hexdigest(),day))
        before=db.execute('SELECT initial,cache,revision FROM sessions WHERE chat_id=1').fetchone()
        cached=db.execute('SELECT day,delta,info,checksum FROM render_frames WHERE chat_id=1').fetchall()
    monkeypatch.setattr(storage,'simulate',lambda *a,**k:pytest.fail('Name update called physics'))
    assert not t.ensure()
    frames=list(t.frames());events=[e for _,i in frames for e in i['captures']]
    assert len(events)==1 and events[0]['name']=='Новомир' and events[0]['named']
    assert frames[-1][1]['recent'][0]['name']=='Новомир'
    assert np.array_equal(frames[-1][0].control,store.load(1)[0].control)
    # A second catalog edit is picked up on rerender without invalidating daily deltas.
    path.write_text(json.dumps(data((320,220),[dict(name='Старовир',x=180,y=105)])),encoding='utf-8')
    assert not t.ensure()
    assert list(t.frames())[-1][1]['recent'][0]['name']=='Старовир'
    with store.connect() as db:
        assert db.execute('SELECT initial,cache,revision FROM sessions WHERE chat_id=1').fetchone()==before
        assert db.execute('SELECT day,delta,info,checksum FROM render_frames WHERE chat_id=1').fetchall()==cached


def test_unknown_never_reuses_number_name_and_presentation_does_not_mutate_input():
    event=dict(id=999,x=200,y=200,name='Населенный пункт 999',side=1,day=1)
    info=dict(source_size=[320,220],label='29.06.2057',captures=[event])
    original=json.dumps(info)
    shown=present_info(info)
    assert shown['captures'][0]['name']=='Без названия'
    assert shown['recent'][0]['label']=='29.06.2057'
    assert json.dumps(info)==original
