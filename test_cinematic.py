import asyncio
import hashlib
import json
import subprocess
import numpy as np
import pytest
from test_storage import store, attack
from test_engine import pocket
from storage import pack_masks, pack_world
from render_timeline import Timeline
from cinematic import export_cinematic, Shot, preview
from cinematic_ui import Broadcast, ease
from postprocess import process
from settlements import Settlements
from visual_settings import settings, validate
from test_bot import FakeTelegram, feed
from aiogram import Bot
from aiogram.types import CallbackQuery, User, Update, Message, Chat
from datetime import datetime, timezone
from bot import WarBot


def test_sparse_timeline_exact_and_rerender_no_physics(store,monkeypatch):
    attack(store)
    expected=[]
    store.replay(1,on_start=lambda w:expected.append((w.control.copy(),w.disputed.copy())),
                 on_day=lambda w:expected.append((w.control.copy(),w.disputed.copy())))
    timeline=Timeline(store,1);assert timeline.ensure()
    assert len(list(timeline.frames()))==7
    for (w,info),(control,grey) in zip(timeline.frames(),expected):
        assert np.array_equal(w.control,control)
        assert np.array_equal(w.disputed,grey)
    import storage
    monkeypatch.setattr(storage,'simulate',lambda *a,**k:pytest.fail('Rerender called physics'))
    assert not Timeline(store,1).ensure()
    assert np.array_equal(list(Timeline(store,1).frames())[-1][0].control,expected[-1][0])


def test_timeline_layer_after_last_day_and_undo(store):
    attack(store);t=Timeline(store,1);t.ensure();old=t.digest
    w,_=store.load(1)
    layer=np.full(w.background.shape,128,np.uint8);layer[:,:,3]=255
    from layers import png_bytes
    store.commit(1,'layer',{'name':'terrain'},png_bytes(layer))
    t=Timeline(store,1);assert t.digest!=old and t.ensure()
    w,info=list(t.frames())[-1]
    assert np.array_equal(w.layers.terrain,layer)
    store.undo(1);t=Timeline(store,1);assert t.digest==old and t.ensure()
    assert list(t.frames())[-1][0].layers.terrain is None


def test_visual_cache_corruption_rebuilds_without_touching_session(store):
    attack(store);t=Timeline(store,1);t.ensure()
    with store.connect() as db:
        original=db.execute('SELECT cache,revision FROM sessions WHERE chat_id=1').fetchone()
        db.execute("UPDATE render_frames SET delta=X'1234' WHERE chat_id=1 AND day=3")
    assert t.ensure()
    with store.connect() as db:
        assert db.execute('SELECT cache,revision FROM sessions WHERE chat_id=1').fetchone()==original


def test_city_capture_after_encirclement_and_multiframe_fade(store):
    w=pocket(True);store.create(1,w)
    cities=Settlements(w);assert len(cities.items)==1
    original=cities.owners(w)
    store.commit(1,'turn',dict(days=40,width=30,seed=0),pack_masks({}))
    t=Timeline(store,1);t.ensure()
    captures=[c for _,info in t.frames() for c in info['captures']]
    assert len(captures)==1 and captures[0]['side']==1
    assert captures[0]['day']>1 and original[captures[0]['id']]==2
    w,info=list(t.frames())[-1];shot=Shot(w,info,validate({}))
    event=dict(captures[0],time=0)
    before=shot.frame(0,[event],1,640,360)
    visible=shot.frame(.5,[event],1,640,360)
    later=shot.frame(.8,[event],1,640,360)
    expired=shot.frame(2.3,[event],1,640,360)
    assert not np.array_equal(visible,before)
    assert not np.array_equal(visible,later) # compact marker opacity changes across frames
    assert np.array_equal(before,expired)


@pytest.mark.parametrize('curved,width',[(False,640),(True,640),(False,3840)])
def test_cinematic_h264_frame_count_progress_and_no_state_change(store,tmp_path,curved,width):
    store.commit(1,'turn',dict(days=1,width=20,seed=0),pack_masks({}))
    with store.connect() as db:before=db.execute('SELECT initial,cache,revision FROM sessions WHERE chat_id=1').fetchone()
    path=tmp_path/'cinematic.mp4';updates=[]
    info=export_cinematic(store,1,path,10,width,dict(curved=curved),progress=lambda *a:updates.append(a))
    assert info['frames']==10+4+22
    assert [d for d,_,stage in updates if stage.startswith('Рендер и кодирование:')]==list(range(1,37))
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)]))['streams'][0]
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    assert int(stream['nb_read_frames'])==36 and stream['width']==width
    assert stream['height']==(round(width*9/16)//2*2)
    data=path.read_bytes();assert data.find(b'moov')<data.find(b'mdat')
    with store.connect() as db:assert db.execute('SELECT initial,cache,revision FROM sessions WHERE chat_id=1').fetchone()==before


def test_grade_repeatable_no_mutation_and_settings_persist(store):
    rgb=np.random.default_rng(1).integers(0,256,(80,100,3),dtype=np.uint8);original=rgb.copy()
    a=process(rgb);assert np.array_equal(a,process(rgb)) and np.array_equal(rgb,original)
    assert np.array_equal(process(rgb,False),rgb) and a.dtype==np.uint8
    settings(store,1,dict(city_glow=False,seconds_per_day=.6,crf=16))
    from storage import Store
    assert settings(Store(store.path),1)['seconds_per_day']==.6
    with pytest.raises(ValueError):settings(store,1,dict(crf=100))
    with pytest.raises(ValueError):validate(dict(unknown=True))
    before=store.load(1)[0].control.copy();preview(store,1,640)
    assert np.array_equal(store.load(1)[0].control,before)


def test_effects_command_and_owner_callback(store):
    async def scenario():
        transport=FakeTelegram();bot=Bot('123456:abcdefghijklmnopqrstuvwxyzABCDEFGHI',session=transport)
        app=WarBot(store,owner=7)
        await feed(app,bot,'/effects seconds_per_day 0.7')
        assert settings(store,1)['seconds_per_day']==.7
        await feed(app,bot,'/effects')
        assert transport.sent[-1].reply_markup.inline_keyboard[0][0].callback_data=='visual:notifications'
        msg=Message(message_id=1,date=datetime.now(timezone.utc),chat=Chat(id=1,type='private'),text='settings')
        for uid,expected in [(8,True),(7,False)]:
            query=CallbackQuery(id=str(uid),from_user=User(id=uid,is_bot=False,first_name='Test'),chat_instance='c',message=msg,data='visual:city_glow')
            await app.dp.feed_update(bot,Update(update_id=uid,callback_query=query))
            assert settings(store,1)['city_glow']==expected
        await bot.session.close()
    asyncio.run(scenario())


def test_easing_and_width_validation():
    assert ease(-1)==0 and ease(2)==1 and ease(.5)==.5
    with pytest.raises(ValueError):Broadcast(100)
    with pytest.raises(ValueError):validate(dict(seconds_per_day=float('nan')))


def test_simultaneous_captures_use_mint_names_and_no_map_cards():
    ui=Broadcast();info=dict(source_size=[320,220])
    rgb=np.zeros((220,320,3),np.uint8)
    events=[dict(id=1,x=80,y=100,side=1,time=0,day=1,name='Новомир'),
            dict(id=2,x=220,y=100,side=2,time=0,day=1,name='Старовир')]
    frame=ui.compose(rgb,info,events,.5)
    mx,my,mw,mh=ui.actual_map
    for e in events:
        x=round(mx+e['x']/320*mw);y=round(my+e['y']/220*mh)
        # Control changes use the incoming side's own color, not universal mint.
        pixel=frame[y-3,x-3]
        if e['side']==1:
            assert pixel[0]>pixel[1]
        else:
            assert pixel[2]>pixel[0]
        assert pixel.max()>60
    # No lower-left notification cards or duplicate lower-right map.
    without=ui.compose(rgb,info,events,.5,dict(notifications=False,city_glow=False))
    assert np.array_equal(frame[850:1050,500:900],without[850:1050,500:900])
    assert np.array_equal(frame[850:1050,1668:1886],without[850:1050,1668:1886])
    assert not np.array_equal(frame,without)


def test_sidebar_ignores_removed_statistics_but_keeps_date_day_and_captures():
    ui=Broadcast();rgb=np.zeros((220,320,3),np.uint8)
    info=dict(label='06.07.2057',days=8,source_size=[320,220],recent=[
        dict(name='Мятное',side=1,label='06.07.2057',day=8)])
    frame=ui.compose(rgb,info)
    changed=dict(info,stats=dict(kefir=123456,yogurt=789,contested=50000,
        kefir_occupation=400,yogurt_occupation=900),territory=12345,
        cities={1:99,2:25,0:8})
    assert np.array_equal(frame,ui.compose(rgb,changed))
    for update in [dict(label='07.07.2057'),dict(days=9),dict(recent=[])]:
        assert not np.array_equal(frame,ui.compose(rgb,info|update))
    # Footer and old counter area stay empty after moving the feed under the date.
    assert np.all(frame[750:1080,0:470]==np.array([7,11,8]))
