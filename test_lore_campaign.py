"""Checks for the official timeline film and the no-invasion rule."""
from datetime import date
from types import SimpleNamespace
import numpy as np

from layers import Layers
from state import World
from palette import COLORS
import lore_campaign


def fixture(monkeypatch):
    h,w=200,320
    rgba=np.empty((h,w,4),np.uint8)
    rgba[:,:,:3]=COLORS[(2,2)]
    rgba[:,:80,:3]=COLORS[(1,1)]
    rgba[:,:,3]=255
    world=World.create(rgba,'2057-06-28',Layers())
    names=[operation[0] for operation in lore_campaign.OPERATIONS]
    known=[dict(name=n,x=95+15*(i%8),y=30+25*(i//8)) for i,n in enumerate(names)]
    dummy=[dict(name='Тестовое село '+str(i),x=120+i%120,y=90+i%60) for i in range(115)]
    cat=SimpleNamespace(size=(w,h),places=known+dummy)
    monkeypatch.setattr(lore_campaign,'catalog',lambda:cat)
    return world


def test_strict_border_and_parallel_fronts(monkeypatch):
    world=fixture(monkeypatch)
    campaign=lore_campaign.Campaign(world)
    campaign.at(date(2057,6,28))
    assert np.array_equal(world.control,world.homeland)
    campaign.at(date(2057,7,18))
    assert np.count_nonzero((world.control==1)&(world.homeland==2))>0
    assert not np.any((world.control==2)&(world.homeland==1))
    campaign.at(date(2058,12,31))
    assert np.count_nonzero((world.control==1)&(world.homeland==2))>0
    assert not np.any((world.control==2)&(world.homeland==1))
    campaign.at(date(2060,11,12))
    assert np.array_equal(world.control,world.homeland)


def test_named_capture_events_track_owner(monkeypatch):
    world=fixture(monkeypatch)
    campaign=lore_campaign.Campaign(world)
    campaign.at(date(2057,7,2))
    info=campaign.at(date(2057,7,3))
    assert isinstance(info['captures'],list)
    for event in info['captures']:
        assert event['name'] and event['side'] in (1,2)
        assert event['label']=='03.07.2057'


def test_scene_calendar_complete():
    days=lore_campaign.dates()
    assert days[0]==lore_campaign.START and days[-1]==lore_campaign.END
    assert len(days)>150
    assert days==sorted(set(days))


def test_canonical_native_map_assets_and_campaign():
    # Integration check: real 4096x3072 political base, named locations and
    # the historical battlefield layers must work without relying on a bot token.
    world=lore_campaign.load_world()
    assert world.control.shape==(3072,4096)
    campaign=lore_campaign.Campaign(world)
    assert len(campaign.paths)>=8
    for when in (
        date(2057,6,28),
        date(2057,7,18),
        date(2058,12,31),
        date(2059,4,4),
        date(2060,11,12),
    ):
        info=campaign.at(when)
        assert info['label']==when.strftime('%d.%m.%Y')
        assert not np.any((world.control==2)&(world.homeland==1))
    assert np.array_equal(world.control,world.homeland)


def test_4k_mode_keeps_map_10fps_but_ui_30fps(monkeypatch,tmp_path):
    """A 30 FPS stream holds each political map state for three UI frames."""
    calls=[]
    movie_frames=[]
    class CampaignStub:
        def __init__(self, world):
            pass
        def at(self,when):
            calls.append(when)
            return dict(label=when.strftime('%d.%m.%Y'),captures=[],
                        source_size=(320,220),recent=[])
    class ShotStub:
        def __init__(self,world,info,options,width):
            self.color=len(calls)
        def frame(self,now,events,phase,width,height):
            movie_frames.append((self.color,now))
            return np.full((height,width,3),self.color,dtype=np.uint8)
    monkeypatch.setattr(lore_campaign,'load_world',lambda:object())
    monkeypatch.setattr(lore_campaign,'Campaign',CampaignStub)
    monkeypatch.setattr(lore_campaign,'Shot',ShotStub)
    monkeypatch.setattr(lore_campaign,'get_settings',lambda *a:{})
    monkeypatch.setattr(lore_campaign,'dates',lambda:[
        date(2057,6,28),date(2057,7,4)])
    path=tmp_path/'split_fps.mp4'
    report=lore_campaign.export_lore(None,0,path,fps=30,width=640,map_fps=10)
    assert path.stat().st_size>0
    assert report['fps']==report['ui_fps']==30
    assert report['map_fps']==10
    assert report['frames']==12
    assert report['map_updates']==4
    assert [x[0] for x in movie_frames]==[1]*3+[2]*3+[3]*3+[4]*3
    assert [round(x[1]*30) for x in movie_frames]==list(range(12))
