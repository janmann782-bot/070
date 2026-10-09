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
