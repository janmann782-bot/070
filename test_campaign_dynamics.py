"""Proof of non-repeating fronts, land-only advances and actual sealed pockets."""
from datetime import date, timedelta

import cv2
import numpy as np

from campaign_dynamics import (path_lanes,lane_progress,paint_corridor,skirmishes,
                               apply_skirmishes,siege_centres,apply_sieges,
                               disputed_front,novomir_hold)


def home_map():
    home=np.zeros((250,420),np.uint8)
    home[:, :85]=1
    home[:, 85:]=2
    return home


def test_advances_not_periodic_mirror_reversals():
    arrive=date(2058,12,31)
    withdraw=date(2059,9,30)
    for lane in range(3):
        values=[lane_progress(arrive-timedelta(days=42-d),arrive,withdraw,60,.77,
                              'Йогуртск',lane) for d in range(47)]
        assert all(values[k]<=values[k+1]+1e-7 for k in range(45))
        assert values[-1]>.6
        assert len({round(n,2) for n in values})>10
    values=[lane_progress(withdraw-timedelta(days=60-d),arrive,withdraw,60,.77,
                          'Йогуртск',0) for d in range(61)]
    assert all(values[k]>=values[k+1]-1e-7 for k in range(60))
    assert values[-1]==0


def test_three_lane_front_has_nonuniform_boundary():
    home=home_map();c=home.copy()
    pts=np.array([(85,110),(130,113),(220,122),(300,140)],np.int32)
    lanes=path_lanes(pts,42,'Йогуртск')
    sl=(slice(0,250),slice(0,420))
    for when in [date(2058,12,5),date(2058,12,19),date(2059,1,10)]:
        before=c.copy();c[:]=home
        paint_corridor(c,home,sl,lanes,when,date(2058,12,31),
                       date(2059,9,30),60,.77,'Йогуртск',42)
        assert (c[100:160,100:290]==1).any()
        assert not np.any((c==2)&(home==1))
        assert np.count_nonzero(c!=before)>0
    boundary=(c==1).sum(axis=0)[110:220]
    assert boundary.max()-boundary.min()>10


def test_cordon_closes_a_real_pocket_and_relief_opens_it():
    home=home_map()
    places={'Йогуртск':dict(x=255,y=90),'Дестар':dict(x=285,y=182)}
    siege=siege_centres(home,places)
    assert len(siege)==2
    c=home.copy()
    hot=np.zeros(home.shape,bool)
    apply_sieges(c,home,date(2059,4,4),siege,hot)
    # Both centres remain Yogurtstan and are ringed by contiguous Kefir units.
    for _name,_start,_closed,_broken,x,y,r in siege:
        assert c[y,x]==2 and c[y,x+r]==1
        n,labels=cv2.connectedComponents((c==2).astype(np.uint8),connectivity=4)
        assert labels[y,x]!=labels[50,390]
    assert hot.any()
    grey=disputed_front(c,home,hot)
    assert grey.any() and grey.dtype==bool
    c=home.copy();hot[:]=False
    apply_sieges(c,home,date(2059,9,30),siege,hot)
    assert np.array_equal(c,home) and not hot.any()


def test_novomir_ceasefire_keeps_small_kefir_area_but_not_city():
    home=home_map()
    x,y=210,115
    zone=novomir_hold(home,{'Новомир':dict(x=x,y=y)},[(85,y)])
    assert zone is not None
    sl,mask=zone
    assert mask.sum()>50
    c=home.copy()
    c[sl][mask]=1
    assert c[y,x]==2
    assert not np.any((c==2)&(home==1))
    assert np.count_nonzero((c==1)&(home==2))>50


def test_skirmishes_are_staggered_one_off_actions():
    home=home_map()
    sl=(slice(0,250),slice(0,420))
    lane=path_lanes(np.array([(85,110),(140,110),(200,122),(260,126)]),
                    46,'Старовир')
    entry=('Старовир',date(2057,7,18),date(2059,8,28),46,140,1.,sl,
           np.array([(85,110),(140,110),(200,122),(260,126)]),lane)
    events=skirmishes([entry])
    assert len(events)>=5
    assert len({e[0] for e in events})>=5
    c=home.copy()
    paint_corridor(c,home,sl,lane,date(2058,7,1),entry[1],
                   entry[2],140,1.,'Старовир',46)
    touched=False
    for start,span,*_ in events:
        test=c.copy()
        hot=np.zeros(home.shape,bool)
        apply_skirmishes(test,home,start+timedelta(days=span//2),events,hot)
        assert not np.any((test==2)&(home==1))
        touched |= np.count_nonzero(test!=c)>0
    assert touched, "local operations must change real control, not just show battle text"
