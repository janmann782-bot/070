"""Lore-grounded standalone campaign film. Does not overwrite interactive saves.

A bounded sequence of geographically constrained local fronts, not an invented
chronological list of exact village captures. Canon anchors are documented below.
"""
from datetime import date, timedelta
from pathlib import Path
import math
import subprocess

import cv2
import numpy as np

import config
from state import World
from layers import Layers, read_png
from renderer import render
from cinematic import Shot
from cinematic_ui import SIDE_COLORS
from visual_settings import settings as get_settings, validate
from city_catalog import catalog
from exporter import ffmpeg_executable
from campaign_dynamics import (path_lanes, paint_corridor, skirmishes,
                               apply_skirmishes, siege_centres, apply_sieges,
                               novomir_hold, disputed_front)


START = date(2057, 6, 28)
END = date(2060, 11, 12)
# Name, canonical-or-approximate endpoint, withdrawal anchor, corridor width,
# withdraw days, fraction (sub-1 = attempt / siege, not capture).
# Only explicitly documented canonical dates are exact; other dates mark
# cinematographic interpolation and MUST NOT be presented as new lore.
OPERATIONS = [
    ('Новомост', '2057-07-03', '2060-02-07', 53, 120, 1.0),
    ('Йогуртград', '2057-07-10', '2059-10-18', 54, 140, 1.0),
    ('Новомир', '2057-07-05', '2060-07-19', 43, 95, 1.0),
    ('Йогуртавск', '2057-07-07', '2059-09-30', 42, 150, .65),
    ('Старовир', '2057-07-18', '2059-08-28', 48, 140, 1.0),
    ('Великий Йогурт', '2057-07-21', '2059-11-11', 38, 115, 1.0),
    ('Кукарекун', '2057-08-22', '2059-09-30', 23, 150, 1.0),
    ('Нижний Златозерь', '2057-07-25', '2059-09-30', 41, 145, .57),
    ('Свитлодолир', '2058-09-04', '2059-05-14', 49, 75, 1.0),
    ('Двуречье', '2058-11-19', '2059-04-04', 38, 70, 1.0),
    ('Дестар', '2058-12-28', '2059-04-04', 57, 52, .80),
    ('Йогуртск', '2058-12-31', '2059-04-04', 65, 60, .77),
]
# Small yogurt counter-actions occur even during kefir advances.
# These are indicative localized motion for cinematic continuity.
RAIDS = [
    ('Новомост', '2057-08-08', 13, 36),
    ('Кукарекун', '2057-08-18', 5, 24),
    ('Йогуртград', '2058-01-11', 7, 43),
    ('Йогуртсавск', '2058-05-12', 10, 46),
    ('Свитлодолир', '2058-08-18', 8, 52),
    ('Йогуртск', '2059-04-04', 34, 110),
    ('Дестар', '2059-04-04', 34, 90),
]


def bounds(shape, points, margin):
    p = np.asarray(points, dtype=np.int32)
    return (max(0,int(p[:,1].min())-margin),
            min(shape[0],int(p[:,1].max())+margin+1),
            max(0,int(p[:,0].min())-margin),
            min(shape[1],int(p[:,0].max())+margin+1))


def load_world():
    root = config.ROOT
    # The raw political artwork contains a thick non-territory outline.
    # The prepared source restores only 3,164 contact pixels of this outline;
    # without it the armies have no legal shared land border to advance across.
    base = read_png((root/'yogurtstan_demo_control.png').read_bytes())
    layers = Layers(**{k:read_png((root/('yogurtstan_'+k+'.png')).read_bytes(),base.shape[:2])
                       for k in ('terrain','cities','roads')})
    return World.create(base, START.isoformat(), layers)


class Campaign:
    def __init__(self, world):
        self.world = world
        self.home = world.homeland
        registry = catalog()
        if registry is None or registry.size != (self.home.shape[1],self.home.shape[0]):
            raise ValueError('Для канонной кампании нужен совпадающий по размеру settlements.json')
        self.places = {p['name']:p for p in registry.places}
        if len(self.places) < 100:
            raise ValueError('Неполный список населенных пунктов')
        # Shared border, excluding coastlines and other countries.
        kefir = (self.home == 1).astype(np.uint8)
        yg = self.home == 2
        frontier = cv2.dilate(kefir,np.ones((3,3),np.uint8)).astype(bool)&yg
        yy,xx=np.nonzero(frontier)
        if not len(xx): raise ValueError('На карте нет сухопутной границы')
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack((xx,yy)))
        self.paths=[]
        for name,arrival,retreat,width,falloff,cap in OPERATIONS:
            place=self.places.get(name)
            if not place:
                # Nonexistent labels cannot create fictional town events.
                continue
            x,y=round(place['x']),round(place['y'])
            if self.home[y,x] != 2:
                continue
            _,idx=tree.query([x,y])
            bx,by=int(xx[idx]),int(yy[idx])
            # Irregular yet continuous land corridor with asymmetric bends.
            dx,dy=x-bx,y-by
            seed=sum(ord(c) for c in name)
            curl=math.sin(seed)*.036
            points=np.array([(bx,by),
                             (round(bx+dx*.28-dy*curl),round(by+dy*.28+dx*curl)),
                             (round(bx+dx*.60+dy*curl),round(by+dy*.60-dx*curl)),
                             (x,y)],np.int32)
            y0,y1,x0,x1=bounds(self.home.shape,points,width+14)
            points[:,0]-=x0
            points[:,1]-=y0
            if name == 'Новомир':
                self.novomir_border = (bx,by)
            self.paths.append((name,date.fromisoformat(arrival),date.fromisoformat(retreat),
                               width,falloff,cap,(slice(y0,y1),slice(x0,x1)),points,
                               path_lanes(points,width,name)))
        if len(self.paths)<8: raise ValueError('Слишком мало канонных операций сопоставлено с картой')
        self.skirmish_events = skirmishes(self.paths)
        self.siege_centres = siege_centres(self.home,self.places)
        self.novomir_zone = (novomir_hold(self.home,self.places,[self.novomir_border])
                             if hasattr(self,'novomir_border') else None)
        self.previous=None
        self.recent=[]
        self.current_captures=[]

    def at(self, when):
        """Compute one deterministic date. Calls may be non-sequential or repeated."""
        c=self.world.control
        c[:]=self.home
        active_battles=np.zeros_like(c,dtype=bool)
        # Each operation owns three independently moving lanes, with staggered
        # breakthroughs, pauses and withdrawals. Nothing is a mirrored ping-pong.
        for name,arrival,retreat,width,falloff,cap,sl,points,lanes in self.paths:
            paint_corridor(c,self.home,sl,lanes,when,arrival,retreat,
                           falloff,cap,name,width)
        # Both sides can make small, separately timed local moves. Reachability
        # is checked against friendly-held pixels, never invented landings.
        apply_skirmishes(c,self.home,when,self.skirmish_events,active_battles)
        # A drawn horseshoe is not an encirclement. The April 2059 rings here
        # form closed, thick polygons while the surrounded cores hold out.
        apply_sieges(c,self.home,when,self.siege_centres,active_battles)
        # The industrial positions near Novomir stay Kefir after the ceasefire.
        # The city centre remains Yogurtstan's; no Kefir homeland is ceded.
        if self.novomir_zone is not None and when>=date(2060,6,1):
            sl,mask=self.novomir_zone
            sector=c[sl]
            sector[mask]=1
        c[self.home==1]=1
        if when>=END:
            c[:]=self.home
            if self.novomir_zone is not None:
                sl,mask=self.novomir_zone
                c[sl][mask]=1
        self.world.elapsed=(when-START).days
        if when>=END:
            self.world.disputed[:]=False
        else:
            self.world.disputed[:]=disputed_front(c,self.home,active_battles)
        self.world.fresh_capture=(np.zeros_like(c,dtype=bool) if self.previous is None
                                  else (c!=self.previous)&(self.home==2))
        self.current_captures=[]
        if self.previous is not None:
            for place in self.places.values():
                x,y=round(place['x']),round(place['y'])
                if not (0<=y<c.shape[0] and 0<=x<c.shape[1]) or self.home[y,x]!=2:
                    continue
                before,after=int(self.previous[y,x]),int(c[y,x])
                if before in (1,2) and after in (1,2) and before!=after:
                    event=dict(name=place['name'],side=after,x=x,y=y,
                               day=self.world.elapsed,label=when.strftime('%d.%m.%Y'))
                    self.current_captures.append(event)
                    self.recent.append(event)
        self.recent=self.recent[-4:]
        self.previous=c.copy()
        return dict(label=when.strftime('%d.%m.%Y'),days=self.world.elapsed,
                    source_size=(c.shape[1],c.shape[0]),recent=list(self.recent),
                    captures=list(self.current_captures),stats=self.world.stats())


def dates():
    stamps={START,END}
    for _,arrival,retreat,*_ in OPERATIONS:
        stamps.add(date.fromisoformat(arrival))
        stamps.add(date.fromisoformat(retreat))
    for _,start,days,_ in RAIDS:
        stamps.add(date.fromisoformat(start))
        stamps.add(min(END,date.fromisoformat(start)+timedelta(days=days//2)))
    now=START
    while now<END:
        now=min(END,now+timedelta(days=2 if (now-START).days<60 else 6))
        stamps.add(now)
    return sorted(d for d in stamps if START<=d<=END)


def export_lore(store,chat_id,path,fps=24,width=1280,progress=None,map_fps=None):
    """Render full UI at fps; terrain/control updates at at most map_fps.

    In the cinematic 4K mode the UI and fading capture labels run at 30 FPS,
    while the underlying political map gets a new state at exactly 10 tick
    opportunities per second (every third output frame). No interpolation
    changes the national borders or historical timeline.
    """
    if not 10<=fps<=60 or not 640<=width<=4096:
        raise ValueError('FPS: 10..60, WIDTH: 640..4096')
    if map_fps is not None and (not isinstance(map_fps,int) or map_fps<1 or fps%map_fps):
        raise ValueError('FPS итогового видео должен делиться на FPS карты без остатка')
    width=width//2*2
    height=round(width*9/16)//2*2
    opts=validate(get_settings(store,chat_id))
    world=load_world()
    campaign=Campaign(world)
    frames_per_scene=max(2,round(fps*.20))
    ticks_between_maps=fps//map_fps if map_fps else frames_per_scene
    moments=dates()
    total=len(moments)*frames_per_scene
    ffmpeg=[ffmpeg_executable(),'-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
            '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
            '-threads','2','-preset','veryfast','-crf',str(opts['crf']),
            '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    proc=None
    written=0
    map_updates=0
    try:
        proc=subprocess.Popen(ffmpeg,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        event_queue=[]
        shot=None
        for scene,dt in enumerate(moments):
            next_date=moments[min(scene+1,len(moments)-1)]
            for k in range(frames_per_scene):
                if shot is None or written%ticks_between_maps==0:
                    # Use calendar interpolation for the sparse cinematic
                    # keyframes, but never invent sub-pixel diplomatic borders.
                    days=(next_date-dt).days
                    when=dt+timedelta(days=min(days,round(days*k/frames_per_scene)))
                    info=campaign.at(when)
                    event_queue[:]=[e for e in event_queue if written/fps-e['time']<2.2]
                    event_queue.extend(dict(e,time=written/fps) for e in info['captures'][-6:])
                    shot=Shot(world,info,opts,width)
                    map_updates+=1
                phase=(written%ticks_between_maps)/max(1,ticks_between_maps-1)
                frame=shot.frame(written/fps,event_queue,phase,width,height)
                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    raise ValueError('FFmpeg: '+proc.stderr.read().decode(errors='replace')[-1000:])
                written+=1
            if progress:progress(written,total,'Канонная карта: '+dt.strftime('%d.%m.%Y'))
        proc.stdin.close()
        err=proc.stderr.read().decode(errors='replace')
        if proc.wait():raise ValueError('FFmpeg: '+err[-1200:])
        return dict(frames=written,width=width,height=height,days=(END-START).days,
                    duration=written/fps,fps=fps,map_fps=map_fps or fps,
                    map_updates=map_updates,ui_fps=fps)
    except BaseException:
        if proc is not None and proc.poll() is None:proc.kill();proc.wait()
        Path(path).unlink(missing_ok=True)
        raise

def preview_lore(store,chat_id,progress=None):
    if progress:progress(0,3,'Загрузка канонной карты')
    world=load_world()
    campaign=Campaign(world)
    info=campaign.at(date(2058,12,31))
    if progress:progress(1,3,'Нанесение участков фронта')
    options=validate(get_settings(store,chat_id))
    shot=Shot(world,info,options,1920)
    image=shot.frame(0,[],1,1920,1080)
    if progress:progress(3,3,'Снимок готов')
    return image
