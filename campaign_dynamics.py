"""Lore-campaign film: non-cyclic territorial evolution and real closures.

Cinematic interpolation only; local actions are not canonically dated captures.
"""
from datetime import date, timedelta
import math
import zlib

import cv2
import numpy as np
from scipy.spatial import cKDTree

K3 = np.ones((3, 3), np.uint8)


def seed_of(text):
    return zlib.crc32(text.encode('utf-8')) & 0xffffffff


def bound(shape, x, y, radius):
    return (slice(max(0, y-radius), min(shape[0], y+radius+1)),
            slice(max(0, x-radius), min(shape[1], x+radius+1)))


def path_lanes(points, width, name):
    """Three offset columns keep local fronts from moving in lockstep."""
    p = np.asarray(points, np.float32)
    parts = [p[i][None, :] + (p[i+1]-p[i])[None, :] *
             np.linspace(0, 1, 24, endpoint=False)[:,None]
             for i in range(len(p)-1)]
    center = np.concatenate(parts + [p[-1:]], axis=0)
    tangent = np.gradient(center, axis=0)
    unit = tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), .0001)
    perp = np.stack((-unit[:,1], unit[:,0]), axis=1)
    phase = (seed_of(name)%509)/83
    t = np.linspace(0, 1, len(center))
    lanes=[]
    for lane in range(3):
        offset = (lane-1)*.31*width
        wobble = (.075*width*np.sin((4.5+lane)*math.pi*t+phase+lane*1.6)
                  +.038*width*np.sin((11+lane)*math.pi*t+phase*.7))
        deviation = np.minimum(1, t*9) * (offset+wobble)
        lanes.append(np.rint(center + perp*deviation[:,None]).astype(np.int32))
    return lanes


def _ramp(u, seed, lane):
    jitter = ((seed >> (lane*5)) & 15)/15
    return float(np.interp(np.clip(u,0,1),
        [0,.12,.29,.46,.65,.83,1],
        [0,.13+.05*jitter,.21+.1*jitter,.48+.05*jitter,
         .56+.08*jitter,.83+.03*jitter,1]))


def _retreat(u, seed, lane):
    jitter = ((seed >> (lane*7)) & 31)/31
    return float(np.interp(np.clip(u,0,1),
        [0,.13,.36,.52,.74,1],
        [0,.07+.11*jitter,.32+.07*jitter,.37+.11*jitter,
         .79+.07*jitter,1]))


def lane_progress(when, arrival, retreat, falloff, cap, name, lane):
    seed=seed_of(name)
    start = max(date(2057,7,1), arrival-timedelta(days=14)) if arrival.year==2057 else max(date(2058,7,1), arrival-timedelta(days=42))
    end_of_push=arrival+timedelta(days=(-1,0,2)[lane])
    if when<=start:return 0.
    p=_ramp((when-start).days/max(1,(end_of_push-start).days),seed,lane)*cap
    if when>retreat-timedelta(days=falloff):
        v=(when-(retreat-timedelta(days=falloff))).days/max(1,falloff)
        lag=(.05,.0,.10)[lane]
        p*=1-_retreat((v-lag)/(1-lag),seed,lane)
    return float(np.clip(p,0,cap))


def paint_corridor(control, home, sl, lanes, when, arrival, retreat, falloff, cap, name, width):
    mask=np.zeros(control[sl].shape,np.uint8)
    for lane,points in enumerate(lanes):
        progress=lane_progress(when,arrival,retreat,falloff,cap,name,lane)
        if progress<=0:continue
        stop=max(2,min(len(points),round((len(points)-1)*progress)+1))
        thickness=max(4,round(width*(.40,.47,.38)[lane]))
        cv2.polylines(mask,[points[:stop]],False,1,thickness,cv2.LINE_8)
        if progress>=.999 and cap>=1:
            cv2.circle(mask,tuple(points[-1]),max(3,thickness//3),1,-1)
    view=control[sl]
    view[(mask>0)&(home[sl]==2)]=1


def skirmishes(paths):
    """One-off sector actions, staggered in time and place, not repeated cycles."""
    events=[]
    for name,arrival,retreat,width,falloff,cap,sl,points,lanes in paths:
        seed=seed_of(name)
        duration=(retreat-arrival).days
        if duration<=45:continue
        count=max(2,min(13,duration//57))
        for i in range(count):
            salt=(seed >> ((i%4)*5))&31
            k=(i+.35+(salt%9)/15)/(count+.7)
            start=arrival+timedelta(days=round(k*max(30,duration-55)))
            if start>=retreat:continue
            span=15+(seed+i*29)%41
            lane=0 if i%2==0 else 2
            fraction=min(.93,max(.22,.31+(seed+i*37)%60/100))
            point=lanes[lane][min(len(lanes[lane])-1,
                                   round((len(lanes[lane])-1)*fraction))]
            side=2 if (i+(seed%5))%3!=0 else 1
            radius=max(9,round(width*(.33+.14*((i*7+seed)%8)/7)))
            events.append((start,span,side,sl,int(point[0]),int(point[1]),radius,seed^i))
    return events


def apply_skirmishes(c,home,when,events,hot):
    for start,span,side,operation_slice,x,y,radius,seed in events:
        d=(when-start).days
        if d<0 or d>=span+15:continue
        strength=min(1.,d/max(4,span*.38)) if d<span else .65*(1-(d-span)/15)
        if strength<=0:continue
        gx=x+operation_slice[1].start
        gy=y+operation_slice[0].start
        radius=max(3,round(radius*strength))
        sl=bound(c.shape,gx,gy,radius+5)
        yy,xx=np.ogrid[sl[0].start:sl[0].stop,sl[1].start:sl[1].stop]
        theta=(seed%157)*math.pi/180
        rx=(xx-gx)*math.cos(theta)+(yy-gy)*math.sin(theta)
        ry=-(xx-gx)*math.sin(theta)+(yy-gy)*math.cos(theta)
        ellipse=(rx*rx/max(1,radius**2)+ry*ry/max(1,(radius*.7)**2))<=1
        region=c[sl]
        # The only legal advances touch friendly-held pixels: no spawn in rear.
        friendly=(region==side).astype(np.uint8)
        steps=max(1,round(1+strength*min(9,radius/2)))
        reach=cv2.dilate(friendly,K3,iterations=steps).astype(bool)
        take=ellipse&reach&(home[sl]==2)
        take &= region==(3-side)
        region[take]=side
        hot[sl] |= take


# The April 2059 Yogurt pocket must really close before September relief.
SIEGES=(
    ('Йогуртск',date(2059,1,4),date(2059,4,4),date(2059,9,30),28),
    ('Дестар',date(2059,1,23),date(2059,4,4),date(2059,9,30),24),
)


def siege_centres(home,places):
    inside=(home==2).astype(np.uint8)
    distance=cv2.distanceTransform(inside,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    centres=[]
    for name,start,closed,broken,r in SIEGES:
        place=places.get(name)
        if place is None:continue
        x,y=round(place['x']),round(place['y'])
        if not (0<=x<home.shape[1] and 0<=y<home.shape[0]) or home[y,x]!=2:continue
        sl=bound(home.shape,x,y,45)
        by,bx=np.unravel_index(np.argmax(distance[sl]),distance[sl].shape)
        cx,cy=sl[1].start+int(bx),sl[0].start+int(by)
        if distance[y,x]>=r+6:cx,cy=x,y
        radius=min(r,max(9,int(distance[cy,cx])-4))
        if radius<12:continue
        centres.append((name,start,closed,broken,cx,cy,radius))
    return centres


def apply_sieges(c,home,when,centres,hot):
    for name,start,closed,broken,x,y,r in centres:
        if not start<=when<broken:continue
        sl=bound(c.shape,x,y,r+13)
        lx=x-sl[1].start;ly=y-sl[0].start
        local=c[sl];land=home[sl]==2
        fraction=min(1.,(when-start).days/max(1,(closed-start).days))
        ring=np.zeros(local.shape,np.uint8)
        thickness=max(5,round(r*.30))
        cv2.ellipse(ring,(lx,ly),(r,r),0,
                    180-round(180*fraction),180+round(180*fraction),
                    1,thickness,cv2.LINE_8)
        core=np.zeros(local.shape,np.uint8)
        cv2.circle(core,(lx,ly),max(3,r-thickness//2-3),1,-1)
        # Link encircling units to an already occupied sector, not open water.
        existing=(local==1)&land
        if existing.any() and ring.any():
            sy,sx=np.where(existing);ry,rx=np.where(ring>0)
            distances,indices=cKDTree(np.column_stack((sx,sy))).query(np.column_stack((rx,ry)))
            i=int(np.argmin(distances))
            a=(int(sx[indices[i]]),int(sy[indices[i]]))
            b=(int(rx[i]),int(ry[i]))
            cv2.line(ring,a,b,1,max(3,thickness//2),cv2.LINE_8)
        local[(ring>0)&land]=1
        local[(core>0)&land]=2
        hot[sl] |= ((ring>0)|(core>0))&land


def novomir_hold(home,places,path):
    """A retained industrial-sector foothold, not occupation of Novomir centre."""
    place=places.get('Новомир')
    if place is None:return None
    x,y=round(place['x']),round(place['y'])
    if not(0<=x<home.shape[1] and 0<=y<home.shape[0]) or home[y,x]!=2:return None
    bx,by=map(int,path[0])
    vec=np.array([bx-x,by-y],np.float32)
    vec/=max(.001,float(np.linalg.norm(vec)))
    middle=(round(x+vec[0]*19),round(y+vec[1]*19))
    sl=bound(home.shape,middle[0],middle[1],35)
    mask=np.zeros((sl[0].stop-sl[0].start,sl[1].stop-sl[1].start),np.uint8)
    angle=math.degrees(math.atan2(vec[1],vec[0]))
    cx,cy=middle[0]-sl[1].start,middle[1]-sl[0].start
    cv2.ellipse(mask,(cx,cy),(27,15),angle,0,360,1,-1)
    cv2.ellipse(mask,(cx+round(vec[1]*7),cy-round(vec[0]*7)),
                (15,9),angle+22,0,360,1,-1)
    cv2.circle(mask,(x-sl[1].start,y-sl[0].start),11,0,-1)
    return sl,((mask>0)&(home[sl]==2))


def disputed_front(c,home,hot):
    """Ordinary 1-2 pixel grey frontline; localized 7-10 pixel battle sectors."""
    k=(c==1).astype(np.uint8)
    y=(c==2).astype(np.uint8)
    contact=(k.astype(bool)&cv2.dilate(y,K3).astype(bool)) | (y.astype(bool)&cv2.dilate(k,K3).astype(bool))
    normal=cv2.dilate(contact.astype(np.uint8),K3).astype(bool)
    wide=contact&cv2.dilate(hot.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
    if wide.any():
        normal|=cv2.dilate(wide.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    return normal&(home==2)
