"""Deterministic intent-driven daily simulation. No autonomous strategic orders."""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from geodesic import propagate, skeleton_distance
import config

K3=np.ones((3,3),np.uint8)

def correlated_noise(shape, seed):
    rng=np.random.default_rng(seed); out=np.zeros(shape,np.float32)
    for scale,weight in [(70,.5),(23,.32),(7,.18)]:
        coarse=rng.uniform(-1,1,(max(2,shape[0]//scale+2),max(2,shape[1]//scale+2))).astype(np.float32)
        out += weight*cv2.resize(coarse,(shape[1],shape[0]),interpolation=cv2.INTER_CUBIC)
    return out

@dataclass
class FrontPlan:
    side:int
    sl:tuple
    arrival:np.ndarray
    tempo:np.ndarray
    supply:np.ndarray


def prepare(world, masks, days, width, seed):
    if not 1<=days<=config.MAX_DAYS or not 2<=width<=config.MAX_WIDTH:
        raise ValueError(f'DAYS: 1..{config.MAX_DAYS}, WIDTH: 2..{config.MAX_WIDTH}')
    plans=[]
    for side,mask in sorted(masks.items()):
        if side not in (1,2) or mask.shape!=world.control.shape: raise ValueError('Некорректная маска приказа')
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        own=world.control==side
        own_distance=cv2.distanceTransform((~own).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        for label in range(1,n):
            x,y,w,h,area=stats[label]
            if area<5: raise ValueError('Линия приказа слишком короткая')
            pad=int(width*1.6+12)
            y0,y1=max(0,y-pad),min(mask.shape[0],y+h+pad)
            x0,x1=max(0,x-pad),min(mask.shape[1],x+w+pad)
            sl=(slice(y0,y1),slice(x0,x1))
            sk=skeletonize(labels[sl]==label)
            count=cv2.filter2D(sk.astype(np.uint8),cv2.CV_16S,K3)-sk
            ends=np.argwhere(sk&(count==1))
            if len(ends)<2: raise ValueError('Замкнутая линия: нужны начало и конец')
            scores=own_distance[sl][ends[:,0],ends[:,1]]
            origin=ends[int(np.argmin(scores))]
            if float(scores.min())>max(8,min(width*.15,20)):
                raise ValueError('Начало линии не связано со своей территорией')
            s=skeleton_distance(sk,int(origin[0]),int(origin[1]))
            length=float(s[sk].max())
            if not np.isfinite(length) or length<8: raise ValueError('Линия слишком короткая или разорвана')
            # Skeleton distance, not x/y direction: curves and S shapes retain intent.
            lateral,indices=distance_transform_edt(~sk,return_indices=True)
            along=s[tuple(indices)].astype(np.float32)
            lateral=lateral.astype(np.float32)
            noise=correlated_noise(sk.shape,seed+side*1009+label*9176)
            envelope=(width*.52*(1+.48*noise)).astype(np.float32)
            terrain=world.territory[sl]
            allowed=terrain&(lateral<width*1.55)&(along<length+1)
            # No endpoint teleportation: every seed is near the proximal line end.
            seeds=own[sl]&allowed&(along<max(12,min(length*.12,width*.65)))&(lateral<width*.68)
            if not seeds.any(): raise ValueError('Линия не имеет снабжаемого начала на своей земле')
            cost=world.layers.movement_cost(sl)*(1+.65*noise)
            cost *= 1+.45*(along/max(length,1))**1.5  # deep advances cost supply
            cost *= 1+np.minimum(world.battle_age[sl],20)*.014
            speed=max(.4,length/days*.86)
            arrival=propagate(cost.astype(np.float32),allowed,seeds,along,lateral,envelope,length*3.4)/speed
            rng=np.random.default_rng(seed+side*997)
            daily=[]; phase=float(rng.uniform(0,6.28)); value=1.0
            for day in range(days):
                target=1+.37*np.sin(day*.58+phase)+.20*np.sin(day*.19+phase*2)
                value=.64*value+.36*target
                daily.append(value)
            tempo=np.cumsum(daily,dtype=np.float32)
            supply=(1+.25*np.exp(-world.layers.road_distance[sl]/18)-.15*along/max(length,1)).astype(np.float32)
            plans.append(FrontPlan(side,sl,arrival,tempo,supply))
    return plans


def _reachable(local_mask, old, side):
    # Claims must retain a connected route from current territory; lost supply cuts advances.
    n,labels=cv2.connectedComponents(local_mask.astype(np.uint8),8)
    roots=np.unique(labels[(old==side)&local_mask]); roots=roots[roots!=0]
    return np.isin(labels,roots) if len(roots) else np.zeros_like(local_mask)


def resolve_pockets(world):
    c=world.control; previous=world.encirclement_age
    new_age=np.zeros_like(previous)
    for side in (1,2):
        n,labels,stats,_=cv2.connectedComponentsWithStats((c==side).astype(np.uint8),8)
        if n<=2: continue
        main=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        for lab in range(1,n):
            if lab==main: continue
            x,y,w,h,area=stats[lab]
            y0,y1=max(0,y-1),min(c.shape[0],y+h+1)
            x0,x1=max(0,x-1),min(c.shape[1],x+w+1)
            sl=(slice(y0,y1),slice(x0,x1)); pocket=labels[sl]==lab
            # Coast, water and map edge do not constitute a closed enemy ring.
            if x==0 or y==0 or x+w==c.shape[1] or y+h==c.shape[0]: continue
            ring=cv2.dilate(pocket.astype(np.uint8),K3).astype(bool)&~pocket
            if not ring.any() or not np.all(c[sl][ring]==3-side): continue
            city=world.layers.city_mask[sl]&pocket
            if not city.any():
                view=c[sl]; view[pocket]=3-side
            else:
                age=min(65534,int(previous[sl][pocket].min())+1)
                new_age[sl][pocket]=age
                # More city pixels and larger pockets buy a bounded additional hold.
                hold=world.layers.settings['CITY_HOLD_DAYS']+min(12,int(np.sqrt(city.sum())/3)+int(np.log1p(area)/3))
                if age>=hold:
                    c[sl][pocket]=3-side; new_age[sl][pocket]=0
    world.encirclement_age=new_age


def update_contested(world, collision=None, changed=None):
    c=world.control; a=(c==1); b=(c==2)
    fa=a&cv2.dilate(b.astype(np.uint8),K3).astype(bool)
    fb=b&cv2.dilate(a.astype(np.uint8),K3).astype(bool)
    front=fa|fb
    # City defence and stalled road junctions accumulate only near the contact line.
    hot=front&((world.layers.city_influence>.35)|world.layers.road_nodes)
    if collision is not None: hot|=collision
    if changed is not None:
        moving=cv2.dilate(changed.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        hot &= ~moving | (collision if collision is not None else False)
    age=world.battle_age
    age[:]=np.where(hot,np.minimum(age.astype(np.uint32)+1,65534),np.maximum(age.astype(np.int32)-2,0)).astype(np.uint16)
    grey=fa.copy()  # exactly one native pixel on ordinary straight front
    grey |= fb&(age>=2)
    # Rare local broad areas, driven by correlated battle history, never whole front.
    for minimum,radius in [(4,1),(8,2),(14,3),(22,4)]:
        source=front&(age>=minimum)
        if source.any():
            spread=cv2.dilate(source.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)).astype(bool)
            grey |= spread&world.territory
    if collision is not None: grey |= collision&world.territory
    world.disputed=grey


def simulate(world, masks, days, width, seed, on_day=None):
    plans=prepare(world,masks,days,width,seed) if masks else []
    # Only moving ROIs need proposal arrays. All plans in one shared daily clock.
    if plans:
        y0=min(p.sl[0].start for p in plans); y1=max(p.sl[0].stop for p in plans)
        x0=min(p.sl[1].start for p in plans); x1=max(p.sl[1].stop for p in plans)
        bounds=(slice(y0,y1),slice(x0,x1))
    for day in range(days):
        collision=np.zeros_like(world.control,bool)
        changed=np.zeros_like(world.control,bool)
        if plans:
            old=world.control[bounds].copy(); shape=old.shape
            arrival={s:np.full(shape,np.inf,np.float32) for s in (1,2)}
            supply={s:np.zeros(shape,np.float32) for s in (1,2)}
            for p in plans:
                loc=(slice(p.sl[0].start-y0,p.sl[0].stop-y0),slice(p.sl[1].start-x0,p.sl[1].stop-x0))
                threshold=p.tempo[day]
                claim=np.isfinite(p.arrival)&(p.arrival<=threshold)
                claim=_reachable(claim,world.control[p.sl],p.side)
                # arrival is expressed in shared nominal days, tempo moves its clock.
                time=p.arrival*(day+1)/max(threshold,.01)
                view=arrival[p.side][loc]
                better=claim&(time<view)
                view[better]=time[better]; supply[p.side][loc][better]=p.supply[better]
            ak,ay=arrival[1],arrival[2]
            k,y=np.isfinite(ak),np.isfinite(ay)
            both=k&y
            delta=np.zeros(shape,np.float32); np.subtract(ak,ay,out=delta,where=both)
            close=both&(np.abs(delta)<1.15)
            age=world.battle_age[bounds]
            # Prolonged meetings resolve using arrival, supply and incumbent defence.
            score=delta-(supply[1]-supply[2])*.65+np.where(old==1,-.12,.12)
            resolved=close&(age>=7)
            takek=k&(~y|((delta<-.0)&~close)|(resolved&(score<0)))
            takey=y&(~k|((delta>=0)&~close)|(resolved&(score>=0)))
            view=world.control[bounds]
            view[takek]=1; view[takey]=2
            collision[bounds]=close&~resolved
            changed[bounds]=view!=old
        resolve_pockets(world)
        update_contested(world,collision,changed)
        world.elapsed+=1
        if on_day: on_day(world)
    return world
