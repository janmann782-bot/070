"""Native-resolution bounded 8-neighbour Dijkstra; compiled, no disk JIT cache."""
import heapq
import numpy as np
from numba import njit

@njit(cache=False)
def propagate(cost, allowed, seeds, along, lateral, width, max_cost):
    h,w=allowed.shape
    dist=np.full((h,w),np.inf,np.float32)
    heap=[(0.0,0,0)]; heap.pop()
    for y in range(h):
        for x in range(w):
            if seeds[y,x] and allowed[y,x]:
                dist[y,x]=0; heapq.heappush(heap,(0.0,y,x))
    while heap:
        d,y,x=heapq.heappop(heap)
        if d>dist[y,x]+1e-5 or d>max_cost: continue
        for dy in range(-1,2):
            for dx in range(-1,2):
                if dx==0 and dy==0: continue
                yy,xx=y+dy,x+dx
                if yy<0 or yy>=h or xx<0 or xx>=w or not allowed[yy,xx]: continue
                if dx!=0 and dy!=0 and (not allowed[y,xx] or not allowed[yy,x]): continue
                ds=along[yy,xx]-along[y,x]
                dl=abs(lateral[yy,xx]-lateral[y,x])
                # No axis assumption; progress runs along the curved skeleton.
                steering=1.0+0.65*max(0.0,-ds)+0.7*dl
                length=1.41421356237 if dx!=0 and dy!=0 else 1.0
                side=0.5*(lateral[y,x]+lateral[yy,xx])/max(width[y,x],2.0)
                nd=d+length*0.5*(cost[y,x]+cost[yy,xx])*steering*(1+side*side*1.8)
                if nd<dist[yy,xx] and nd<=max_cost:
                    dist[yy,xx]=nd; heapq.heappush(heap,(np.float64(dist[yy,xx]),yy,xx))
    return dist

@njit(cache=False)
def skeleton_distance(skeleton, sy, sx):
    h,w=skeleton.shape; d=np.full((h,w),np.inf,np.float32)
    d[sy,sx]=0; heap=[(0.0,sy,sx)]
    while heap:
        v,y,x=heapq.heappop(heap)
        if v>d[y,x]+1e-5: continue
        for dy in range(-1,2):
            for dx in range(-1,2):
                yy,xx=y+dy,x+dx
                if (dx==0 and dy==0) or yy<0 or yy>=h or xx<0 or xx>=w or not skeleton[yy,xx]: continue
                nv=v+(1.41421356237 if dx and dy else 1.0)
                if nv<d[yy,xx]:
                    d[yy,xx]=nv; heapq.heappush(heap,(np.float64(d[yy,xx]),yy,xx))
    return d
