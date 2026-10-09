"""Reproducible 60-day native-map smoke and deliverable generation. Run python -B smoke_real.py."""
import sys
sys.dont_write_bytecode=True
import os
os.environ.setdefault('NUMBA_NUM_THREADS','2')
import time
import json
import hashlib
import subprocess
import numpy as np
import cv2
from pathlib import Path
import config
from layers import read_png,Layers,png_bytes
from state import World
from storage import Store,pack_masks
from renderer import render
from exporter import export_video
from engine import simulate,update_contested

CHAT=-2057

def paths(shape,lines):
    mask=np.zeros(shape,np.uint8)
    for points in lines: cv2.polylines(mask,[np.array(points,np.int32)],False,1,5)
    return mask.astype(bool)

def sha(a): return hashlib.sha256(a.tobytes()).hexdigest()

def main():
    t=time.perf_counter(); root=config.ROOT
    layers=Layers(**{k:read_png((root/f'yogurtstan_{k}.png').read_bytes()) for k in ['terrain','cities','roads']})
    w=World.create(read_png((root/'yogurtstan_demo_control.png').read_bytes()),'2057-06-28',layers)
    assert w.control.shape==(3072,4096)
    store=Store(root/'example_state.sqlite3'); store.create(CHAT,w)
    baseline=w.control.copy(); home=sha(w.homeland)
    report={'demo_preparation':json.loads((root/'demo_preparation.json').read_text()),'native_size':[4096,3072],'start_date':w.start_date,'engine':config.ENGINE_VERSION,'operations':[]}
    points=[[(2609,1040),(2600,1120),(2560,1210),(2630,1350),(2670,1480),(2670,1600)],[(1770,1050),(1830,1260),(1740,1440),(1630,1640)],[(3340,900),(3330,1020),(3370,1160),(3410,1280)]]
    mask=paths(w.control.shape,points)
    print('Native first offensive…',flush=True)
    w=store.commit(CHAT,'turn',{'days':20,'width':210,'seed':20570628},pack_masks({1:mask}))
    (root/'example_day20.png').write_bytes(png_bytes(render(w,1920,date_label=True)))
    report['operations'].append(w.stats()); print(w.stats(),flush=True)
    # Counterstroke starts in surviving enemy land near the central advance.
    yy,xx=np.where((w.control==2)&(np.indices(w.control.shape)[0]>1300)&(np.indices(w.control.shape)[0]<1550)&(np.indices(w.control.shape)[1]>2420)&(np.indices(w.control.shape)[1]<2530))
    if len(xx)==0: raise AssertionError('Counterstroke source not found')
    y,x=int(yy[len(yy)//2]),int(xx[len(xx)//2])
    yogurt=paths(w.control.shape,[[(x,y),(x+90,y-100),(2580,1160),(2640,1030)]])
    # Another fresh attack from actual captured territory, near the road tongue.
    kpoints=[[(2609,1040),(2660,1230),(2720,1410),(2800,1570),(2880,1730)]]
    print('Native simultaneous second offensive…',flush=True)
    w=store.commit(CHAT,'turn',{'days':20,'width':200,'seed':20570718},pack_masks({1:paths(w.control.shape,kpoints),2:yogurt}))
    (root/'example_day40.png').write_bytes(png_bytes(render(w,1920,date_label=True)))
    report['operations'].append(w.stats()); print(w.stats(),flush=True)
    print('Native third operation / west front…',flush=True)
    k=paths(w.control.shape,[[(1770,1050),(1740,1380),(1610,1550),(1450,1760),(1310,1850)]])
    y=paths(w.control.shape,[[(3500,1400),(3390,1210),(3340,1020),(3300,910)]])
    w=store.commit(CHAT,'turn',{'days':20,'width':190,'seed':20570807},pack_masks({1:k,2:y}))
    report['operations'].append(w.stats()); print(w.stats(),flush=True)
    assert w.elapsed==60 and w.date.isoformat()=='2057-08-27'
    assert sha(w.homeland)==home
    assert np.all(w.control[~w.territory]==0)
    report['captured_px']=int(((w.control==1)&(baseline==2)).sum())
    assert report['captured_px']>1000
    report['final_control_sha256']=sha(w.control)
    (root/'example_preview.png').write_bytes(png_bytes(render(w,3840,date_label=True)))
    # Native-scale detail for manual review: actual contact, roads, terrain and cities.
    crop=render(w)[1050:1850,2350:3150]
    (root/'example_detail.png').write_bytes(png_bytes(crop))
    print('Doctor replay…',flush=True)
    same,replayed=store.doctor(CHAT); assert same
    report['doctor_pixel_perfect']=same
    # Native map fixture: intentionally isolated real city, then cityless adjacent pocket.
    # This is a separate validation state; never alters the movie/log.
    print('Native encirclement fixtures…',flush=True)
    yy,xx=np.where(layers.city_mask&(baseline==2))
    found=None
    for cy,cx in zip(yy[::20],xx[::20]):
        if cy<1200 or cx<2100 or cy>2972 or cx>3996: continue
        sl=(slice(cy-45,cy+46),slice(cx-45,cx+46))
        if np.all(w.territory[sl]) and np.all(w.control[sl]==2): found=(int(cx),int(cy)); break
    assert found is not None
    cx,cy=found
    pocket_world=World.create(w.background.copy(),'2057-06-28',layers)
    pocket_world.control[pocket_world.territory]=1
    pocket_world.control[:,:1500]=w.initial_control[:,:1500]
    pocket_world.control[cy-45:cy+46,cx-45:cx+46]=1
    pocket_world.control[cy-14:cy+15,cx-14:cx+15]=2
    simulate(pocket_world,{},1,20,0)
    assert pocket_world.control[cy,cx]==2
    simulate(pocket_world,{},40,20,0)
    assert pocket_world.control[cy,cx]==1
    report['native_city_hold_then_fall']=True
    # Cityless enclosed pocket deliberately located on solid real-map territory.
    cityless=w.control.copy(); yy,xx=np.where((baseline==2)&(layers.city_influence<.005))
    found=False
    for cy,cx in zip(yy[::1000],xx[::1000]):
        if cy<50 or cx<2100 or cy>3022 or cx>4046: continue
        sl=(slice(cy-12,cy+13),slice(cx-12,cx+13))
        if np.all(w.territory[sl]) and not layers.city_mask[sl].any():
            pocket_world.control[cy-16:cy+17,cx-16:cx+17]=1
            pocket_world.control[sl]=2
            simulate(pocket_world,{},1,20,0)
            assert np.all(pocket_world.control[sl]==1); found=True; break
    assert found; report['native_cityless_pocket_falls']=True
    del pocket_world,cityless,baseline
    # Separate native-map stress fixture: sustained opposing claims along real contact.
    # This does not alter the operation log or pretend to be a day from the movie.
    print('Native prolonged meeting / rare broad grey fixture…',flush=True)
    stress=World.create(w.background.copy(),'2057-06-28',layers)
    stress.control=w.control.copy()
    one=(stress.control==1);two=(stress.control==2);kernel=np.ones((3,3),np.uint8)
    contact=(one&cv2.dilate(two.astype(np.uint8),kernel).astype(bool))|(two&cv2.dilate(one.astype(np.uint8),kernel).astype(bool))
    for _ in range(32):update_contested(stress,contact)
    distance=cv2.distanceTransform((~contact).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    broad=stress.disputed&(distance>2.1)
    assert broad.any()
    assert stress.meeting_age.max()==32
    yy,xx=np.where(broad);cy,cx=int(yy[len(yy)//2]),int(xx[len(xx)//2])
    crop=render(stress)[max(0,cy-160):cy+160,max(0,cx-180):cx+180]
    (root/'example_battle_detail.png').write_bytes(png_bytes(crop))
    report['native_prolonged_meeting_rare_grey']=True
    report['front_model']=w.layers.settings.get('FRONT_MODEL',1)
    report['rare_battle_pixels_beyond_2px']=int(broad.sum())
    del stress,one,two,contact,distance,broad
    for mode,name in [(False,'example.mp4'),(True,'example_curved.mp4')]:
        print('Export',name,flush=True)
        info=export_video(store,CHAT,root/name,10,1920,mode)
        assert info['frames']==61
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(root/name)]))['streams'][0]
        assert probe['codec_name']=='h264' and probe['pix_fmt']=='yuv420p'
        assert int(probe['nb_read_frames'])==61
        subprocess.run(['ffmpeg','-v','error','-i',str(root/name),'-f','null','-'],check=True)
        report[name]=info
    report['duration_seconds']=round(time.perf_counter()-t,2)
    (root/'smoke_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SMOKE PASS',report['duration_seconds'],'seconds',flush=True)

if __name__=='__main__': main()
