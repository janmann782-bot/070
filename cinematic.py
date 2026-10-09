"""Multiframe broadcast export from an immutable presentation timeline."""
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image
from exporter import ffmpeg_executable
from renderer import render, Curve
from cinematic_ui import Broadcast
from visual_settings import validate, settings as get_settings
from render_timeline import Timeline
from postprocess import process


class Shot:
    def __init__(self, world, info, options, width=1920):
        self.info, self.options = info, options
        self.ui = Broadcast(1920)
        rgb = process(render(world,max(1920,width)), options['postprocess'])
        self.curve = Curve(rgb.shape) if options['curved'] else None
        if self.curve:
            rgb = cv2.remap(rgb,self.curve.mx,self.curve.my,cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,borderValue=(7,11,8))
        self.base = Image.fromarray(self.ui.compose(rgb,info,settings=options))
        x,y,w,h = self.ui.actual_map
        self.base_pixels = np.asarray(self.base)
        self.detail = None
        if width>1920:
            low = np.asarray(Image.fromarray(rgb).resize((w,h),Image.Resampling.LANCZOS))
            self.hud_mask = np.any(self.base_pixels[y:y+h,x:x+w] != low,axis=2)
            scale=width/1920
            self.detail=Image.fromarray(rgb).resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
        self.mask = None
        if options['attack_glow'] and world.fresh_capture is not None and world.fresh_capture.any():
            mask = cv2.resize(world.fresh_capture.astype(np.float32), (rgb.shape[1],rgb.shape[0]), interpolation=cv2.INTER_AREA)
            if self.curve:
                mask = cv2.remap(mask,self.curve.mx,self.curve.my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            self.mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_AREA)[:,:,None]
            self.patch = np.asarray(self.base.crop((x,y,x+w,y+h))).astype(np.float32)

    def frame(self, now, events, phase, width, height):
        im = self.base.copy()
        if self.mask is not None:
            alpha = self.mask*(.10*(1-phase)**2)
            patch = np.clip(self.patch*(1-alpha)+np.array([169,243,143])*alpha,0,255).astype(np.uint8)
            im.paste(Image.fromarray(patch),self.ui.actual_map[:2])
        mapped = events
        if self.curve:
            # Curved capture locators follow exactly the same remap as the map.
            mapped = []
            for event in events:
                e = dict(event)
                h,w = self.curve.mx.shape
                tx,ty = e['x']/self.info['source_size'][0]*w,e['y']/self.info['source_size'][1]*h
                # Fixed-point inverse of Curve's mild bow/perspective. No full
                # distance image for each city on each animation frame.
                dx,dy=tx,ty
                for _ in range(5):
                    v=(dy-h/2)/(h/2)
                    dx=w/2+(tx-w/2)*(1-.035*(1-v))
                    u=(dx-w/2)/(w/2)
                    dy=(ty-.033*h*u*u)/(1+.024*u*u)
                e['x'] = dx/w*self.info['source_size'][0]
                e['y'] = dy/h*self.info['source_size'][1]
                mapped.append(e)
        if self.options['city_glow'] or self.options['notifications']:
            self.ui._events(im,mapped,now,self.options,self.info['source_size'])
        if width !=1920:
            small=np.asarray(im)
            im=im.resize((width,height),Image.Resampling.LANCZOS)
            if self.detail is not None:
                x,y,w,h=self.ui.actual_map;scale=width/1920
                mask=self.hud_mask | np.any(small[y:y+h,x:x+w]!=self.base_pixels[y:y+h,x:x+w],axis=2)
                alpha=Image.fromarray((mask*255).astype(np.uint8)).resize(self.detail.size,Image.Resampling.NEAREST)
                # Keep the high-resolution map; overlay only HUD and animated pixels.
                position=(round(x*scale),round(y*scale))
                region=im.crop((*position,position[0]+self.detail.width,position[1]+self.detail.height))
                detail=self.detail.copy();detail.paste(region,(0,0),alpha)
                im.paste(detail,position)
        return np.asarray(im)


def export_cinematic(store, chat_id, path, fps=30, width=1920, options=None, progress=None, _daily=False, _curve=None):
    if not isinstance(fps,int) or not 1<=fps<=60:
        raise ValueError('FPS: 1..60')
    if not isinstance(width,int) or not (64 if _daily else 320)<=width<=4096:
        raise ValueError('WIDTH: '+('64' if _daily else '320')+'..4096')
    options=validate(options if options is not None else get_settings(store,chat_id))
    if _curve is not None:options['curved']=_curve
    width=width//2*2;height=(round(width*9/16)//2)*2
    timeline=Timeline(store,chat_id)
    if _daily and progress:progress(0,timeline.days+1,'Подготовка истории')
    built=timeline.ensure(None if _daily else progress)
    per_day=1 if _daily else max(1,round(fps*options['seconds_per_day']))
    initial=1 if _daily else fps; outro=0 if _daily else round(2.2*fps)
    total=initial+timeline.days*per_day+outro
    if progress:progress(0,total,'Временная линия готова; подготовка кадров')
    command=[ffmpeg_executable(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2',
             '-preset','medium','-crf',str(options['crf']),'-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    proc=None; count=0; events=[]
    try:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        def write(shot,phase):
            nonlocal count
            now=count/fps
            events[:]=[e for e in events if now-e['time']<2.2]
            rgb=shot.frame(now,events,phase,width,height)
            try:proc.stdin.write(rgb.tobytes())
            except BrokenPipeError:
                raise ValueError('FFmpeg: '+proc.stderr.read().decode(errors='replace')[-1000:])
            count+=1
            if progress:progress(count,total,f"Рендер и кодирование: {shot.info['label']}")
        for world,info in timeline.frames():
            shot=Shot(world,info,options,width)
            if _daily:events.clear()
            events.extend(dict(e,time=count/fps-(.5 if _daily else 0)) for e in info['captures'])
            duration=initial if world.elapsed==0 else per_day
            for j in range(duration):write(shot,j/max(1,duration-1))
        for j in range(outro):write(shot,1)
        if progress:progress(count,total,'Завершение H.264, индексация MP4 / faststart')
        proc.stdin.close()
        error=proc.stderr.read().decode(errors='replace'); rc=proc.wait()
        if rc:raise ValueError('FFmpeg: '+error[-1000:])
        return dict(frames=count,width=width,height=height,date=world.date.isoformat(),
                    days=timeline.days,cached=not built,duration=count/fps,seconds_per_day=per_day/fps)
    except BaseException:
        if proc is not None and proc.poll() is None:proc.kill();proc.wait()
        Path(path).unlink(missing_ok=True)
        raise


def preview(store,chat_id,width=1920,options=None,progress=None):
    options=validate(options if options is not None else get_settings(store,chat_id))
    if progress:progress(0,3,'Загрузка текущего состояния')
    world,_=store.load(chat_id)
    from settlements import Settlements
    counts=Settlements(world).describe(world,{})[1]
    info=dict(label=world.date.strftime('%d.%m.%Y'),days=world.elapsed,stats=world.stats(),
              territory=int(world.territory.sum()),cities=counts,activity=[],
              source_size=[world.control.shape[1],world.control.shape[0]])
    timeline=Timeline(store,chat_id)
    with store.connect() as db:
        row=db.execute('SELECT f.info FROM render_frames f JOIN render_cache c USING(chat_id) WHERE f.chat_id=? AND c.digest=? ORDER BY f.day DESC LIMIT 1',(chat_id,timeline.digest)).fetchone()
    if row:
        import json
        info['activity']=json.loads(row[0])['activity']
        from city_catalog import present_info
        recent=[]
        with store.connect() as db:
            rows=db.execute('SELECT info FROM render_frames WHERE chat_id=? ORDER BY day',(chat_id,)).fetchall()
        for saved, in rows:
            recent=present_info(json.loads(saved),recent)['recent']
        info['recent']=recent
    if progress:progress(1,3,'Постобработка карты и композиция UI')
    shot=Shot(world,info,options,width)
    ui=Broadcast(width)
    rgb=shot.frame(0,[],1,ui.width,ui.height)
    if progress:progress(3,3,'Cinematic preview готов')
    return rgb
