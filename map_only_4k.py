"""Native cartographic 4K war film, 10 FPS, no sidebar or broadcast UI.

Keeps the entire 4:3 territory visible inside a 3840x2160 UHD stream.
Side margins are flat black instead of cropping geographical boundaries.
"""
from __future__ import annotations
import json
import math
import subprocess
from collections import deque
from pathlib import Path
from datetime import date
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lore_campaign import Campaign, load_world, dates, START, END
from renderer import render
from exporter import ffmpeg_executable

WIDTH, HEIGHT, FPS = 3840, 2160, 10
MAP_W = 2880
MARGIN_X = (WIDTH - MAP_W) // 2
PILLAR = (6, 10, 11)
CAPTURE_RGB = {1: (255,167,164), 2: (118,171,255)}


def shade_mask():
    # Dark peripheral corners, preserving interior military colors.
    y,x = np.ogrid[-1:1:complex(HEIGHT),-1:1:complex(MAP_W)]
    distance=np.maximum(0, .64*x*x + .68*y*y - .20)
    return np.rint(255*(1-np.minimum(.24,distance*.17))).astype(np.uint8)


def make_lut():
    # Slightly brighter mids with crisp but restrained separation.
    t=np.arange(256,dtype=np.float32)/255
    t=np.clip((t-.36)*1.065+.36,0,1)
    t=np.clip(np.power(t,.975),0,1)
    return np.rint(t*255).astype(np.uint8)


def draw_capture_names(canvas, events, font):
    if not events:
        return
    overlay=Image.new('RGBA',(WIDTH,HEIGHT))
    draw=ImageDraw.Draw(overlay)
    used=[]
    for e in reversed(events):
        x=int(MARGIN_X+e['x']/4096*MAP_W)
        y=int(e['y']/3072*HEIGHT)
        color=CAPTURE_RGB.get(e['side'],(169,243,143))
        txt=str(e['name'])
        if not txt or txt=='Без названия':continue
        box=draw.textbbox((0,0),txt,font=font,stroke_width=0)
        tw=box[2]-box[0]
        th=box[3]-box[1]
        if tw>MAP_W//3:continue
        w,h=tw+18, max(37,th+14)
        positions=((x+13,y-15),(x-w-13,y-15),(x-w//2,y-48),(x-w//2,y+20))
        selected=None
        for sx,sy in positions:
            sx=max(MARGIN_X+4,min(MARGIN_X+MAP_W-w-4,sx))
            sy=max(4,min(HEIGHT-h-4,sy))
            rect=(sx,sy,sx+w,sy+h)
            if not any(sx<v[2]+6 and sx+w>v[0]-6 and sy<v[3]+6 and sy+h>v[1]-6 for v in used):
                selected=rect
                break
        if selected is None:continue
        used.append(selected)
        x0,y0,x1,y1=selected
        draw.rectangle(selected,fill=(7,11,8,112))
        # 55% text opacity, matching original Aurelia overlay request.
        draw.text((x0+9,y0+7),txt,font=font,fill=(*color,155))
        draw.rectangle((x-3,y-3,x+3,y+3),outline=(*color,225),width=2)
    canvas=Image.fromarray(canvas)
    canvas.paste(overlay,(0,0),overlay)
    return np.asarray(canvas)


def export(path,progress=None):
    world=load_world()
    campaign=Campaign(world)
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    shade=shade_mask()
    shade3=cv2.merge((shade,shade,shade))
    lut=make_lut()
    font=ImageFont.truetype(str(Path(__file__).resolve().with_name('Isaac.ttf')),26)
    when_list=dates()
    command=[
        ffmpeg_executable(),'-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{WIDTH}x{HEIGHT}','-framerate',str(FPS),
        '-i','pipe:0','-an','-c:v','libx264','-threads','2',
        '-preset','veryfast','-crf','23','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(target),
    ]
    proc=None
    recent=deque(maxlen=10)
    try:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        for i,when in enumerate(when_list):
            info=campaign.at(when)
            if np.any((world.control==2)&(world.homeland==1)):
                raise ValueError('Канон нарушен: Йогуртстан вторгся в Кефирстан')
            rgb=render(world,width=MAP_W,postprocess=False)
            rgb=cv2.LUT(rgb,lut)
            rgb=cv2.multiply(rgb,shade3,scale=1/255)
            frame=np.empty((HEIGHT,WIDTH,3),np.uint8)
            frame[:]=PILLAR
            frame[:,MARGIN_X:MARGIN_X+MAP_W]=rgb
            for event in info['captures'][-8:]:
                recent.append(dict(event,expires=i+12))
            recent=deque((e for e in recent if e['expires']>i),maxlen=10)
            if recent:
                frame=draw_capture_names(frame,list(recent),font)
            try:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError:
                raise ValueError('Ошибка кодирования MP4: '+proc.stderr.read().decode(errors='replace')[-1000:])
            if progress and (i%8==0 or i==len(when_list)-1):
                progress(i+1,len(when_list),when.strftime('%d.%m.%Y'))
        proc.stdin.close()
        stderr=proc.stderr.read().decode(errors='replace')
        if proc.wait()!=0:
            raise ValueError('FFmpeg: '+stderr[-1500:])
        if not target.is_file() or target.stat().st_size==0:
            raise ValueError('Пустой MP4')
        return dict(width=WIDTH,height=HEIGHT,fps=FPS,frames=len(when_list),
                    seconds=len(when_list)/FPS,bytes=target.stat().st_size,
                    start=START.isoformat(),end=END.isoformat(),map_only=True)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        target.unlink(missing_ok=True)
        raise


if __name__=='__main__':
    import sys
    destination=sys.argv[1] if len(sys.argv)>1 else 'aurelia_yogurtstan_4k_10fps_map.mp4'
    report=export(destination,lambda n,t,d: print(f'{n}/{t}: {d}',flush=True))
    Path(destination+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
