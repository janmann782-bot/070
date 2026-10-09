"""Replay the supplied 60-day real-map example with the daily visual overlay.

Run python -B verify_fresh_capture.py after smoke_real.py. No Telegram token needed.
Use --no-video to verify existing videos without encoding them again.
"""
import sys
sys.dont_write_bytecode=True
import os
os.environ.setdefault('NUMBA_NUM_THREADS','2')
import hashlib
import json
import subprocess
import time
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import config
from storage import Store
from layers import png_bytes
from renderer import render
from exporter import export_video


def main():
    root=config.ROOT; store=Store(root/'example_state.sqlite3'); chat=-2057
    report=json.loads((root/'smoke_report.json').read_text())
    old_sha=report['final_control_sha256']; previous=None; captured=0; panels=[]; crop_box=None
    started=time.perf_counter()
    def start(w):
        nonlocal previous
        previous=w.control.copy()
        assert w.fresh_capture is None
    def day(w):
        nonlocal previous,captured,crop_box
        delta=(w.control!=previous)&w.territory
        assert np.array_equal(delta,w.fresh_capture)
        previous=w.control.copy(); captured+=int(delta.sum())
        if w.elapsed in (20,40):
            (root/f'example_day{w.elapsed}.png').write_bytes(png_bytes(render(w,1920,date_label=True)))
        if w.elapsed==20:
            visible=delta&~w.disputed&(w.control==1)&(w.layers.city_influence<.01)
            n,labels,stats,centers=cv2.connectedComponentsWithStats(visible.astype(np.uint8),8)
            assert n>1
            best=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            x,y=np.rint(centers[best]).astype(int)
            x=max(210,min(x,w.control.shape[1]-210));y=max(180,min(y,w.control.shape[0]-180))
            crop_box=(slice(y-180,y+180),slice(x-210,x+210))
        if w.elapsed in (20,21):
            panels.append((w.date.strftime('%d.%m.%Y'),render(w)[crop_box]))
        if w.elapsed==60:
            (root/'example_preview.png').write_bytes(png_bytes(render(w,3840,date_label=True)))
            (root/'example_detail.png').write_bytes(png_bytes(render(w)[1050:1850,2350:3150]))
    print('Native 60-day replay and daily masks...',flush=True)
    final=store.replay(chat,on_start=start,on_day=day)
    control_sha=hashlib.sha256(final.control.tobytes()).hexdigest()
    assert control_sha==old_sha,'The visual overlay must never change the historical control state'
    assert final.elapsed==60 and captured>0
    # Two consecutive real-map frames, enlarged without smoothing away the small daily gains.
    preview=Image.new('RGB',(1704,838),(24,27,34));draw=ImageDraw.Draw(preview)
    font=ImageFont.truetype('DejaVuSans.ttf',25);small=ImageFont.truetype('DejaVuSans.ttf',20)
    for index,(date,rgb) in enumerate(panels):
        left=12+852*index
        draw.text((left,12),date,font=font,fill='white')
        preview.paste(Image.fromarray(rgb).resize((840,720),Image.Resampling.NEAREST),(left,52))
    draw.text((12,789),'Светло-красный: захвачено сегодня. На следующий день — обычный цвет.',font=small,fill=(255,190,187))
    preview.save(root/'fresh_capture_preview.png')
    # Repair the old example cache, retaining the authoritative operations and ownership.
    if '--no-video' in sys.argv:repaired=store.load(chat)[0]
    else:_,repaired=store.doctor(chat)
    assert np.array_equal(repaired.control,final.control)
    assert np.array_equal(repaired.fresh_capture,final.fresh_capture)
    videos={}
    for curved,name in [(False,'example.mp4'),(True,'example_curved.mp4')]:
        print('Export',name,flush=True)
        if '--no-video' in sys.argv:
            info={'frames':61,'width':1920,'height':1440,'date':final.date.isoformat()}
        else:info=export_video(store,chat,root/name,10,1920,curved)
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(root/name)]))['streams'][0]
        assert info['frames']==61 and int(stream['nb_read_frames'])==61
        assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
        subprocess.run(['ffmpeg','-v','error','-i',str(root/name),'-f','null','-'],check=True)
        videos[name]=info
    verification={'days':final.elapsed,'daily_delta_pixel_perfect':True,'control_unchanged':True,
        'control_sha256':control_sha,'daily_capture_pixels_total':captured,'videos':videos,
        'duration_seconds':round(time.perf_counter()-started,2)}
    (root/'fresh_capture_report.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DAILY HIGHLIGHT PASS',verification,flush=True)


if __name__=='__main__': main()
