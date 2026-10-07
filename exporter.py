"""Stream replay frames directly to FFmpeg: no PNG sequence or export directory."""
import shutil
import subprocess
from pathlib import Path
from renderer import render,Curve
import config


def ffmpeg_executable():
    executable=shutil.which('ffmpeg')
    if executable: return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError,RuntimeError) as e:
        raise ValueError('Не найден FFmpeg. Повторите pip install -r requirements.txt или установите FFmpeg в PATH') from e


def export_video(store,chat_id,path,fps=30,width=1920,curved=False):
    if not 1<=fps<=config.MAX_FPS: raise ValueError('FPS: 1..60')
    if not 64<=width<=config.MAX_EXPORT_WIDTH: raise ValueError('WIDTH: 64..4096')
    width=width//2*2
    ffmpeg=ffmpeg_executable()
    process=None; curve=None; count=0
    def frame(world):
        nonlocal process,curve,count
        rgb=render(world,width,date_label=not curved)
        h=rgb.shape[0]//2*2; rgb=rgb[:h]
        if curved:
            if curve is None: curve=Curve(rgb.shape)
            rgb=curve.apply(rgb)
            # Date stays straight after visual deformation.
            from PIL import Image,ImageDraw,ImageFont
            im=Image.fromarray(rgb); draw=ImageDraw.Draw(im)
            try: font=ImageFont.truetype('DejaVuSans.ttf',max(12,width//80))
            except OSError: font=ImageFont.load_default()
            draw.text((12,12),world.date.strftime('%d.%m.%Y'),font=font,fill='white',stroke_width=2,stroke_fill='black')
            import numpy as np
            rgb=np.array(im)
        if process is None:
            process=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{h}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try: process.stdin.write(rgb.tobytes())
        except BrokenPipeError:
            raise ValueError('FFmpeg: '+process.stderr.read().decode(errors='replace')[-1000:])
        count+=1
    try:
        final=store.replay(chat_id,on_start=frame,on_day=frame)
        process.stdin.close(); error=process.stderr.read().decode(errors='replace'); rc=process.wait()
        if rc: raise ValueError('FFmpeg: '+error[-1000:])
        return {'frames':count,'width':width,'height':int(round(final.control.shape[0]*width/final.control.shape[1]))//2*2,'date':final.date.isoformat()}
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill(); process.wait()
        Path(path).unlink(missing_ok=True)
        raise
