import json
import os
import subprocess
import tempfile
import numpy as np
import pytest
from test_storage import store,attack,assert_same
from exporter import export_video

@pytest.mark.parametrize('curved,width',[(False,320),(True,320),(False,3840)])
def test_export_video(store,curved,width):
    attack(store)
    before=store.load(1)[0]
    fd,path=tempfile.mkstemp(suffix='.mp4'); os.close(fd)
    try:
        updates=[]
        info=export_video(store,1,path,30,width,curved,progress=lambda *args:updates.append(args))
        assert info['frames']==7
        assert [done for done,total,stage in updates if stage.startswith('Рендер и кодирование:')]==list(range(1,8))
        assert all(total==7 for done,total,stage in updates)
        assert updates[-1][2].startswith('Завершение H.264')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',path]))
        stream=probe['streams'][0]
        assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
        assert int(stream['nb_read_frames'])==7 and stream['width']==width
        assert stream['avg_frame_rate']=='30/1'
        data=open(path,'rb').read(); assert data.find(b'moov')<data.find(b'mdat')
        assert_same(before,store.load(1)[0])
        subprocess.run(['ffmpeg','-v','error','-i',path,'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    finally: os.unlink(path)

def test_bundled_ffmpeg_without_system_install(store,monkeypatch):
    import exporter
    monkeypatch.setattr(exporter.shutil,'which',lambda _:None)
    fd,path=tempfile.mkstemp(suffix='.mp4'); os.close(fd)
    try:
        assert exporter.ffmpeg_executable()
        info=exporter.export_video(store,1,path,10,320)
        assert info['frames']==1 and os.path.getsize(path)>100
    finally: os.unlink(path)
