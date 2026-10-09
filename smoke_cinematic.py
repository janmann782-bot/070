"""Render the existing real-map 60-day war, without making any new orders."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
from storage import Store
from render_timeline import Timeline
from cinematic import export_cinematic, Shot
from visual_settings import validate


def probe(path):
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]))
    stream=result['streams'][0]
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    with open(path,'rb') as file:head=file.read(1024*1024)
    assert head.find(b'moov')>=0 and head.find(b'moov')<head.find(b'mdat')
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    return dict(width=stream['width'],height=stream['height'],frames=int(stream['nb_read_frames']),
                fps=stream['avg_frame_rate'],codec=stream['codec_name'],format=stream['pix_fmt'],
                duration=result['format']['duration'],bytes=Path(path).stat().st_size)


def main():
    root=Path(__file__).resolve().parent
    source=root/'example_state.sqlite3'; target=root/'cinematic_state.sqlite3'
    if not target.exists():shutil.copyfile(source,target)
    store=Store(target);chat=-2057
    with store.connect() as db:before=db.execute('SELECT initial,cache,revision FROM sessions WHERE chat_id=?',(chat,)).fetchone()
    import storage
    simulate=storage.simulate
    def forbidden(*args,**kwargs):raise AssertionError('UI/name rerender called simulate')
    existing_cache=Timeline(store,chat)
    with store.connect() as db:
        has_cache=db.execute('SELECT 1 FROM render_cache WHERE chat_id=? AND digest=?',(chat,existing_cache.digest)).fetchone()
    if has_cache:storage.simulate=forbidden
    timeline=Timeline(store,chat);timeline.ensure(lambda d,t,s:print(d,t,s,flush=True) if d%10==0 else None)
    captures=[]; active=[]; hashes=[]
    for world,info in timeline.frames():
        hashes.append(hashlib.sha256(world.control.tobytes()).hexdigest())
        captures.extend(info['captures'])
        time=1+(max(0,world.elapsed-1))*.4
        active.extend(dict(e,time=time) for e in info['captures'])
        active=[e for e in active if time-e['time']<2.2]
        if world.elapsed in [0,8,24,56,60]:
            shot=Shot(world,info,validate({}))
            rgb=shot.frame(time+.3,active,.75,1920,1080)
            Image.fromarray(rgb).save(root/f'cinematic_day{world.elapsed:02d}.png')
        if world.elapsed==56:Image.fromarray(rgb).save(root/'Aurelia_UI_v1_4.png')
    current,_=store.load(chat)
    assert np.array_equal(world.control,current.control)
    assert np.array_equal(world.disputed,current.disputed)
    assert hashes[-1]=='50cd43bea1f03349cbb95ee56002d5b794f62d3a8c95d8391faedc2afde730c5'
    def progress(done,total,stage):
        if done%60==0 or done==total:print(done,total,stage,flush=True)
    outputs={}
    outputs['normal']=export_cinematic(store,chat,root/'Aurelia_Cinematic_v1_4.mp4',30,1920,progress=progress)
    # Both renders reuse the existing visual timeline when present.
    storage.simulate=forbidden
    try:
        outputs['curved']=export_cinematic(store,chat,root/'Aurelia_Cinematic_v1_4_curved.mp4',10,1280,dict(curved=True),progress=progress)
    finally:storage.simulate=simulate
    for name,filename in [('normal','Aurelia_Cinematic_v1_4.mp4'),('curved','Aurelia_Cinematic_v1_4_curved.mp4')]:
        data=probe(root/filename)
        assert data['frames']==outputs[name]['frames']
        outputs[name]['probe']=data
    with store.connect() as db:assert db.execute('SELECT initial,cache,revision FROM sessions WHERE chat_id=?',(chat,)).fetchone()==before
    assert len(captures)==70 and all(c['named'] for c in captures)
    from settlements import Settlements
    places=Settlements(current).items
    report=dict(native_size=[4096,3072],days=60,start='2057-06-28',end='2057-08-27',
                control_sha256=hashes[-1],city_captures=len(captures),named_captures=sum(c['named'] for c in captures),
                named_markers=sum(c['named'] for c in places),total_markers=len(places),read_only=True,
                capture_names=[c['name'] for c in captures],ui_version='1.4.4',
                per_day_control_sha256=hashes,outputs=outputs)
    (root/'cinematic_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='per_day_control_sha256'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
