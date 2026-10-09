"""Reconstruct one *actual* 4K film frame without rendering the full MP4.

Use the same dates, campaign snapshots, capture queue and Shot.frame() math as
export_lore(..., fps=30, width=3840, map_fps=10).
"""
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lore_campaign import Campaign, dates, load_world
from cinematic import Shot
from visual_settings import validate, settings as get_settings
from storage import Store


def render_frame(number=239, output="aurelia_frame_0239.png"):
    if number < 1:
        raise ValueError("Frame numbering starts at 1")
    fps, map_fps, width = 30, 10, 3840
    height = 2160
    frames_per_scene = max(2, round(fps * .20))
    ticks_between_maps = fps // map_fps
    moments = dates()
    total = len(moments) * frames_per_scene
    if number > total:
        raise ValueError(f"No frame {number}; film has {total} frames")
    index = number - 1

    store = Store("preview_frame_0239.sqlite3")
    options = validate(get_settings(store, 0))
    world = load_world()
    campaign = Campaign(world)
    events = []
    info = None
    shot = None
    rendered_map_tick = None

    for frame_index in range(index + 1):
        scene, k = divmod(frame_index, frames_per_scene)
        dt = moments[scene]
        following = moments[min(scene + 1, len(moments) - 1)]
        if shot is None or frame_index % ticks_between_maps == 0:
            days = (following - dt).days
            when = dt + timedelta(days=min(days, round(days * k / frames_per_scene)))
            info = campaign.at(when)
            events[:] = [e for e in events if frame_index/fps - e["time"] < 2.2]
            events.extend(dict(e, time=frame_index/fps) for e in info["captures"][-6:])
            rendered_map_tick = frame_index
            if frame_index >= index - ticks_between_maps + 1:
                shot = Shot(world, info, options, width)
            else:
                # Only compute expensive 4K frame on final 10-FPS map tick.
                shot = "cached-placeholder"
        if frame_index == index:
            if not isinstance(shot, Shot):
                shot = Shot(world, info, options, width)
            phase = (frame_index % ticks_between_maps) / max(1, ticks_between_maps - 1)
            frame = shot.frame(frame_index / fps, events, phase, width, height)
            Image.fromarray(frame).save(output, optimize=True)
            meta = dict(frame=number,frame_index=frame_index,total=total,
                        date=info["label"],time_seconds=frame_index / fps,
                        last_map_tick=rendered_map_tick,video_fps=fps,
                        map_fps=map_fps,resolution=[width,height],
                        capture_notifications=len(events),
                        ui=True)
            Path(output+".json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
            return meta
    raise AssertionError("Frame loop should always produce output")


if __name__ == "__main__":
    print(json.dumps(render_frame(),ensure_ascii=False),flush=True)
