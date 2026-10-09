"""FFmpeg discovery and classic daily cadence for the v1.4 broadcast export."""
import shutil


def ffmpeg_executable():
    executable=shutil.which('ffmpeg')
    if executable: return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError,RuntimeError) as e:
        raise ValueError('Не найден FFmpeg. Повторите pip install -r requirements.txt или установите FFmpeg в PATH') from e


def export_video(store,chat_id,path,fps=30,width=1920,curved=False,progress=None):
    """Classic cadence (initial + one frame/day), with the v1.4 broadcast UI."""
    from cinematic import export_cinematic
    return export_cinematic(store,chat_id,path,fps,width,progress=progress,
                            _daily=True,_curve=curved)
