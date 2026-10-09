import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('NUMBA_NUM_THREADS','2')
import asyncio
import logging
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='Aurelia native daily war bot')
    parser.add_argument('--self-check',action='store_true',help='offline startup/assets/FFmpeg check, no token needed')
    args=parser.parse_args()
    if args.self_check:
        import shutil
        from layers import read_png,Layers
        from state import World
        import config
        arrays={k:read_png((config.ROOT/('yogurtstan_demo_control.png' if k=='base_control' else f'yogurtstan_{k}.png')).read_bytes()) for k in ['base_control','terrain','cities','roads']}
        w=World.create(arrays.pop('base_control'),'2057-06-28',Layers(**arrays))
        import bot
        print('Python',sys.version.split()[0],'| aiogram imported | native map',w.control.shape[::-1])
        print('Territory pixels:',int(w.territory.sum()),'Cities:',int(w.layers.city_mask.sum()))
        from exporter import ffmpeg_executable
        print('FFmpeg:',ffmpeg_executable())
        return
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    print('Aurelia: launcher started; loading bot libraries...', flush=True)
    from bot import main as polling
    try: asyncio.run(polling())
    except (ValueError,OSError) as e: print(str(e),file=sys.stderr); sys.exit(1)

if __name__=='__main__': main()
