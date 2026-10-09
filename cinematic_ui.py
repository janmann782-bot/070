"""Flat Aurelia map layout. Presentation only, designed at 1920 × 1080."""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MINT = '#A9F38F'
WHITE = '#E5EDDF'
MUTED = '#94A28E'
ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=64)
def font(size, isaac=False):
    if isaac:
        return ImageFont.truetype(str(ROOT/'Isaac.ttf'), size)
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.truetype(str(ROOT/'Isaac.ttf'), size)


def ease(x):
    x = min(1., max(0., x))
    return x*x*(3.-2.*x)


def fitted(draw, xy, text, size=20, fill=WHITE, isaac=False, limit=400):
    face = font(size, isaac)
    while draw.textlength(text, font=face) > limit and size > 12:
        size -= 1
        face = font(size, isaac)
    if draw.textlength(text, font=face) > limit:
        while text and draw.textlength(text+'…', font=face) > limit:
            text = text[:-1]
        text += '…'
    draw.text(xy, text, font=face, fill=fill)


def intersects(a, b):
    return a[0] < b[2]+6 and a[2]+6 > b[0] and a[1] < b[3]+6 and a[3]+6 > b[1]


class Broadcast:
    map_box = (480, 0, 1920, 1080)  # Native 4:3 map, full frame, no crop.

    def __init__(self, width=1920):
        if not 320 <= width <= 4096:
            raise ValueError('Cinematic WIDTH: 320..4096')
        self.width = width//2*2
        self.height = (round(self.width*9/16)//2)*2

    def compose(self, map_rgb, info, events=(), now=0., settings=None):
        settings = settings or {}
        im = Image.new('RGB', (1920, 1080), '#070B08')
        d = ImageDraw.Draw(im)
        source = Image.fromarray(map_rgb)
        x0, y0, x1, y1 = self.map_box
        ratio = min((x1-x0)/source.width, (y1-y0)/source.height)
        mw, mh = round(source.width*ratio), round(source.height*ratio)
        mx, my = x0+(x1-x0-mw)//2, y0+(y1-y0-mh)//2
        im.paste(source.resize((mw, mh), Image.Resampling.LANCZOS), (mx, my))
        self.actual_map = (mx, my, mw, mh)
        recent = info.get('recent', []) if settings.get('notifications', True) else []
        # The contour belongs to the panel's actual edge next to the map.
        d.line((x0-2, 0, x0-2, 1079), fill=MINT, width=2)
        fitted(d, (80, 34), 'AURELIA', 52, MINT, True, 320)
        fitted(d, (80, 101), '@AureliaAOH2', 20, MINT, True, 320)
        fitted(d, (80, 148), 'ЙОГУРТСТАНСКАЯ', 24, WHITE, True, 320)
        fitted(d, (80, 180), 'ВОЙНА', 24, WHITE, True, 320)
        fitted(d, (80, 224), info.get('label', '28.06.2057'), 42, MINT, True, 320)
        fitted(d, (80, 278), f"ДЕНЬ {info.get('days', 0):03d}", 18, MUTED, True, 320)

        # The sidebar contains only the elements selected by the user:
        # branding, war title, date/day and the four latest captures.
        d.line((80, 318, 400, 318), fill='#344730', width=1)
        fitted(d, (80, 337), 'ПОСЛЕДНИЕ ЗАХВАТЫ', 20, MINT, True, 320)
        if not recent:
            fitted(d, (80, 385), 'Пока нет', 14, MUTED, False, 320)
        for i, e in enumerate(reversed(recent[-4:])):
            y = 385+i*64
            fitted(d, (80, y), e.get('name', 'Без названия'), 22, WHITE, True, 320)
            side = 'Кефирстан' if e['side']==1 else 'Йогуртстан'
            detail = f"{side} · {e.get('label', 'день '+str(e.get('day', 0)))}"
            fitted(d, (80, y+29), detail, 14, MUTED, False, 320)
        if settings.get('notifications', True) or settings.get('city_glow', True):
            self._events(im, events, now, settings, info.get('source_size', source.size))
        result = im if self.width == 1920 else im.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.asarray(result).copy()

    def _events(self, im, events, now, settings, source_size):
        # One small square and a name. No expanding circles, glow or side colors.
        active = [e for e in events if 0 <= now-e['time'] < 2.2]
        mx, my, mw, mh = self.actual_map
        layer = Image.new('RGBA', im.size)
        d = ImageDraw.Draw(layer)
        occupied = []
        for e in reversed(active):  # Newest labels receive the closest free positions.
            age = now-e['time']
            alpha = ease(age/.22)*(1-ease((age-1.6)/.6))
            x = round(mx+e['x']/source_size[0]*mw)
            y = round(my+e['y']/source_size[1]*mh)
            if settings.get('city_glow', True):
                pulse = .82+.18*math.cos(age*math.tau)
                d.rectangle((x-4,y-4,x+4,y+4), fill=(7,11,8,int(230*alpha)))
                d.rectangle((x-3,y-3,x+3,y+3), outline=(169,243,143,int(255*alpha*pulse)), width=2)
            if not settings.get('notifications', True):
                continue
            name = e.get('name', 'Без названия')
            size = 20
            face = font(size, True)
            while d.textlength(name,font=face)>350 and size>12:
                size -= 1
                face = font(size, True)
            label = name
            if d.textlength(label,font=face)>350:
                while label and d.textlength(label+'…',font=face)>350:
                    label=label[:-1]
                label+='…'
            tw = math.ceil(d.textlength(label,font=face))
            w,h = tw+14,32
            positions = [(x+12,y-16),(x-w-12,y-16),(x-w//2,y-44),(x-w//2,y+12)]
            box = None
            for px,py in positions:
                px = max(mx+3,min(mx+mw-w-3,px)); py=max(my+3,min(my+mh-h-3,py))
                candidate=(px,py,px+w,py+h)
                if not any(intersects(candidate,b) for b in occupied):
                    box=candidate;break
            if box is None:
                continue  # The sidebar still lists the name; keep a crowded map readable.
            occupied.append(box)
            px,py,_,_=box
            opacity = int(140*alpha)  # City names: 45% transparent, 55% opaque.
            # 45% opaque backdrop; text keeps its full contrast during the hold.
            d.rectangle(box,fill=(7,11,8,int(115*alpha)))
            # PIL's anchor normalizes the supplied font's unusual ascender metrics.
            d.text((px+7,py+6),label,font=face,fill=(169,243,143,opacity),anchor='lt')
        im.paste(layer,(0,0),layer)
