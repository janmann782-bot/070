import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from palette import COLORS, CONTESTED, FRESH_CAPTURE
from layers import png_bytes
import config

def render(world, width=None, date_label=False, stamp=None, postprocess=False):
    rgb=world.background[:,:,:3].copy(); territory=world.territory
    for (home,side),color in COLORS.items():
        rgb[(world.homeland==home)&(world.control==side)]=color
    if world.fresh_capture is not None:
        for side,color in FRESH_CAPTURE.items():
            rgb[world.fresh_capture&territory&(world.control==side)]=color
    if world.layers.terrain is not None:
        lum=cv2.cvtColor(world.layers.terrain[:,:,:3],cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Luminance modulation preserves side hue; height remains clearly visible.
        strength=world.layers.settings['TERRAIN_STRENGTH']
        rgb[territory]=np.rint(rgb[territory].astype(np.float32)*(1-strength)+lum[territory,None]*strength).astype(np.uint8)
    rgb[world.disputed&territory]=CONTESTED
    if world.layers.cities is not None:
        city=world.layers.cities; alpha=city[:,:,3:4].astype(np.float32)/255
        rgb=np.rint(rgb*(1-alpha)+city[:,:,:3]*alpha).astype(np.uint8)
    if width is not None and width!=rgb.shape[1]:
        height=round(rgb.shape[0]*width/rgb.shape[1])
        rgb=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_LANCZOS4 if postprocess else (cv2.INTER_AREA if width<rgb.shape[1] else cv2.INTER_NEAREST))
    if postprocess:
        from postprocess import process
        rgb=process(rgb)
    if date_label or stamp:
        im=Image.fromarray(rgb); draw=ImageDraw.Draw(im)
        size=max(12,round(im.width/100))
        try: font=ImageFont.truetype('DejaVuSans.ttf',size)
        except OSError: font=ImageFont.load_default(size=size)
        text=world.date.strftime('%d.%m.%Y')
        if stamp: text+=' | '+stamp
        bounds=draw.textbbox((0,0),text,font=font)
        pad=max(5,size//3)
        draw.rectangle((pad,pad,bounds[2]+3*pad,bounds[3]+3*pad),fill=(25,25,29))
        draw.text((2*pad,2*pad),text,font=font,fill='white')
        rgb=np.array(im)
    return rgb

def to_png(world,width=None,stamp=None,postprocess=True): return png_bytes(render(world,width,stamp=stamp,postprocess=postprocess))

class Curve:
    """Mild perspective + bow, only after rendering. Fixed output resolution."""
    def __init__(self,shape):
        h,w=shape[:2]; yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
        u=(xx-w/2)/(w/2); v=(yy-h/2)/(h/2)
        # Reverse remap: mild narrower top and curved lower edge; not a globe.
        scale=1-.035*(1-v)
        self.mx=((xx-w/2)/scale+w/2).astype(np.float32)
        self.my=(yy+.045*h*u*u+.012*h*v*u*u).astype(np.float32)
    def apply(self,rgb):
        return cv2.remap(rgb,self.mx,self.my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(242,242,242))
