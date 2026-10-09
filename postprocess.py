"""Map-only color and detail processing; the HUD is drawn afterward."""
import cv2
import numpy as np


def process(rgb, enabled=True):
    if not enabled:
        return rgb.copy()
    # A small edge-preserving pass smooths texture noise without smearing
    # contrasting town markers or political boundaries.
    smooth = cv2.bilateralFilter(rgb, 5, 14, 1.6)
    image = (rgb.astype(np.float32)*.7 + smooth.astype(np.float32)*.3)/255
    lum = image@np.array([.2126,.7152,.0722],np.float32)
    image = lum[:,:,None]+(image-lum[:,:,None])*1.05
    image = (image-.35)*1.075+.35
    # Restrained broadcast-grade shadows and slightly cooler highlights.
    image[:,:,0] *= .992
    image[:,:,1] *= 1.015
    image[:,:,2] *= 1.026
    # Wide-scale terrain detail and a restrained final unsharp pass.
    broad = cv2.GaussianBlur(lum,(0,0),3.)
    image += np.clip(lum-broad,-.08,.08)[:,:,None]*.15
    blurred = cv2.GaussianBlur(image,(0,0),.65)
    image += np.clip(image-blurred,-.06,.06)*.32
    return np.clip(np.rint(image*255),0,255).astype(np.uint8)


def attack_pulse(rgb, world, phase, enabled=True):
    if not enabled or world.fresh_capture is None or not world.fresh_capture.any():
        return rgb.copy()
    changed = cv2.resize(world.fresh_capture.astype(np.float32), (rgb.shape[1],rgb.shape[0]), interpolation=cv2.INTER_AREA)
    amount = .10*(1-phase)**2
    alpha = changed[:,:,None]*amount
    return np.clip(rgb.astype(np.float32)*(1-alpha)+np.array([169,243,143])*alpha,0,255).astype(np.uint8)
