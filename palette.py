import numpy as np
KEFIR, YOGURT = 1, 2
COLORS = {(1,1):(74,0,0), (2,2):(0,19,74), (2,1):(177,40,35), (1,2):(31,82,166)}
ORDERS = {1:(255,0,255), 2:(0,255,102)}
CONTESTED = (112,116,122)
NAMES = {1:'Кефирстан', 2:'Йогуртстан'}
def decode(rgb):
    homeland = np.zeros(rgb.shape[:2], np.uint8)
    control = np.zeros_like(homeland)
    for (home, side), color in COLORS.items():
        hit = np.all(rgb[:,:,:3] == color, axis=2)
        homeland[hit], control[hit] = home, side
    if not np.any(control == 1) or not np.any(control == 2):
        raise ValueError('На карте нужны территории обеих сторон в точных политических цветах')
    homeland.flags.writeable = False
    return homeland, control
