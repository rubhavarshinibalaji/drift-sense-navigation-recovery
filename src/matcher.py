import cv2
import numpy as np
class MultiScaleLocalizer:
 def __init__(self,scales=(.25,.35,.5,.7,1,1.25,1.5,2)): self.scales=scales
 def predict(self,r,s):
  cs=[]
  for q in self.scales:
   h,w=r.shape; tw=max(8,int(w*q)); th=max(8,int(h*q))
   if tw>=s.shape[1] or th>=s.shape[0]: continue
   t=cv2.resize(r,(tw,th),interpolation=cv2.INTER_AREA); z=cv2.matchTemplate(s,t,cv2.TM_CCOEFF_NORMED)
   y,x=np.unravel_index(np.argmax(z),z.shape); cs.append((float(z[y,x]),x+tw//2,y+th//2))
  if not cs: raise RuntimeError('No valid scale')
  best=max(cs,key=lambda c:c[0]); return int(best[1]),int(best[2])
