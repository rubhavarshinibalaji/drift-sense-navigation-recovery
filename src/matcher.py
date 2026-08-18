import cv2
import numpy as np

class MultiScaleLocalizer:
    def __init__(self,scales=(.25,.35,.5,.7,1.0,1.25,1.5,2.0),top_k=20): self.scales=tuple(scales); self.top_k=top_k
    def _norm(self,x):
        x=x.astype(np.float32); s=float(x.std())
        return np.zeros_like(x) if s<1e-6 else (x-x.mean())/s
    def predict(self,reference,search):
        if reference.ndim!=2 or search.ndim!=2: raise ValueError('Images must be grayscale.')
        candidates=[]
        for scale in self.scales:
            h,w=reference.shape; tw=max(8,int(round(w*scale))); th=max(8,int(round(h*scale)))
            if tw>=search.shape[1] or th>=search.shape[0]: continue
            t=self._norm(cv2.resize(reference,(tw,th),interpolation=cv2.INTER_AREA)); s=self._norm(search)
            r=cv2.matchTemplate(s,t,cv2.TM_CCOEFF_NORMED); k=min(self.top_k,r.size)
            for idx in np.argpartition(r.ravel(),-k)[-k:]:
                y,x=np.unravel_index(idx,r.shape); candidates.append((float(r[y,x]),int(x+tw/2),int(y+th/2)))
        if not candidates: raise RuntimeError('No valid scale fits.')
        candidates.sort(reverse=True); kept=[]
        for c in candidates:
            if all((c[1]-q[1])**2+(c[2]-q[2])**2>20**2 for q in kept): kept.append(c)
        best=kept[0][0]; amb=[c for c in kept if c[0]>=best-.02]
        cx=search.shape[1]/2; cy=search.shape[0]/2
        return min(amb,key=lambda c:((c[1]-cx)**2+(c[2]-cy)**2,-c[0]))[1:]
