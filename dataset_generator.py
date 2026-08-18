#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import cv2,numpy as np

def dram(c,r):
 h,w=c.shape; px=int(r.integers(18,32)); py=int(r.integers(18,32)); lw=max(2,px//7)
 for y in range(py//2,h,py): cv2.rectangle(c,(0,y-lw//2),(w-1,y+lw//2),180,-1)
 for x in range(px//2,w,px): cv2.rectangle(c,(x-lw//2,0),(x+lw//2,h-1),120,-1)
 for y in range(py//2,h,py):
  for x in range(px//2,w,px):
   if ((x//px)+(y//py))%2==0: cv2.circle(c,(x,y),max(2,lw//2),235,-1)
 return c

def finfet(c,r):
 h,w=c.shape; p=int(r.integers(12,22)); gw=max(2,p//5)
 for x in range(p,w,p): cv2.rectangle(c,(x-gw//2,0),(x+gw//2,h-1),190,-1)
 for y in range(p*2,h,p*3):
  cv2.rectangle(c,(0,y,w-1,y+max(2,gw//2)),100,-1)
  for x in range(p,w,p*2): cv2.circle(c,(x,y),max(2,gw),235,-1)
 return c

def degrade(img,r):
 o=img.astype(np.float32)*float(r.uniform(.8,1.2))+float(r.uniform(-18,18))
 if r.random()<.85:
  k=int(r.choice([3,5,7])); o=cv2.GaussianBlur(o,(k,k),float(r.uniform(.3,1.6)))
 o+=r.normal(0,float(r.uniform(2,14)),o.shape); return np.clip(o,0,255).astype(np.uint8)

def pair(style,r,n):
 b=np.zeros((n,n),np.uint8); b[:]=int(r.integers(20,60)); b=dram(b,r) if style=='DRAM' else finfet(b,r)
 b=np.clip(b.astype(np.float32)+r.normal(0,5,b.shape),0,255).astype(np.uint8); m=180
 cx=int(r.integers(m,n-m)); cy=int(r.integers(m,n-m)); rw=int(r.integers(70,120)); rh=int(r.integers(70,120))
 ref=b[cy-rh//2:cy+rh//2,cx-rw//2:cx+rw//2].copy(); return degrade(ref,r),degrade(b,r),(cx,cy)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--architecture',choices=['DRAM','FinFET'],required=True); p.add_argument('--num-pairs',type=int,required=True); p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--search-size',type=int,default=1000); a=p.parse_args()
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); r=np.random.default_rng(a.seed); records=[]
 for i in range(a.num_pairs):
  ref,s,c=pair(a.architecture,r,a.search_size); rn=f'reference_{i:04d}.png'; sn=f'search_{i:04d}.png'; cv2.imwrite(str(out/rn),ref); cv2.imwrite(str(out/sn),s); records.append({'id':i,'architecture':a.architecture,'reference':rn,'search':sn,'true_center_xy':[c[0],c[1]]})
 (out/'ground_truth.json').write_text(json.dumps(records,indent=2)); print(f'Generated {a.num_pairs} pairs in {out}')
if __name__=='__main__': main()
