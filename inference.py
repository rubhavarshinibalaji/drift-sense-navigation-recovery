#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
from src.matcher import MultiScaleLocalizer

def main():
    p=argparse.ArgumentParser(description='Locate reference inside search image.')
    p.add_argument('reference'); p.add_argument('search')
    p.add_argument('--scales',default='0.25,0.35,0.5,0.7,1.0,1.25,1.5,2.0')
    p.add_argument('--top-k',type=int,default=20)
    a=p.parse_args()
    if not Path(a.reference).is_file(): raise FileNotFoundError(a.reference)
    if not Path(a.search).is_file(): raise FileNotFoundError(a.search)
    ref=cv2.imread(a.reference,cv2.IMREAD_GRAYSCALE); search=cv2.imread(a.search,cv2.IMREAD_GRAYSCALE)
    if ref is None or search is None: raise ValueError('Could not read one or both images.')
    scales=tuple(float(x) for x in a.scales.split(',') if x.strip())
    x,y=MultiScaleLocalizer(scales=scales,top_k=a.top_k).predict(ref,search)
    print(f'({x}, {y})')
if __name__=='__main__': main()
