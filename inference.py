#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
from src.matcher import MultiScaleLocalizer

def main():
 p=argparse.ArgumentParser(); p.add_argument('reference'); p.add_argument('search'); a=p.parse_args()
 if not Path(a.reference).is_file(): raise FileNotFoundError(a.reference)
 if not Path(a.search).is_file(): raise FileNotFoundError(a.search)
 r=cv2.imread(a.reference,0); s=cv2.imread(a.search,0)
 if r is None or s is None: raise ValueError('Could not read images')
 x,y=MultiScaleLocalizer().predict(r,s); print(f'({x}, {y})')
if __name__=='__main__': main()
