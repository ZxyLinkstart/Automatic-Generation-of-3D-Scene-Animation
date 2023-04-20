import pickle as pkl
import numpy as np
import os


pkldatafilepath = '/data1/zxy/Long-term-Motion-in-3D-Scenes/data/PROXD/MPH11_00034_01/results/s001_frame_00001__00.00.00.027/000.pkl'

data = pkl.load(open(pkldatafilepath, "rb"))

meta = {}