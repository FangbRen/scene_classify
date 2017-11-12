import pandas as pd
import numpy as np
import array
import sys,os
import json
import argparse
import time
path = '/home/fangbo/caffe/'
a=[]
dic = {}
images_google = pd.read_json('/home/fangbo/caffe/examples/lmdb_img/pre.json')
google = images_google.values
images_res = pd.read_json('/home/fangbo/caffe/examples/lmdb_img/preres.json')
res = images_res.values
cout = 0
for key in range(7040):
    if google[key][1][0] == res[key][1][0] or google[key][1][0] == res[key][1][1] or google[key][1][0] == res[key][1][2] or \
            google[key][1][1] == res[key][1][0] or google[key][1][1] == res[key][1][1] or google[key][1][0] == res[key][1][2] or \
            google[key][1][2] == res[key][1][0] or google[key][1][2] == res[key][1][1] or google[key][1][2] == res[key][1][2]:
        cout += 1
        dic = {"label_id": (res[key][1][0], res[key][1][1], res[key][1][2]), "image_id": res[key][0]}
        a.append(dic)
    else:
        res[key][1][2] = google[key][1][0]
        dic = {"label_id": (res[key][1][0], res[key][1][1], res[key][1][2]), "image_id": res[key][0]}
        a.append(dic)

pre_json = json.dumps(a)

with open(path + 'examples/lmdb_img/preffff.json','w') as f:
    #json.dump(a,f)
    f.write(pre_json)
    f.close()