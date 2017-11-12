import pandas as pd
import numpy as np
import sys,os

images_df = pd.read_json('/home/fangbo/scences/ai_challenger_scene_validation_20170908/val.json')
images = images_df.values
print(np.shape(images))
imgfile = images[:,0:3:2]


train_txt = open('train.txt', 'w')
for img in imgfile:
    strimg1 = str(img[0])
    strimg2 = str(img[1])
    str2 = strimg1 + ' ' +  strimg2 + '\n'
    train_txt.writelines(str2)
train_txt.close()
