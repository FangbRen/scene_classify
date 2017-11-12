# coding=utf-8
import sys,os
import numpy as np
import pandas as pd
path = '/home/fangbo/caffe/'
sys.path.insert(0, path + 'python')
import caffe
os.chdir(path)
import json
caffe.set_mode_cpu()

deplot_prototxt_path = '/home/fangbo/caffe/examples/lmdb_img/ResNet_50_deploy.prototxt'
caffe_model_path = '/home/fangbo/caffe/examples/lmdb_img/resnet_iter_3500.caffemodel'

#加载网络模型
net = caffe.Net(
        deplot_prototxt_path,
        caffe_model_path,
        caffe.TEST
)
#python接口读取的数据为.npy所以需要把.binproto文件转换一下

def convert_mean(binmean,npymean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binmean,'rb').read()
    blob.ParseFromString(bin_mean)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    npy_mean = arr[0]#一个数组中可能有多组均值文件存在，选择其中一组
    np.save(npymean,npy_mean)

binMean=path + 'examples/lmdb_img/mean.binaryproto'
npyMean=path + 'examples/lmdb_img/mean.npy'
convert_mean(binMean,npyMean)

#对图像进行预处理
#设置预处理参数
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
# 改变维度的顺序，由原始图片维度(width, height, channel)变为(channel, width, height)
transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',np.load(npyMean).mean(1).mean(1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

#对单张图片的测试识别
img_txt = pd.read_csv(path + 'examples/lmdb_img/test.txt', sep='\n', encoding='utf8')
imgfile = img_txt.values
dic={}
a = []
lables_filename = '/home/fangbo/scences/test/scene_classes.csv'
labels = pd.read_csv('/home/fangbo/scences/test/scene_classes.csv')
labels = labels.values
#f = open(path + 'examples/lmdb_img/pre_json.json','w')
for img in imgfile:
    im = caffe.io.load_image('/home/fangbo/scences/test/test/'+img[0])
    # 预处理图片
    transformed_image = transformer.preprocess('data', im)
    net.blobs['data'].data[...] = transformed_image
    # 执行测试阶段
    # 前向传播
    out = net.forward()


    prob = net.blobs['prob'].data[0].flatten()  # 取出最后一层（Softmax）属于某个类别的概率值，并打印
    # print prob
    order = prob.argsort()[-1]  # 将概率值排序，取出最大值所在的序号
    #print (labels[order][0]) # 将该序号转换成对应的类别名称，并打印
    print(order)
    # 取出前五个较大值所在的序号
    top_inds = prob.argsort()[::-1][:3]
    #predict = zip(prob[top_inds], labels[top_inds][0])
    #print ('probabilities and labels:', labels[top_inds][0])
    #print(top_inds)
    dic = {"label_id":(top_inds[0],top_inds[1],top_inds[2]),"image_id":img[0]}
    #pre_json = json.dumps(dic)
    #f.writelines(pre_json)
    a.append(dic)
#f.close()
#ImageId=[i+1 for i in range(n)]
pre_json = json.dumps(a)

with open(path + 'examples/lmdb_img/preres.json','w') as f:
    #json.dump(a,f)
    f.write(pre_json)
    f.close()