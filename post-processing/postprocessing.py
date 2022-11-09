# -*- coding: UTF-8 -*-

import cv2 as cv
import json
import numpy as np
import math
from subprocess import Popen
import os
import pickle
import scipy.ndimage


city='AerialKITTI'
modelname='test'

image_root=city+'/image/'
segment_root=city+'/Segmentation/'
config_root=city+'/config/'
roadnetwork_root=city+'/roadnetwork/'
viz_root=city+'/viz/数据集处理/'
viz_root2=city+'/viz/数据集处理onimage/'


output_root=city+'/output/mask/bothedge/cnngnn/%s/'%(modelname)
anno_root=city+'/output/annoresult/bothedge/cnngnn/test/'
viz_output_root=city+'/output/viz/'
Popen("mkdir -p %s" % (viz_output_root), shell=True).wait()
Popen("mkdir -p %s" % (output_root), shell=True).wait()
image_list=list(os.listdir(segment_root))
name_list=list(map(lambda x: x[:-4] , image_list))

name_list=['16','17','18','19','20']
output_folder=city+'/'

# COLOR=(250,206,135)
COLOR=(0,255,255)


def get_point(x,y,d,k):
    d=int(d)
    if k == None:
        x_try = x
        y_try = y - d
    else:
        a=math.atan(k)
        if (abs(k)) < 1:
            x_try = int(x + d * math.cos(a))
            y_try = int(y + d * math.sin(a))
        else:
            x_try = int(x + d * math.cos(a))
            y_try = int(y + d * math.sin(a))
    return x_try,y_try

def get_point2(x,y,d,dlat,dlon):
    x_try=x+d*dlon
    y_try=y+d*dlat
    x_try=int(x_try)
    y_try=int(y_try)
    return x_try,y_try

def generate_image(name):
    config=json.load(open(config_root+"config_"+name+'.json', "r"))
    image = scipy.ndimage.imread(config['image_dir']).astype(np.uint8)
    segmentation=scipy.ndimage.imread(config['segmentation_dir'],mode='P').astype(np.uint8)
    h=segmentation.shape[0]
    w=segmentation.shape[1]
    values=np.zeros((h, w,3), dtype='uint8')
    values2= np.zeros((h, w,3), dtype='uint8')
    viz=cv.imread(viz_root+name+'_viz.png')
    annotation = pickle.load(open(anno_root+name+'.p', "rw"))
    # annotation = pickle.load(open(anno_root+'annotation_osm_'+name+'.p', "rw"))
    roadNetwork = pickle.load(open(roadnetwork_root+"roadnetwork_"+ name +'.p', "r"))


    segmentation[segmentation>=125]=255
    segmentation[segmentation<=125]=0
    single=0

    nid2d1={}
    nid2d2={}
    nid2angle={}
    nid2headingvector={}
    single=0
    c=0
    #record test output(non intersection nodes)
    for node,neighbor in roadNetwork.node_degree.iteritems():
        if len(neighbor)<=2:
            nid2d1[node]=annotation[node]['d1']
            nid2d2[node]=annotation[node]['d2']
            # nid2angle[node]=annotation[node]['angle_vertical']
            nid2headingvector[node]=annotation[node]['heading_vector_vertical']

    drawed_edge=[]
    for edge in roadNetwork.edges:
        edge_verse=(edge[1],edge[0])
        if edge_verse in drawed_edge:
            continue
        node1=edge[0]
        node2=edge[1]
        loc1 = roadNetwork.nid2loc[node1]
        loc2 = roadNetwork.nid2loc[node2]
        if len(roadNetwork.node_degree[node1])>2:
            if len(roadNetwork.node_degree[node2])>2:
                print('error')
                continue
            else:
                # angle=annotation[node2]
                nid2d1[node1]=nid2d1[node2]
                nid2d2[node1]=nid2d2[node2]
                # nid2angle[node1]=nid2angle[node2]
                nid2headingvector[node1]=nid2headingvector[node2]
        if len(roadNetwork.node_degree[node2])>2:
            if len(roadNetwork.node_degree[node1])>2:
                print('error')
                continue
            else:
                nid2d1[node2]=nid2d1[node1]
                nid2d2[node2]=nid2d2[node1]
                # nid2angle[node2]=nid2angle[node1]
                nid2headingvector[node2]=nid2headingvector[node1]


        if len(roadNetwork.node_degree[node1])==1 or len(roadNetwork.node_degree[node1])>2:
            # draw circle on nodes
            if max(abs(nid2d1[node1]),abs(nid2d2[node1]))<=0:
                pass
            else:
                cv.circle(values, (loc1[1], loc1[0]), max(abs(nid2d1[node1]),abs(nid2d2[node1])), (255, 255, 255), -1)
                cv.circle(values2, (loc1[1], loc1[0]), max(abs(nid2d1[node1]),abs(nid2d2[node1])), COLOR, -1)

        if len(roadNetwork.node_degree[node2])==1 or len(roadNetwork.node_degree[node2])>2:
            if max(abs(nid2d1[node2]),abs(nid2d2[node2]))<=0:
                pass
            else:
                cv.circle(values, (loc2[1], loc2[0]), max(abs(nid2d1[node2]),abs(nid2d2[node2])), (255, 255, 255), -1)
                cv.circle(values2, (loc2[1], loc2[0]), max(abs(nid2d1[node2]),abs(nid2d2[node2])), COLOR, -1)




        angle1 = math.degrees(math.atan2(nid2headingvector[node1][0], nid2headingvector[node1][1]))
        angle1=(angle1+360)%360
        angle2 = math.degrees(math.atan2(nid2headingvector[node2][0], nid2headingvector[node2][1]))
        angle2=(angle2+360)%360
        dlat1=nid2headingvector[node1][0]
        dlon1=nid2headingvector[node1][1]
        dlat2=nid2headingvector[node2][0]
        dlon2=nid2headingvector[node2][1]
        d11=nid2d1[node1]
        d12=nid2d2[node1]
        d21=nid2d1[node2]
        d22=nid2d2[node2]
        p1=get_point2(loc1[1],loc1[0],d11,dlat1,dlon1)
        p2=get_point2(loc1[1],loc1[0],d12,dlat1,dlon1)
        p3=get_point2(loc2[1],loc2[0],d21,dlat2,dlon2)
        p4=get_point2(loc2[1],loc2[0],d22,dlat2,dlon2)

        if abs(angle1-angle2)>90:
            pts=np.array([p2,p1,p4,p3])
        else:
            pts=np.array([p1,p2,p4,p3])
        cv.fillPoly(values,np.int32([pts]),(255,255,255))
        cv.fillPoly(values2,np.int32([pts]),COLOR)
        drawed_edge.append(edge)

    viz_v=cv.addWeighted(viz,0.7,values2,0.3,0)
    # viz_v_image=cv.addWeighted(viz2,0.7,values2,0.3,0)
    cv.imwrite(viz_output_root+name+'_viz_onmask.png',viz_v)
    cv.imwrite(output_root+name+'_widthmask.png',values)
    print (name+'done!')


for name in name_list:
    generate_image(name)

