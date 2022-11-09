# -*- coding: UTF-8 -*-
import json
from subprocess import Popen
from time import time, sleep
import pdb
from road_network import *
import json
import cv2
from subprocess import Popen
import math
from PIL import Image
from scipy.misc import imresize
import sys
import numpy as np
import os
import pickle
import scipy.ndimage
import argparse

city = 'Wuhan'

parser = argparse.ArgumentParser()

parser.add_argument('--image_root', type=str, default=city + '/image/')
parser.add_argument('--center_root', type=str, default=city + '/Centerline/')
parser.add_argument('--mask_root', type=str, default=city + '/Segmentation/')
parser.add_argument('--resample_root', type=str, default=city + '/resample_graph/')
parser.add_argument('--config_root', type=str, default= city + '/config/')
parser.add_argument('--roadnetwork_root', type=str, default=city + '/roadnetwork/')
args = parser.parse_args()

image_root = args.image_root
center_root = args.center_root
segment_root = args.mask_root
resample_graph_root = args.resample_root
config_root = args.config_root
roadnetwork_root = args.roadnetwork_root

viz_root = city + '/viz/数据集处理/'
tjunction_viz_root = viz_root + 'test/'
# canny_root=city+'/canny_2/'
# width_graph_root=city+'/width_graph/'


Popen("mkdir -p %s" % (config_root), shell=True).wait()
Popen("mkdir -p %s" % (roadnetwork_root), shell=True).wait()
Popen("mkdir -p %s" % (tjunction_viz_root), shell=True).wait()

image_list = list(os.listdir(resample_graph_root))
name_list = list(map(lambda x: x[:-6], image_list))

output_folder = city + '/'
T = 10
WIDTH_MAX = 100


def read_graph(graph_dir):
    with open(graph_dir, "r") as f:
        lines = f.readlines()
    edge_info = []
    nodes_list = []
    is_edge = False
    for line in lines:
        temp = line.strip().split()
        if len(temp) >= 2 and is_edge == False:
            nodes_list.append([int(temp[0]), int(temp[1]), int(temp[2])])
            # nodes_list.append([int(temp[0]),int(temp[1])])
        elif len(temp) == 2 and is_edge:
            start = int(temp[0])
            end = int(temp[1])
            # if[end,start] in edge_info:
            #     continue
            edge_info.append([int(temp[0]), int(temp[1])])
        else:
            print("a white line and break")
            is_edge = True
    return nodes_list, edge_info


def generate_config_file(name):
    # for name in name_list:
    config = {}
    config['name'] = name
    # img=cv2.imread(image_root+name+'.jpg')
    img = cv2.imread(image_root + name + '.png')
    config['height'] = img.shape[0]
    config['width'] = img.shape[1]
    # config['graph_dir']=width_graph_root+name+'_width.graph'
    config['resample_graph_dir'] = resample_graph_root + name + '.graph'
    # config['image_dir']=image_root+name+'.jpg'
    config['image_dir'] = image_root + name + '.png'
    config['centerline_dir'] = center_root + name + '.png'
    # config['segmentation_dir']=segment_root+name+'.tif'
    config['segmentation_dir'] = segment_root + name + '.png'
    # config['canny_dir']=canny_root+name+'.png'
    json.dump(config, open(config_root + "config_" + name + '.json', "w"))


def generate_roadnetwork(name):
    roadNetwork = RoadNetwork()
    cfg = json.load(open(config_root + "config_" + name + '.json', "r"))
    roadNetwork.region = [cfg['height'], cfg['width']]
    roadNetwork.image_file = cfg['image_dir']
    # nodes,edges=read_graph(cfg['graph_dir'])
    nodes, edges = read_graph(cfg['resample_graph_dir'])
    # r_nodes= []
    # for node in nodes:
    #     n=roadNetwork.AddNode(node[1],node[0],node[2])
    #     r_nodes.append(n)
    for edge in edges:
        node1 = nodes[edge[0]]
        node2 = nodes[edge[1]]
        if node1[2] == 1 or node2[2] == 1:
            continue
        # nid1=roadNetwork.AddNode(node1[1],node1[0],node1[2])
        # nid2=roadNetwork.AddNode(node2[1],node2[0],node2[2])
        nid1 = roadNetwork.AddNode(node1[1], node1[0])
        nid2 = roadNetwork.AddNode(node2[1], node2[0])
        roadNetwork.AddEdge(nid1, nid2)

    roadNetwork.DumpToFile(roadnetwork_root + "roadnetwork_" + name + '.p')


def annotate_dataset_osm(name):
    config = json.load(open(config_root + "config_" + name + '.json', "r"))
    roadNetwork = pickle.load(open(roadnetwork_root + "roadnetwork_" + name + '.p', "rb"))
    # annotation_file = roadnetwork_root+"annotation_osm_"+config['name']+'.p'

    # if os.path.isfile(annotation_file):
    #     annotation = pickle.load(open(annotation_file, "r"))
    # else:
    annotation = {}
    for anid in roadNetwork.nid2loc.keys():
        annotation[anid] = {}
        annotation[anid]['remove'] = 0
        annotation[anid]['labelled'] = 1
        heading_vector_lat = 0
        heading_vector_lon = 0

        if len(roadNetwork.node_degree[anid]) > 2:
            heading_vector_lat = 0
            heading_vector_lon = 0
        elif len(roadNetwork.node_degree[anid]) == 1:
            loc1 = roadNetwork.nid2loc[anid]
            loc2 = roadNetwork.nid2loc[roadNetwork.node_degree[anid][0]]

            dlat = loc1[0] - loc2[0]
            dlon = (loc1[1] - loc2[1])  # * math.cos(math.radians(loc1[0]/111111.0))
            l = np.sqrt(dlat * dlat + dlon * dlon)
            dlat /= l
            dlon /= l

            heading_vector_lat = dlat
            heading_vector_lon = dlon
        elif len(roadNetwork.node_degree[anid]) == 2:
            loc1 = roadNetwork.nid2loc[roadNetwork.node_degree[anid][1]]
            loc2 = roadNetwork.nid2loc[roadNetwork.node_degree[anid][0]]

            dlat = loc1[0] - loc2[0]
            # dlon = (loc1[1] - loc2[1]) * math.cos(math.radians(loc1[0]/111111.0))
            dlon = loc1[1] - loc2[1]

            l = np.sqrt(dlat * dlat + dlon * dlon)

            dlat /= l
            dlon /= l

            heading_vector_lat = dlat
            heading_vector_lon = dlon

        annotation[anid]['heading_vector'] = (heading_vector_lat, heading_vector_lon)
        annotation[anid]['degree'] = len(roadNetwork.node_degree[anid])
    pickle.dump(annotation, open(roadnetwork_root + "annotation_osm_" + config['name'] + '.p', "w"))


def get_width(p, mask, ymax, xmax):
    rect = np.zeros((WIDTH_MAX * 2, WIDTH_MAX * 2), dtype='uint8')
    viz = np.zeros((WIDTH_MAX * 2, WIDTH_MAX * 2, 3), dtype='uint8')
    y = int(p[1])
    x = int(p[0])

    startx = 0
    starty = 0
    endx = WIDTH_MAX * 2 - 1
    endy = WIDTH_MAX * 2 - 1

    if y < WIDTH_MAX:
        starty = WIDTH_MAX - y
        endy = WIDTH_MAX * 2 - 1
    if y > ymax - WIDTH_MAX:
        starty = 0
        endy = WIDTH_MAX + ymax - y - 1
    if x < WIDTH_MAX:
        startx = WIDTH_MAX - x
        endx = WIDTH_MAX * 2 - 1
    if x > xmax - WIDTH_MAX:
        startx = 0
        endx = WIDTH_MAX + xmax - x - 1

    sx = max(x - WIDTH_MAX, 0)
    ex = min(x + WIDTH_MAX - 1, xmax - 1)
    sy = max(y - WIDTH_MAX, 0)
    ey = min(y + WIDTH_MAX - 1, ymax - 1)
    rect[starty:endy, startx:endx] = mask[sy:ey, sx:ex]  # 裁减出mask

    viz[starty:endy, startx:endx, 0] = mask[sy:ey, sx:ex]
    viz[starty:endy, startx:endx, 1] = mask[sy:ey, sx:ex]
    viz[starty:endy, startx:endx, 2] = mask[sy:ey, sx:ex]


    minn = 0
    maxx = WIDTH_MAX

    rr = 0
    while True:
        # allblack=999

        mid = (maxx + minn) // 2
        r = mid
        if r == 0:
            rr = mid
            break
        x = np.zeros((WIDTH_MAX * 2, WIDTH_MAX * 2, 1), dtype='uint8')
        # cv2.circle(x,(160,160),r,(255,255,255),-1)
        cv2.circle(x, (WIDTH_MAX, WIDTH_MAX), r, (255, 255, 255), -1)

        allblack = 999
        for i in range(WIDTH_MAX * 2):
            for j in range(WIDTH_MAX * 2):
                if x[i, j] == 255:
                    if rect[i, j] == 255:
                        continue
                    else:
                        allblack = 0
                else:
                    continue

        if allblack == 0:
            minn = 0
            maxx = mid
        else:
            rr = mid
            break

    for r in xrange(rr, WIDTH_MAX):
        if r == WIDTH_MAX:
            return r
        x = np.zeros((WIDTH_MAX * 2, WIDTH_MAX * 2, 1), dtype='uint8')
        # cv2.circle(x,(160,160),r,(255,255,255),-1)
        cv2.circle(x, (WIDTH_MAX, WIDTH_MAX), r, (255, 255, 255))

        for i in range(WIDTH_MAX * 2):
            for j in range(WIDTH_MAX * 2):
                if x[i, j] == 255:
                    if rect[i, j] == 255:
                        continue
                    else:
                        return r
                else:
                    continue


def get_point2(x, y, d, dlat, dlon):
    x_try = x + d * dlon
    y_try = y + d * dlat
    x_try = int(x_try)
    y_try = int(y_try)
    return x_try, y_try


def get_point(x, y, d, k):
    d = int(d)
    if k == None:
        x_try = x
        y_try = y - d
    else:
        a = math.atan(k)
        if (abs(k)) < 1:
            x_try = int(x + d * math.cos(a))
            y_try = int(y + d * math.sin(a))
        else:
            # y_try = int(point.y + d)
            # x_try = int(point.x + d / k)
            x_try = int(x + d * math.cos(a))
            y_try = int(y + d * math.sin(a))
    return x_try, y_try


def dealwith_T_junction(name, angle_threshold):
    THRESHOLD = angle_threshold
    annotation = pickle.load(open(roadnetwork_root + 'annotation_osm_' + name + '.p', "r"))
    roadNetwork = pickle.load(open(roadnetwork_root + "roadnetwork_" + name + '.p', "rw"))
    img = cv2.imread(viz_root + name + '_viz.png')
    img3 = np.pad(img, ((512, 512), (512, 512), (0, 0)), 'constant')

    for kk, v in annotation.iteritems():
        if annotation[kk]['degree'] == 3:
            false_T = False
            for nid in roadNetwork.node_degree[kk]:
                # num_nei=roadNetwork.node_degree[nid]
                if len(roadNetwork.node_degree[nid]) == 1:
                    false_T = True
                    break
            if false_T:
                continue
            loc = roadNetwork.nid2loc[kk]
            yy = loc[0] + 512
            xx = loc[1] + 512
            clip = img3[yy - 255:yy + 256, xx - 255:xx + 256, :]
            cv2.imwrite(viz_root + '/test/clip_' + name + '_' + str(kk) + '.png', clip)

            neighbor_angle = []
            for nid in roadNetwork.node_degree[kk]:
                v = annotation[nid]
                heading_vector_lat, heading_vector_lon = v['heading_vector']
                if heading_vector_lon * heading_vector_lon + heading_vector_lat * heading_vector_lat < 0.1:
                    angle = 0.0
                else:
                    angle = math.degrees(math.atan2(heading_vector_lat, heading_vector_lon))
                if math.isnan(angle):
                    angle = 0.0
                angle = (angle + 360) % 360
                if angle >= 180:
                    angle = (angle + 180) % 180
                neighbor_angle.append(angle)
            if abs(neighbor_angle[0] - neighbor_angle[1]) <= THRESHOLD:
                if abs(neighbor_angle[1] - neighbor_angle[2]) > THRESHOLD and abs(
                        neighbor_angle[0] - neighbor_angle[2]) > THRESHOLD:
                    p1 = roadNetwork.node_degree[kk][0]
                    p2 = roadNetwork.node_degree[kk][1]
                    loc1 = roadNetwork.nid2loc[p1]
                    loc2 = roadNetwork.nid2loc[p2]
                    y = (loc1[0] + loc2[0]) / 2
                    x = (loc1[1] + loc2[1]) / 2
                    roadNetwork.nid2loc[kk] = [y, x]
                    cv2.circle(img, (x, y), 4, (0, 255, 128), -1)
                    cv2.line(img, (loc1[1], loc1[0]), (x, y), (121, 255, 0))
                    cv2.line(img, (x, y), (loc2[1], loc2[0]), (121, 255, 0))



            elif abs(neighbor_angle[0] - neighbor_angle[2]) <= THRESHOLD:
                if abs(neighbor_angle[1] - neighbor_angle[2]) > THRESHOLD and abs(
                        neighbor_angle[0] - neighbor_angle[1]) > THRESHOLD:
                    p1 = roadNetwork.node_degree[kk][0]
                    p2 = roadNetwork.node_degree[kk][2]
                    loc1 = roadNetwork.nid2loc[p1]
                    loc2 = roadNetwork.nid2loc[p2]
                    y = (loc1[0] + loc2[0]) / 2
                    x = (loc1[1] + loc2[1]) / 2
                    roadNetwork.nid2loc[kk] = [y, x]
                    cv2.circle(img, (x, y), 4, (0, 255, 128), -1)
                    cv2.line(img, (loc1[1], loc1[0]), (x, y), (121, 255, 0))
                    cv2.line(img, (x, y), (loc2[1], loc2[0]), (121, 255, 0))


            elif abs(neighbor_angle[1] - neighbor_angle[2]) <= THRESHOLD:
                if abs(neighbor_angle[1] - neighbor_angle[0]) > THRESHOLD and abs(
                        neighbor_angle[0] - neighbor_angle[2]) > THRESHOLD:
                    p1 = roadNetwork.node_degree[kk][1]
                    p2 = roadNetwork.node_degree[kk][2]
                    loc1 = roadNetwork.nid2loc[p1]
                    loc2 = roadNetwork.nid2loc[p2]
                    y = (loc1[0] + loc2[0]) / 2
                    x = (loc1[1] + loc2[1]) / 2
                    roadNetwork.nid2loc[kk] = [y, x]
                    cv2.circle(img, (x, y), 4, (0, 255, 128), -1)
                    cv2.line(img, (loc1[1], loc1[0]), (x, y), (121, 255, 0))
                    cv2.line(img, (x, y), (loc2[1], loc2[0]), (121, 255, 0))

            else:
                print('not T junction')

    # cv2.imwrite(viz_root+'/test/vizT_'+name+'_.png',img)
    roadNetwork.DumpToFile(roadnetwork_root + "roadnetwork_" + name + '.p')


def generate_per_node_image(name, output_name="tiles", scale=1.0):
    config = json.load(open(config_root + "config_" + name + '.json', "r"))
    image = cv2.imread(config['image_dir']).astype(np.uint8)
    centerline = cv2.imread(config['centerline_dir']).astype(np.uint8)
    # segmentation=scipy.ndimage.imread(config['segmentation_dir'],mode='P').astype(np.uint8)
    segmentation = cv2.imread(config['segmentation_dir'], 0).astype(np.uint8)
    ymax = segmentation.shape[0] + 1024  # height
    xmax = segmentation.shape[1] + 1024  # width
    # canny=scipy.ndimage.imread(config['canny_dir']).astype(np.uint8)?
    annotation = pickle.load(open(roadnetwork_root + 'annotation_osm_' + name + '.p', "rw"))
    roadNetwork = pickle.load(open(roadnetwork_root + "roadnetwork_" + name + '.p', "r"))

    img2 = cv2.imread('Cheng_resize/viz/数据集处理/' + name + '_viz.png')

    # if os.path.isdir(output_folder +'tiles/'+output_name+'_'+name+"/")==True:
    #     pass
    # else:

    Popen("mkdir -p %s" % (output_folder + 'tiles/' + output_name + '_' + name + "/"), shell=True).wait()

    # pad 512
    image = np.pad(image, ((512, 512), (512, 512), (0, 0)), 'constant')
    centerline = np.pad(centerline, ((512, 512), (512, 512), (0, 0)), 'constant')
    segmentation = np.pad(segmentation, ((512, 512), (512, 512)), 'constant')
    segmentation[segmentation >= 125] = 255
    segmentation[segmentation <= 125] = 0
    # canny = np.pad(canny, ((512,512), (512,512)), 'constant')

    # ---------------------------------------------------------------------------calculate d1,d2
    max_d1 = 0
    max_d2 = 0
    for kk, v in annotation.iteritems():
        loc = roadNetwork.nid2loc[kk]
        y = loc[0]
        x = loc[1]
        y += 512
        x += 512

        heading_vector_lat, heading_vector_lon = v['heading_vector']

        if heading_vector_lon * heading_vector_lon + heading_vector_lat * heading_vector_lat < 0.1:
            angle = 0.0
            annotation[kk]['d1'] = 0
            annotation[kk]['d2'] = 0
            annotation[kk]['angle'] = 0
            annotation[kk]['heading_vector_vertical'] = (0, 0)
            annotation[kk]['angle_vertical'] = 0
            continue
        else:
            angle = math.degrees(math.atan2(heading_vector_lat, heading_vector_lon))

        if math.isnan(angle):
            angle = 0.0

        d1 = 0
        d2 = 0

        angle = (angle + 360) % 360
        if angle >= 180:
            angle = (angle + 180) % 180
            heading_vector_lat, heading_vector_lon = -heading_vector_lat, -heading_vector_lon

        heading_vector_lat_vertical = heading_vector_lon
        heading_vector_lon_vertical = -heading_vector_lat

        if angle == 0:
            angle3 = angle + 90
            k3 = None
        else:
            angle3 = angle + 90
            k3 = math.tan(math.radians(angle3))
        angle2 = math.degrees(math.atan2(heading_vector_lat_vertical, heading_vector_lon_vertical))
        angle2 = (angle2 + 360) % 360

        if angle2 == 90:
            k2 = None
        else:
            k2 = math.tan(math.radians(angle2))
        # print(angle2,k)
        print(angle2, angle3, k2, k3)
        for d in range(1, 512, 1):
            x1, y1 = get_point2(x, y, d, heading_vector_lat_vertical, heading_vector_lon_vertical)
            if segmentation[y1, x1] == 0:
                d1 = d - 1
                break
        for d in range(1, 512, 1):
            x2, y2 = get_point2(x, y, -d, heading_vector_lat_vertical, heading_vector_lon_vertical)
            if segmentation[y2, x2] == 0:
                d2 = -d + 1
                # cv2.circle(img3,(x2-512,y2-512),3,(0,125,255),1)
                break

        if d1 > max_d1:
            max_d1 = d1
        if d2 < max_d2:
            max_d2 = d2

        annotation[kk]['d1'] = d1
        annotation[kk]['d2'] = d2
        annotation[kk]['angle'] = angle
        # x1,y1=get_point(x,y,d1,k2)
        x1, y1 = get_point2(x, y, d1, heading_vector_lat_vertical, heading_vector_lon_vertical)
        # yy=y-heading_vector_lat_vertical*d1
        # xx=x-heading_vector_lon_vertical*d1
        annotation[kk]['p1'] = (x1 - 512, y1 - 512)
        # x2,y2=get_point(x,y,d2,k2)
        x2, y2 = get_point2(x, y, d2, heading_vector_lat_vertical, heading_vector_lon_vertical)
        annotation[kk]['p2'] = (x2 - 512, y2 - 512)
        annotation[kk]['heading_vector_vertical'] = (heading_vector_lat_vertical, heading_vector_lon_vertical)
        annotation[kk]['angle_vertical'] = angle2
        annotation[kk]['heading_vector'] = (heading_vector_lat, heading_vector_lon)

    print('max d1 = %i' % max_d1)
    print('max d2 = %i' % max_d2)

    visitedNodes = []
    intersectionNodes = {}
    road_segment = {}
    rsnum = 0

    for node, degree in roadNetwork.node_degree.items():
        if node in visitedNodes:
            continue
        degree = len(degree)
        if degree != 2:
            intersectionNodes[node] = 1

    node_degree = {}
    for node in intersectionNodes.keys():
        if intersectionNodes[node] == 1:
            node_degree[node] = []

    loc2node = {}
    for node, degree in roadNetwork.node_degree.items():
        ##step1 find a intersection point  #step2 find this point's neighbor---until the end of the whole roadsegment  #get the mid point and save them in result[]
        if node in visitedNodes:
            continue
        cur_node = node
        next_nodes = {}

        for nn in degree:
            next_nodes[nn] = 1

        if len(next_nodes.keys()) == 2:
            continue

        for nextnode in next_nodes.keys():
            if nextnode in visitedNodes and nextnode not in intersectionNodes:
                continue
            if nextnode in node_degree[node]:
                continue

            node_list = [node]
            cur_node = nextnode
            while True:
                node_list.append(cur_node)

                neighbor = {}
                for nn in roadNetwork.node_degree[cur_node]:
                    neighbor[nn] = 1

                if len(neighbor.keys()) != 2:
                    node_degree[cur_node].append(node_list[-2])
                    break

                if node_list[-2] == neighbor.keys()[0]:
                    cur_node = neighbor.keys()[1]
                else:
                    cur_node = neighbor.keys()[0]

            for i in range(1, len(node_list) - 1):  # dont record the first and the last one
                visitedNodes.append(node_list[i])

            road_segment[rsnum] = node_list
            rsnum += 1

    for _, list in road_segment.items():
        d1_list = []
        d2_list = []
        for node in list:
            d1 = annotation[node]['d1']
            d2 = annotation[node]['d2']
            dd1 = 0
            dd2 = 0
            if d1 > WIDTH_MAX:
                loc = roadNetwork.nid2loc[node]
                y = loc[0] + 512
                x = loc[1] + 512
                point = [x, y]
                dd1 = get_width(point, segmentation, ymax, xmax)
                annotation[node]['d1'] = int(dd1)
                d1 = dd1

            if abs(d2) > WIDTH_MAX:
                if dd1 != 0:
                    dd2 = dd1
                else:
                    loc = roadNetwork.nid2loc[node]
                    y = loc[0] + 512
                    x = loc[1] + 512
                    point = [x, y]
                    dd2 = get_width(point, segmentation, ymax, xmax)
                annotation[node]['d2'] = int(-dd2)
                d2 = -dd2
            d1_list.append(d1)
            d2_list.append(d2)
        d1_mid = np.median(d1_list)
        d2_mid = np.median(d2_list)
        for node in list:
            d1 = annotation[node]['d1']
            d2 = annotation[node]['d2']
            dd1 = 0
            if abs(d1 - d1_mid) > T:
                loc = roadNetwork.nid2loc[node]
                y = loc[0] + 512
                x = loc[1] + 512
                point = [x, y]
                dd1 = get_width(point, segmentation, ymax, xmax)
                annotation[node]['d1'] = int(dd1)
            if abs(d2 - d2_mid) > T:
                if dd1 != 0:
                    dd2 = dd1
                else:
                    loc = roadNetwork.nid2loc[node]
                    y = loc[0] + 512
                    x = loc[1] + 512
                    point = [x, y]
                    dd2 = get_width(point, segmentation, ymax, xmax)
                annotation[node]['d2'] = int(-dd2)

    pickle.dump(annotation, open(roadnetwork_root + "annotation_osm_" + config['name'] + '.p', "w"))

    # --------------------------------------------------------cut image

    for kk, v in annotation.iteritems():
        loc = roadNetwork.nid2loc[kk]
        y = loc[0]
        x = loc[1]
        y += 512
        x += 512

        heading_vector_lat_vertical, heading_vector_lon_vertical = v['heading_vector_vertical']
        # heading_vector_lat_vertical, heading_vector_lon_vertical = v['heading_vector']
        # d1=v['d1']
        # d2=v['d2']
        rr = int(272 * scale)
        subimage = image[y - rr:y + rr, x - rr:x + rr]
        subcenter = centerline[y - rr:y + rr, x - rr:x + rr]
        subseg = segmentation[y - rr:y + rr, x - rr:x + rr]
        # yy=int(loc[0]/float(4))+512
        # xx=int(loc[1]/float(4))+512
        # subcanny=canny[yy-rr:yy+rr,xx-rr:xx+rr]
        subimage = scipy.misc.imresize(subimage, (272 * 2, 272 * 2)).astype(np.uint8)
        subcenter = scipy.misc.imresize(subcenter, (272 * 2, 272 * 2)).astype(np.uint8)
        subseg = scipy.misc.imresize(subseg, (272 * 2, 272 * 2)).astype(np.uint8)
        # subcanny=scipy.misc.imresize(subcanny, (272*2, 272*2)).astype(np.uint8)

        if heading_vector_lon_vertical * heading_vector_lon_vertical + heading_vector_lat_vertical * heading_vector_lat_vertical < 0.1:
            angle = 0.0
        else:
            angle = math.degrees(math.atan2(heading_vector_lat_vertical, heading_vector_lon_vertical))

        if math.isnan(angle):
            angle = 0.0

        print(kk, angle)
        img = scipy.ndimage.interpolation.rotate(subimage, angle)
        cl = scipy.ndimage.interpolation.rotate(subcenter, angle)
        seg = scipy.ndimage.interpolation.rotate(subseg, angle)
        # can=scipy.ndimage.interpolation.rotate(subcanny,angle)
        center = np.shape(cl)[0] / 2

        r = 128  # 128
        result = img[center - r:center + r, center - r: center + r, :]
        result2 = cl[center - r:center + r, center - r: center + r, :]
        result3 = seg[center - r:center + r, center - r: center + r]
        # result4 = can[center-r:center+r, center-r: center+ r]

        Image.fromarray(result).save(output_folder + 'tiles/' + output_name + '_' + name + "/" + "img_%d.png" % kk)
        Image.fromarray(result2).save(output_folder + 'tiles/' + output_name + '_' + name + "/" + "cl_%d.png" % kk)
        Image.fromarray(result3).save(output_folder + 'tiles/' + output_name + '_' + name + "/" + "seg_%d.png" % kk)
        # Image.fromarray(result4).save(output_folder +'tiles/'+output_name+'_'+name+"/"+"canny_%d.png" % k)


for name in name_list:
    # step 1 : image config
    generate_config_file(name)

    # step2  : Roadnetwork
    generate_roadnetwork(name)

    # step3  : annatation(width,heading_vector)
    annotate_dataset_osm(name)

    # step3.5  : deal with T-junctions
    # dealwith_T_junction(name,angle_threshold=45)
    # #
    # annotate_dataset_osm(name)

    # step4  : cut each node's image 128*128
    generate_per_node_image(name)
