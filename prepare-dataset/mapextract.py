# -*- coding: UTF-8 -*-
import sys

import rdp
import scipy.ndimage
import skimage.morphology
import os
from PIL import Image
import math
import numpy
import argparse
from multiprocessing import Pool
import subprocess

#step2：centerline --> .graph file

def extract(in_fname, threshold, out_fname, mergin=32):
    im = scipy.ndimage.imread(in_fname, flatten=True)
    end1 = im.shape[0] - mergin
    end2 = im.shape[1] - mergin
    # im2=numpy.zeros([end1-mergin,end2-mergin,3],dtype='uint8')
    im2 = im[mergin:end1, mergin:end2]
    im = numpy.swapaxes(im2, 0, 1)
    im = im > threshold

    # apply morphological dilation and thinning
    selem = skimage.morphology.disk(2)
    im = skimage.morphology.binary_dilation(im, selem)
    im = skimage.morphology.thin(im)
    im = im.astype('uint8')

    # extract a graph by placing vertices every THRESHOLD pixels, and at all intersections
    vertices = []
    edges = set()

    def add_edge(src, dst):
        if (src, dst) in edges or (dst, src) in edges:
            return
        elif src == dst:
            return
        edges.add((src, dst))

    point_to_neighbors = {}
    q = []
    while True:
        if len(q) > 0:
            lastid, i, j = q.pop()
            path = [vertices[lastid], (i, j)]
            if im[i, j] == 0:
                continue
            point_to_neighbors[(i, j)].remove(lastid)
            if len(point_to_neighbors[(i, j)]) == 0:
                del point_to_neighbors[(i, j)]
        else:
            w = numpy.where(im > 0)
            if len(w[0]) == 0:
                break
            i, j = w[0][0], w[1][0]
            lastid = len(vertices)
            vertices.append((i, j))
            path = [(i, j)]

        while True:
            im[i, j] = 0
            neighbors = []
            for oi in [-1, 0, 1]:
                for oj in [-1, 0, 1]:
                    ni = i + oi
                    nj = j + oj
                    if ni >= 0 and ni < im.shape[0] and nj >= 0 and nj < im.shape[1] and im[ni, nj] > 0:
                        neighbors.append((ni, nj))
            if len(neighbors) == 1 and (i, j) not in point_to_neighbors:
                ni, nj = neighbors[0]
                path.append((ni, nj))
                i, j = ni, nj
            else:
                if len(path) > 1:
                    path = rdp.rdp(path, 2)
                    if len(path) > 2:
                        for point in path[1:-1]:
                            curid = len(vertices)
                            vertices.append(point)
                            add_edge(lastid, curid)
                            lastid = curid
                    neighbor_count = len(neighbors) + len(point_to_neighbors.get((i, j), []))
                    if neighbor_count == 0 or neighbor_count >= 2:
                        curid = len(vertices)
                        vertices.append(path[-1])
                        add_edge(lastid, curid)
                        lastid = curid
                for ni, nj in neighbors:
                    if (ni, nj) not in point_to_neighbors:
                        point_to_neighbors[(ni, nj)] = set()
                    point_to_neighbors[(ni, nj)].add(lastid)
                    q.append((lastid, ni, nj))
                for neighborid in point_to_neighbors.get((i, j), []):
                    add_edge(neighborid, lastid)
                break

    with open(out_fname, 'w') as f:
        for vertex in vertices:
            f.write('{} {}\n'.format(vertex[0] + mergin, vertex[1] + mergin))
        f.write('\n')
        for edge in edges:
            f.write('{} {}\n'.format(edge[0], edge[1]))
            f.write('{} {}\n'.format(edge[1], edge[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_root', type=str, default='/home/yanjingjing/data/temp/graph/')
    parser.add_argument('--skeleton_root', type=str, default='/home/yanjingjing/data/temp/cen/')
    args = parser.parse_args()

    region_root = args.skeleton_root
    graph_root = args.graph_root
    os.mkdir(graph_root)
    imagelist = list(os.listdir(region_root))
    region_list = list(map(lambda x: x[:-4], imagelist))
    for region in region_list:
        in_fname = region_root + region + ".png"
        out_fname = graph_root + region + ".graph"
        extract(in_fname, 128, out_fname, mergin=2)
        print(region + " OK!")
