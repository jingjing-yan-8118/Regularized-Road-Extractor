# -*- coding: UTF-8 -*-
import numpy as np
from subprocess import Popen
import graph
import geom
import os
import argparse

# step3: resample graph nodes

parser = argparse.ArgumentParser()

parser.add_argument('--graph_root', type=str, default='/home/yanjingjing/data/temp/graph/')
parser.add_argument('--resample_root', type=str, default='/home/yanjingjing/data/temp/resample_graph/')
parser.add_argument('--density', type=int, default=50)
args = parser.parse_args()

graph_root = args.graph_root
save_root = args.resample_root
density = args.density
Popen("mkdir -p %s" % (save_root), shell=True).wait()


def distance(p1, p2):
    a = p1.x - p2.x
    # b = (p1[1] - p2[1])*math.cos(math.radians(p1[0]))
    b = p1.y - p2.y
    return np.sqrt(a * a + b * b)


if __name__ == '__main__':
    image_list = list(os.listdir(graph_root))
    name_list = list(map(lambda x: x[:-6], image_list))
    # name_list = ['329','269']
    # graphs={}
    for name in name_list:
        print(name + ' begin!')
        wrong = 0
        graph1 = graph.read_graph(graph_root + name + '.graph')

        # graphs[name]=graph1

        visitedNodes = []
        intersectionNodes = {}
        new_Graph = graph.Graph()

        for node in graph1.vertices:
            nodeid = node.id
            if nodeid in visitedNodes:
                continue
            degree = len(node.degree)
            if degree != 2:
                intersectionNodes[nodeid] = 1

        node_degree = {}
        for node in intersectionNodes.keys():
            if intersectionNodes[node] == 1:
                node_degree[node] = []

        loc2node = {}
        for node in graph1.vertices:
            ##step1 find a intersection point  #step2 find this point's neighbor---until the end of the whole roadsegment  #get the mid point and save them in result[]
            nodeid = node.id
            if nodeid in visitedNodes:
                continue
            cur_node = nodeid
            next_nodes = {}

            for nn in node.degree:
                next_nodes[nn] = 1

            if len(next_nodes.keys()) == 2:
                continue

            for nextnode in next_nodes.keys():
                if nextnode in visitedNodes and nextnode not in intersectionNodes:
                    continue
                if nextnode in node_degree[nodeid]:
                    continue

                node_list = [nodeid]
                cur_node = nextnode
                while True:
                    node_list.append(cur_node)

                    neighbor = {}
                    for nn in graph1.vertices[cur_node].degree:
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

                dists = []
                dist = 0
                for i in range(0, len(node_list) - 1):
                    dists.append(dist)
                    dist += distance(graph1.vertices[node_list[i]].point, graph1.vertices[node_list[i + 1]].point)

                dists.append(dist)

                # if dist < density/2:
                #	continue
                if dist < density:
                    if graph1.vertices[node_list[0]].point not in loc2node.keys():
                        v1 = new_Graph.add_vertex(graph1.vertices[node_list[0]].point)
                        loc2node[graph1.vertices[node_list[0]].point] = v1
                    else:
                        v1 = loc2node[graph1.vertices[node_list[0]].point]
                    for i in range(1, len(node_list)):
                        if graph1.vertices[node_list[i]].point not in loc2node.keys():
                            v2 = new_Graph.add_vertex(graph1.vertices[node_list[i]].point)
                            loc2node[graph1.vertices[node_list[i]].point] = v2
                        else:
                            v2 = loc2node[graph1.vertices[node_list[i]].point]
                        new_Graph.add_edge(v1, v2)
                        new_Graph.add_edge(v2, v1)
                        v1 = v2
                else:

                    n = max(int(dist / density), 1)

                    alphas = [float(x + 1) / float(n + 1) for x in range(n)]

                    road_segments = []

                    for alpha in [0] + alphas + [1.0]:
                        for j in range(len(node_list) - 1):

                            # Don't add starting locations in the tunnel
                            # if metaData is not None:
                            #     nnn1 = OSMMap.nodeHashReverse[node_list[j]]
                            #     nnn2 = OSMMap.nodeHashReverse[node_list[j+1]]
                            #
                            #     if metaData.edgeProperty[metaData.edge2edgeid[(nnn1,nnn2)]]['layer'] < 0:
                            #         tunnel_skip_num += 1
                            #         continue
                            #
                            #
                            #     lane = metaData.edgeProperty[metaData.edge2edgeid[(nnn1,nnn2)]]['lane']
                            #
                            #     road_info = metaData.edgeProperty[metaData.edge2edgeid[(nnn1,nnn2)]]
                            # else:
                            #     lane = -1
                            #     road_info = {}
                            # find the mid point
                            if alpha * dist >= dists[j] and alpha * dist <= dists[j + 1]:
                                a = (alpha * dist - dists[j]) / (dists[j + 1] - dists[j])
                                y = (1 - a) * graph1.vertices[node_list[j]].point.y + a * graph1.vertices[
                                    node_list[j + 1]].point.y
                                x = (1 - a) * graph1.vertices[node_list[j]].point.x + a * graph1.vertices[
                                    node_list[j + 1]].point.x

                                road_segments.append((int(x), int(y)))

                    if new_Graph is not None:
                        if geom.Point(road_segments[0][0], road_segments[0][1]) not in loc2node.keys():
                            v1 = new_Graph.add_vertex(geom.Point(road_segments[0][0], road_segments[0][1]))
                            loc2node[geom.Point(road_segments[0][0], road_segments[0][1])] = v1
                        else:
                            v1 = loc2node[geom.Point(road_segments[0][0], road_segments[0][1])]
                        for i in range(1, len(road_segments)):
                            if geom.Point(road_segments[i][0], road_segments[i][1]) not in loc2node.keys():
                                v2 = new_Graph.add_vertex(geom.Point(road_segments[i][0], road_segments[i][1]))
                                loc2node[geom.Point(road_segments[i][0], road_segments[i][1])] = v2
                            else:
                                v2 = loc2node[geom.Point(road_segments[i][0], road_segments[i][1])]
                            if v1 == v2:
                                wrong += 1
                                continue
                            new_Graph.add_edge(v1, v2)
                            new_Graph.add_edge(v2, v1)
                            v1 = v2

        inter_visited = []
        for node in new_Graph.vertices:
            if node.id in inter_visited:
                continue
            inter_list = []
            neighbor_list = []
            node_list = []
            if len(node.degree) > 2:
                inter_list.append(node.id)
                node_list.append(node.id)
                while True:
                    cur_nn = node_list.pop(0)
                    for nextid in new_Graph.vertices[cur_nn].degree:
                        if len(new_Graph.vertices[nextid].degree) > 2:
                            if nextid in inter_list:
                                continue
                            elif nextid in inter_visited:
                                continue
                            else:
                                inter_list.append(nextid)
                                node_list.append(nextid)
                        else:
                            neighbor_list.append(nextid)
                    if len(node_list) == 0:
                        break

            if len(inter_list) > 1:
                for edge in new_Graph.edges:
                    if edge.dst.id in inter_list or edge.src.id in inter_list:
                        edge.is_del = True
                xx = 0
                yy = 0
                for internode in inter_list:
                    inter_visited.append(internode)
                    x = new_Graph.vertices[internode].point.x
                    xx += x
                    y = new_Graph.vertices[internode].point.y
                    yy += y

                x_ave = xx / len(inter_list)
                y_ave = yy / len(inter_list)
                v1 = new_Graph.add_vertex(geom.Point(x_ave, y_ave))
                for neighbornode in neighbor_list:
                    for internode in inter_list:
                        new_Graph.vertices[internode].is_del = True
                        if internode in new_Graph.vertices[neighbornode].degree:
                            new_Graph.vertices[neighbornode].degree.remove(internode)
                    v2 = new_Graph.vertices[neighbornode]
                    new_Graph.add_edge(v1, v2)
                    new_Graph.add_edge(v2, v1)

        eid = 0
        for edge in new_Graph.edges:
            if edge.src.id in inter_visited or edge.dst.id in inter_visited:
                if new_Graph.edges[eid].is_del == False:
                    new_Graph.edges[eid].is_del = True
            eid += 1

        new_Graph.save(save_root + name + '.graph')
        print('wrong= ', wrong)
