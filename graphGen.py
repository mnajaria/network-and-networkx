from datetime import datetime

import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
from networkx.readwrite import json_graph

def main1(m=[3, 6, 3]):
    g = nx.DiGraph()
    # g.add_node('s', x=0, y=0)
    g.add_node('s', pos=(0, 0))
    layerWidth = 5
    lo = 0

    masterCounter = 0
    layerNodeDict = {}
    for i in m:
        layerNodeDict = []
        for k in range(i):
            masterCounter += 1
            lo += layerWidth
            la = k * pow(-1, k) * 3
            previousNodes = g.nodes
            # g.add_node(masterCounter,x=lo,y=la)
            g.add_node(masterCounter, pos=(la, lo))
            randomEdges(g, masterCounter, previousNodes)
            # for nod in previousNodes:
            #     if random.random() >= 0.5:
            #         g.add_edge(nod, masterCounter)
    previousNodes = g.nodes
    g.add_node('t', pos=(lo + layerWidth, 0))
    randomEdges(g, 't', previousNodes)
    nx.draw(g, pos=nx.get_node_attributes(g, 'pos'), with_labels=True)
    plt.show()


def randomGraphGen(m=[2, 2]):
    g = nx.DiGraph()
    g.add_node('s', pos=(0, 0))
    nodeName = 0
    for i in m:
        for k in range(i):
            nodeName += 1
            randomEdges(g, nodeName)
    randomEdges(g, 't')
    for nod in g.nodes:
        if len(list(g.successors(nod))) < 1 and nod != 't':
            g.add_edge(nod, 't',weight = random.randint(1,10),capacity = random.randint(1,10))
    pos = posGen(g)
    return g, pos


def posGen(g):
    stride = 5
    layers = {0: ['s']}
    traversed = set(['s'])
    remained = set(g.node).difference(traversed)
    ll = 0
    while len(remained) > 0:
        ll += 1
        layers[ll] = []
        trav = traversed.copy()
        for n in remained.copy():
            if set(g.predecessors(n)).issubset(trav):
                layers[ll].append(n)
                traversed.add(n)
                remained.remove(n)
    pos = {}
    for k, v in layers.items():
        hh = [(h - len(v) / 2) * 5 for h in range(len(v))]
        for c, n in enumerate(v):
            pos[n] = (k * stride, hh[c])

    for n, p in pos.items():
        g.node[n]['pos'] = pos[n]
    # morePosAdjustment(g,pos)
    nx.write_gml(g,"graph"+datetime.datetime.now().strftime("%Y%m%d_%H%M")+".gml")
    return pos


def randomEdges(g, nodeName):
    previousNodes = list(g.nodes)
    for nod in previousNodes:
        if random.random() >= 0.7:
            g.add_edge(nod, nodeName, weight = random.randint(1,10), capacity = random.randint(1,10))


def morePosAdjustment(g,pos):
    nodes = list(g.nodes)
    for i in range(len(nodes) - 2):
        for j in range(i + 1, len(nodes)-1):
            if g.has_edge(nodes[i], nodes[j]):
                if pos[nodes[i]][1] == pos[nodes[j]][1]:
                    la, lo = pos[nodes[j]]
                    print(nodes[i],nodes[j], la,lo)
                    pos[j] = (la, lo  - 0.02)
# %%
# def main():


if __name__ == '__main__':
    g,pos = randomGraphGen([3, 6,7, 5])

    # nx.set_node_attributes(g, pos )

    nx.draw(g, pos=pos, with_labels=True)
    plt.show()
