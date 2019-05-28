import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import networkx as nx
from prettytable import PrettyTable

file_dir = os.path.dirname('__file__')
sys.path.append(file_dir)
from two_component.netAnalysis import netAnalysis
# from two_component.netAnalysis import import_graph_from_excel
from two_component.graphGen import randomGraphGen

# %% Giant Component
# net1 = netAnalysis.import_graph_from_excel("net1", header=[0, 1], nodepos=True)
# net1.doAllTheTasks()

# %%
net2 = netAnalysis.from_GML_file("graph1.gml")
net2.maxFlow_gurobi()
# net2.doAllTheTasks()
#%%



# %% https://pythonhosted.org/pygurobi/
def mf_gurobi():
    cap = {}
    wei = {}
    g=net2.G
    for k in g.edges(data=True):
        cap[(k[0], k[1])] = k[2]["capacity"]
        wei[(k[0], k[1])] = k[2]["weight"]

    import gurobipy as gu

    edgeIdx, capacity = gu.multidict(cap)
    m = gu.Model()
    x = m.addVars(edgeIdx, name='flow')

    consts1 = m.addConstrs((x.sum('*', v) == x.sum(v, '*') for v in set(g.nodes).difference(set(['s', 't']))),
                 name="flow_conservation")
    m.addConstrs((x[e[0], e[1]] <= capacity[e[0], e[1]] for e in edgeIdx), "capacityConst")
    m.setObjective(x.sum('s', '*') , gu.GRB.MAXIMIZE)
    # m.write("model.lp")
    m.optimize()
    maxFlowVal = m.objVal
    m.addConstr(x.sum('*','t')==maxFlowVal)
    m.setObjective(gu.quicksum(wei[e[0],e[1]]*x[e[0], e[1]]  for e in edgeIdx ),gu.GRB.MAXIMIZE)
    # y[0] = quicksum(x.select(1, '*'))
    # y[1] = quicksum(x.select(8, '*'))
    #
    # m.setObjectiveN( | y[0] - y[1] |, 1)
    m.optimize()


    xVal = {}
    for f in x:
        xVal[f] = x[f].x

    # print(m.getAttr(gu.GRB.Attr.X, m.getVars()))
    # print("number of variables = ", m.numVars)
    # v = m.getVars()[0]
    # v.ub



    # print('Variable Information Including Sensitivity Information:')
    # tVars = PrettyTable(['Variable Name', ' Value', 'ReducedCost', 'SensLow', ' SensUp'])  #column headers
    # for eachVar in m.getVars():
    #     tVars.add_row([eachVar.varName,eachVar.x,eachVar.RC,eachVar.SAObjLow,eachVar.SAObjUp])
    # print(tVars)

    # not good because it includes all the constraints and the name attrib was not retrieved
    # for c in m.getConstrs():
    #     print( c.PI)
    #
    for c in consts1:
        print(c,consts1[c].PI)
# %%
# dff = pd.DataFrame({"gu": pd.Series(xVal),
#               "nx": pd.Series(net2.edgeStatistics['flow'])})

# %% to create a dataframe of edge data run net.edgetoDF. then net.edgeDF will be available

#%%
mf = nx.maximum_flow(net2.G,_s='s',_t='t')
sum(mf[1]['s'].values())

#%%
