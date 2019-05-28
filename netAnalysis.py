import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from datetime import date
import seaborn as sns


class netAnalysis():
    def __init__(self, name, G):
        self.name = name
        self.G = G
        self.__flowCentrality = {}
        _, self.__mfmc, self.__edgeSlack = self.maxFlowMinCost(self.G)
        self.mfv = nx.maximum_flow_value(self.G, _s='s', _t='t', capacity="capacity")
        self.__edge_betweeness = self.__edgeBetweennessFunc(self.G)
        self.percolation1 = {}
        self.percolation1['flow'], self.percolation1['mf'], self.percolation1['dict'] = self.__faultScen_single_edge()
        self.percolation2 = {}
        self.percolation2['flow'], self.percolation2['mf'], self.percolation2['dict'] = self.__falutScen_two_edge()
        self.percolation2['table'] = pd.DataFrame.from_dict(pd.Series(self.percolation2['mf'])).unstack()[0]
        # self.percolation2['table'].to_csv("twoc.csv")

        self.all_residuals()
        self.edgeStatistics = self.__statisticsAllTogether()

    def maxFlowMinCost(self, diNet):
        ''' graph must have one node s and one node t '''
        w = {}
        flow = {}
        residual_capacity = {}
        mfmc = nx.max_flow_min_cost(diNet, s='s', t='t', capacity='capacity', weight='weight')
        for k, v in mfmc.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    w[(k, k2)] = (diNet[k][k2]['capacity'], v2)
                    flow[(k, k2)] = v2
                    residual_capacity[(k, k2)] = diNet.get_edge_data(k, k2)['capacity'] - v2
            else:
                print("oops")
        return w, flow, residual_capacity

    def __faultScen_single_edge(self):
        edgeFaultEffectonFlow = {}
        faultyEdgeFlow = {}
        for e in self.G.edges:
            g = self.G.copy()
            g[e[0]][e[1]]['capacity'] = 0
            _, edgeFaultEffectonFlow[e], _ = self.maxFlowMinCost(g)

            faultyEdgeFlow[e] = nx.maximum_flow_value(g, _s='s', _t='t')
        faultDF = pd.DataFrame.from_dict(edgeFaultEffectonFlow)
        faultDF.set_index(faultDF.index.values, inplace=True)
        faultDF.columns = faultDF.columns.values

        return faultDF, faultyEdgeFlow, edgeFaultEffectonFlow

    def __faultScen_multi_edge(self, edgeGroupList):
        edgeFaultEffectonFlow = {}
        faultyEdgeFlow = {}
        for edgeGroup in edgeGroupList:
            g = self.G.copy()
            for e in edgeGroup:
                g[e[0]][e[1]]['capacity'] = 0
            _, edgeFaultEffectonFlow[edgeGroup], _ = self.maxFlowMinCost(g)

            faultyEdgeFlow[edgeGroup] = nx.maximum_flow_value(g, _s='s', _t='t')
        faultDF = pd.DataFrame.from_dict(edgeFaultEffectonFlow)
        faultDF.set_index(faultDF.index.values, inplace=True)
        faultDF.columns = faultDF.columns.values

        return faultDF, faultyEdgeFlow, edgeFaultEffectonFlow

    def __falutScen_two_edge(self):
        s = itertools.combinations(self.G.edges, 2)
        return self.__faultScen_multi_edge(s)

    def __edgeBetweennessFunc(self, g):
        edge_betweeness = nx.edge_betweenness_centrality(g)
        edge_betweeness = dict(zip(edge_betweeness.keys(), [round(v, 3) for v in edge_betweeness.values()]))
        return edge_betweeness

    def plot_networkChart(self, figsize=(10, 10), saveToFile=True):
        plt.figure(figsize=figsize)
        e_labels = self.__edgeLabelGen(self.__mfmc, self.__edge_betweeness, self.mfv)
        nx.draw(self.G, pos=nx.get_node_attributes(self.G, 'pos'), with_labels=True, node_size=200, font_size=12)
        nx.draw_networkx_edge_labels(self.G, pos=nx.get_node_attributes(self.G, 'pos'), edge_labels=e_labels)
        if saveToFile:
            plt.savefig(self.name + ".jpg")
        plt.show()

    def __edgeLabelGen(self, mfmc, ebc, mfv):
        e_labels = {}
        for e, f in mfmc.items():
            e_labels[e] = str(tuple([self.G[e[0]][e[1]]["capacity"], f]))
            e_labels[e] += "-" + str(mfv - self.percolation1['mf'][e])
            e_labels[e] += "\n" + str(round(ebc[e], 3))
        return e_labels

    def __successorEdges(self, e, g):
        edges = []
        if len(list(g.successors(e[1]))) < 1:
            edges.append(e)
        else:
            for w in g.successors(e[1]):
                edges += self.__successorEdges((e[1], w), g) + [e]
        return edges

    def __predecessorEdges(self, e, g):
        edges = []
        if len(list(g.predecessors(e[0]))) < 1:
            edges.append(e)
        else:
            for w in g.predecessors(e[0]):
                edges += self.__predecessorEdges((w, e[0]), g) + [e]
        return edges

    def __residual_network_passing_edge(self, e):
        rnpe = self.__successorEdges(e, self.G)
        rnpe += self.__predecessorEdges(e, self.G)
        return set(rnpe)

    def all_residuals(self):
        ar = {}
        for e in self.G.edges:
            temp = self.__residual_network_passing_edge(e)
            ar[e] = self.__resGraphs(temp)
            x = nx.maximum_flow(ar[e], _s='s', _t='t')
            self.__flowCentrality[e] = x[0] / self.mfv
        return ar

    def __resGraphs(self, residualEdges):
        nds = []
        for e in residualEdges:
            nds += list(e)
        g2 = self.G.subgraph(set(nds))
        assert nx.is_frozen(g2)
        # To “unfreeze” a graph you must make a copy by creating a new graph object:
        g2 = nx.DiGraph(g2)
        edges_should_be_removed = []
        for e in g2.edges:
            if e not in residualEdges:
                edges_should_be_removed.append(e)
        g2.remove_edges_from(edges_should_be_removed)
        return g2

    def __forwardPass(self, v, minf, g, residualFlow):
        nodeInPath = []

        if len(list(g.successors(v))) < 1:
            nodeInPath.append(v)
        else:
            flowNotPassedForwardYet = True
            for w in self.G.successors(v):
                if flowNotPassedForwardYet:
                    if residualFlow[(v, w)] >= minf:
                        residualFlow[(v, w)] -= minf
                        flowNotPassedForwardYet = False

                        nodeInPath += [v] + self.__forwardPass(w, minf, g, residualFlow)

        return nodeInPath

    def __backwardPass(self, v, minf, g, residualFlow):
        nodeInFlowPath = []

        if len(list(g.predecessors(v))) < 1:
            nodeInFlowPath.append(v)
        else:
            flowNotPassedForwardYet = True
            for w in self.G.predecessors(v):
                if flowNotPassedForwardYet:
                    if residualFlow[(w, v)] >= minf:
                        residualFlow[(w, v)] -= minf
                        flowNotPassedForwardYet = False

                        nodeInFlowPath += self.__backwardPass(w, minf, g, residualFlow) + [v]
        return nodeInFlowPath

    def __minNonZeroDictKey(self, d):
        # {x: y for x, y in net1.mfmc.items() if y != 0}
        min_val = None
        result = None
        for k, v in d.items():
            if v and (min_val is None or v < min_val):
                min_val = v
                result = k
        return result

    def extractFlowStreams(self):
        residualFlow = self.__mfmc.copy()
        i = 0
        k = self.__minNonZeroDictKey(residualFlow)
        pathsAndFlows = {}
        while k:
            i = i + 1
            minf = residualFlow[k]
            residualFlow[k] = 0
            path1 = self.__backwardPass(k[0], minf, self.G, residualFlow) \
                    + self.__forwardPass(k[1], minf, self.G, residualFlow)
            pathsAndFlows[(*k, minf)] = path1
            k = self.__minNonZeroDictKey(residualFlow)
        functionalConnectivity = pd.DataFrame(columns=residualFlow.keys(), index=residualFlow.keys())
        # functionalConnectivity.fillna(0,inplace=True)
        for k, v in pathsAndFlows.items():
            ix = (k[0], k[1])
            for i, j in enumerate(v[:-1]):
                functionalConnectivity.loc[ix, (j, v[i + 1])] = k[2]
        return pathsAndFlows, functionalConnectivity

    def __statisticsAllTogether(self):
        cap = self.G.edges.data('capacity')
        capac = {}
        for k in cap:
            capac[(k[0], k[1])] = k[2]
        capac

        df = pd.DataFrame.from_dict({"capacity": pd.Series(capac),
                                     "flow": self.__mfmc,
                                     "slakc": self.__edgeSlack,
                                     "betweenness": pd.Series(self.__edge_betweeness),
                                     "flowCentrality": pd.Series(self.__flowCentrality)
                                     })
        return df

    def plot_statistics(self):
        ax1 = plt.subplot(211)
        self.edgeStatistics.iloc[:, :2].plot(ax=ax1, style=["--", '-'], color='k', lw=2)
        ax1.set_ylabel('Utits', color='k')
        tix = [i for i, x in enumerate(self.edgeStatistics.index.values)]
        plt.fill_between(tix, self.edgeStatistics.iloc[:, 0], self.edgeStatistics.iloc[:, 1], color='b', alpha=0.2)

        ax2 = plt.subplot(212)  # instantiate a second axes that shares the same x-axis

        self.edgeStatistics.iloc[:, 3:].plot(ax=ax2, color='k', style=[':', "-."])
        ax2.set_ylabel('Centrality', color='k')

        ax2.set_xticks(range(len(self.G.edges)))
        ax2.set_xticklabels(self.edgeStatistics.index.values, rotation=45)
        # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=70)

        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.set_xticklabels([])
        plt.show()

    @classmethod
    def from_GML_file(cls, filename):
        g = nx.read_gml(filename)
        return netAnalysis(filename, g)

    @classmethod
    def import_graph_from_excel(cls, sheetName, header=0, nodepos=False):
        df = read_data(sheetName, header, nodepos)
        g = createGraph(df['nodes'], df['edges'], nodepos)
        return cls(sheetName, g)

    def maxFlow_gurobi(self):
        cap = {}
        wei = {}
        g = self.G
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
        m.setObjective(x.sum('s', '*'), gu.GRB.MAXIMIZE)
        # m.write("model.lp")
        m.optimize()
        maxFlowVal = m.objVal
        m.addConstr(x.sum('*', 't') == maxFlowVal)
        m.setObjective(gu.quicksum(wei[e[0], e[1]] * x[e[0], e[1]] for e in edgeIdx), gu.GRB.MAXIMIZE)
        # y[0] = quicksum(x.select(1, '*'))
        # y[1] = quicksum(x.select(8, '*'))
        #
        # m.setObjectiveN( | y[0] - y[1] |, 1)
        m.optimize()

        xvar = pd.DataFrame(columns=['varName', 'value', 'reducedCost', 'lb', 'ub'])
        for i, v in enumerate(x):
            xvar.loc[i] = [v, x[v].x, x[v].RC, x[v].SAObjLow, x[v].SAObjUp]
        xvar.set_index('varName', inplace=True)
        duals = {}
        for c in consts1:
            duals[c] = consts1[c].PI
        self.x = xvar
        self.dual = duals
        return xvar, duals

    def doAllTheTasks(self):
        self.plot_networkChart(saveToFile=True)
        ar = self.all_residuals()
        self.plot_statistics()
        if len(self.percolation2) > 10:
            plt.figure(figsize=(20, 20))
        sns.heatmap(self.percolation2['table'])
        # sorted(net1.percolation2['mf'].items(), key=operator.itemgetter(1))[0]

        _, functionalConnectivity = self.extractFlowStreams()
        functionalConnectivity.dropna(how='all', inplace=True)
        functionalConnectivity.sum(axis=0)  # actual flow
        functionalConnectivity.count(axis=0)  # how many paths does an edge is engaged in

    def edgeToDF(self):
        df1 = pd.DataFrame.from_dict(self.G.edges(data=True))
        # new = df1[2].str.split(',',n=1, expand =True)
        df1['capacity'] = df1[2].apply(lambda x: x["capacity"])
        df1['weight'] = df1[2].apply(lambda x: x["weight"])
        df1.drop(2, axis=1, inplace=True)  # %%
        self.edgeDF = df1


def read_data(sheetName, header=0, nodepos=False):
    sourceFile = "network_flow20190516.xlsx"
    if not os.path.isfile(sourceFile):
        sourceFile = os.path.join("two_component", "network_flow20190516.xlsx")
    assert os.path.isfile(sourceFile)
    df = pd.read_excel(io=sourceFile, sheet_name=sheetName, header=header)
    if isinstance(header, list):
        if len(header) > 1:
            print("data contains two rows of header")
            nets = {}
            for i in df.columns.levels[0]:
                nets[i] = df[i].dropna()
            df = nets
    if nodepos:
        df['nodes']['pos'] = list(zip(df['nodes'].x, df['nodes'].y))
        df['nodes'].set_index(df['nodes']['node'], inplace=True)
    return df


def createGraph(nodes, edges, nodePos=False):
    nodesDF = nodes
    edgesDF = edges
    G = nx.from_pandas_edgelist(edgesDF, source='source', target='target',
                                edge_attr=True, create_using=nx.DiGraph)
    if nodePos:
        pos = nodesDF['pos'].to_dict()
    else:
        pos = nx.circular_layout(G)
    for i in G.nodes:
        G.nodes[i]['pos'] = pos[i]
    return G


# Python program to demonstrate
# use of class method and static method.


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

        # a class method to create a Person object by birth year.

    @classmethod
    def fromBirthYear(cls, name, year):
        return cls(name, date.today().year - year)

        # a static method to check if a Person is adult or not.

    @staticmethod
    def isAdult(age):
        return age > 18


if __name__ == "__main__":
    person1 = Person('mayank', 21)
    person2 = Person.fromBirthYear('mayank', 1996)

    print
    person1.age
    print
    person2.age

    # print the result
    print
    Person.isAdult(22)
