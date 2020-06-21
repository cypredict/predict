import networkx as nx
import matplotlib.pyplot as plt

#定义图的节点和边
nodes=['0','1','2','3','4','5','a','b','c']
edges=[('0','0',1),('0','1',1),('0','5',1),('0','5',2),('1','2',3),('1','4',5),('2','1',7),('2','4',6),('a','b',0.5),('b','c',0.5),('c','a',0.5)]

#定义graph
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edges)

#抽取图G的节点作为子图
sub_graph = G.subgraph(['0', '1', '2'])
#pos=nx.spring_layout(sub_graph, k=1)
pos=nx.spring_layout(G, k=1)
nx.draw(sub_graph, pos, with_labels=True, node_size=30, font_size=10)
plt.show()
nx.draw(G, pos, with_labels=True, node_size=30, font_size=10)
plt.show()
