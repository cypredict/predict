import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt

# 数据加载
G=nx.read_gml('./football.gml')
# 可视化
nx.draw(G,with_labels=True) 
plt.show()
# 社区发现
communities = list(community.label_propagation_communities(G))
print(communities)
print(len(communities))
