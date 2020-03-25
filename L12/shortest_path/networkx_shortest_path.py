import networkx as nx
import matplotlib.pyplot as plt

G=nx.read_gml('./football.gml')
nx.draw(G,with_labels=True) 
#plt.show() 
print(nx.shortest_path(G, source='Buffalo', target='Kent'))
print(nx.shortest_path(G, source='Buffalo', target='Rice'))

# Dijkstra算法
print(nx.single_source_dijkstra_path(G, 'Buffalo'))
print(nx.multi_source_dijkstra_path(G, {'Buffalo', 'Rice'}))
# Flody算法
print(nx.floyd_warshall(G, weight='weight'))
