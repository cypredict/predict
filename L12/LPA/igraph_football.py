import igraph
g = igraph.Graph.Read_GML('./football.gml')
igraph.plot(g)
print(g.community_label_propagation())

