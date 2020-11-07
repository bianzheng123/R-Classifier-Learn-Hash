import _init_paths

with open('graph/delaunay_n15.graph', 'r') as f:
    graph_data = f.read().split('\n')

vertics, edges = graph_data[0].split(' ')
sum_vertics = len(graph_data) - 2
assert int(sum_vertics) == int(vertics)
sum_edges = 0
for vertics_info in graph_data[1:]:
    neighbor = vertics_info.split(' ')
    # print(len(neighbor) - 1)
    sum_edges += len(neighbor) - 1

assert int(sum_edges) == int(edges) * 2
