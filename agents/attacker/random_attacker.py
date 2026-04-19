import networkx as nx
import random

def random_directed_traversal(graph: nx.DiGraph, start_node=None, max_steps=10):

    if start_node is None:
        start_node = random.choice(list(graph.nodes))

    path = [start_node]
    current_node = start_node

    for _ in range(max_steps):
        successors = list(graph.successors(current_node))
        if not successors:
            break
        current_node = random.choice(successors)
        path.append(current_node)

    return path