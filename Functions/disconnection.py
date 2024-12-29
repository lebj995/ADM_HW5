import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

def bfs_find_path(residual, source, sink, parent):
    """Perform Breadth-First Search (BFS) to find a path in the residual graph.
    
    Args:
        residual (dict): The residual graph represented as an adjacency list.
        source (int): The source node.
        sink (int): The sink node.
        parent (dict): A dictionary to store the path.

    Returns:
        bool: True if a path from source to sink is found, False otherwise.
    """
    visited = {node: False for node in residual}  # Track visited nodes
    queue = deque([source])  # Initialize the queue with the source node
    visited[source] = True

    while queue:
        current = queue.popleft()

        for neighbor in residual[current]:
            # If neighbor is unvisited and residual capacity is positive
            if not visited[neighbor] and residual[current][neighbor] > 0:
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = current  # Record the path
                if neighbor == sink:  # Stop if we reach the sink node
                    return True
    return False

def edmonds_karp_min_cut_custom(graph, source, sink):
    """Custom implementation of the Edmonds-Karp algorithm to calculate the minimum cut.
    
    Args:
        graph (nx.Graph): The input graph with weights as capacities.
        source (int): The source node.
        sink (int): The sink node.

    Returns:
        tuple: A tuple containing:
            - cut_edges (list): Edges in the minimum cut.
            - subgraph1 (set): Nodes in the first subset.
            - subgraph2 (set): Nodes in the second subset.
            - max_flow (int): The maximum flow value.
            - cut_weight (int): The total weight of the minimum cut.
    """
    # Initialize the residual graph
    residual = {u: {} for u in graph}
    for u, v, data in graph.edges(data=True):
        residual[u][v] = data['weight']
        residual[v][u] = data['weight']  # Symmetric capacities for undirected graph

    max_flow = 0
    parent = {}

    # Edmonds-Karp algorithm: Find maximum flow using BFS
    while bfs_find_path(residual, source, sink, parent):
        # Find the maximum flow along the path found
        path_flow = float('Inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        # Update residual capacities in the graph
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u

        max_flow += path_flow

    # Identify nodes reachable from the source in the residual graph
    visited = {node: False for node in graph}
    queue = deque([source])
    visited[source] = True

    while queue:
        current = queue.popleft()
        for neighbor in residual[current]:
            if not visited[neighbor] and residual[current][neighbor] > 0:
                visited[neighbor] = True
                queue.append(neighbor)

    # Identify edges crossing the cut (from reachable to non-reachable nodes)
    cut_edges = []
    for u in visited:
        for v in graph[u]:
            if visited[u] and not visited[v]:
                cut_edges.append((u, v))

    # Separate nodes into two subsets
    subgraph1 = set(u for u in visited if visited[u])
    subgraph2 = set(graph) - subgraph1

    # Calculate the total weight of the cut
    cut_weight = sum(graph[u][v]['weight'] for u, v in cut_edges)

    return cut_edges, subgraph1, subgraph2, max_flow, cut_weight

def visualize_cut_result(graph, cut_edges, subgraph1, subgraph2):
    """Visualize the result of the minimum cut with separated subgraphs.
    
    Args:
        graph (nx.Graph): The original graph.
        cut_edges (list): Edges in the minimum cut.
        subgraph1 (set): Nodes in the first subset.
        subgraph2 (set): Nodes in the second subset.
    """
    # Create a figure for the visualization
    plt.figure(figsize=(10, 7))

    # Create subgraphs for each subset
    G1 = graph.subgraph(subgraph1)
    G2 = graph.subgraph(subgraph2)

    # Compute layout for the entire graph to maintain consistency
    pos = nx.spring_layout(graph)

    # Extract positions for each subgraph
    pos1 = {node: pos[node] for node in subgraph1}
    pos2 = {node: pos[node] for node in subgraph2}

    # Draw the first subgraph
    plt.subplot(1, 2, 1)  # Add first subgraph on the left
    nx.draw_networkx_nodes(G1, pos1, node_color='lightblue', label="Subgraph 1")
    nx.draw_networkx_edges(G1, pos1, edge_color='gray')
    nx.draw_networkx_labels(G1, pos1)
    nx.draw_networkx_edge_labels(G1, pos1, edge_labels=nx.get_edge_attributes(G1, 'weight'), font_color='blue')
    plt.title("Subgraph 1")

    # Draw the second subgraph
    plt.subplot(1, 2, 2)  # Add second subgraph on the right
    nx.draw_networkx_nodes(G2, pos2, node_color='lightgreen', label="Subgraph 2")
    nx.draw_networkx_edges(G2, pos2, edge_color='gray')
    nx.draw_networkx_labels(G2, pos2)
    nx.draw_networkx_edge_labels(G2, pos2, edge_labels=nx.get_edge_attributes(G2, 'weight'), font_color='blue')
    plt.title("Subgraph 2")

    # Highlight the cut edges in the original graph (red dashed lines)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='red', width=2, style='dashed', label="Cut Edges")

    # Display the final visualization
    plt.suptitle("Result of Min-Cut - Separated Subgraphs")
    plt.show()

def create_test_graph():
    """Create a test graph for demonstrating the min-cut algorithm.

    Returns:
        nx.Graph: A test graph with weighted edges.
    """
    G = nx.Graph()
    edges = [
        (1, 3, 10),
        (2, 3, 7),
        (2, 4, 8),
        (2, 5, 3),
        (2, 6, 6),
        (3, 5, 2),
        (5, 6, 4),
        (4, 6, 10)
    ]

    # Add edges with weights to the graph
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G
 