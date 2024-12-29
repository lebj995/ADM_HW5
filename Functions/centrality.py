from collections import defaultdict, deque
import pandas as pd
from scipy.stats import percentileofscore
import pandas as pd


'''

Functions used for Part 2

'''

def calculate_betweenness_centrality(graph):
    """
    Calculate the betweenness centrality for all nodes in a graph.
    
    param:
        graph: Dictionary representing the graph {node: [list_of_neighbors]}.
    
    return:
        Dictionary {node: betweenness_centrality}.
    """
    # Initialize betweenness centrality for all nodes
    betweenness = {node: 0 for node in graph}

    # Iterate over all nodes as source
    for source in graph:
        # BFS initialization
        stack = []  # Order of exploration
        paths = defaultdict(list)  # Shortest paths to each node
        paths[source] = [[source]]
        queue = deque([source])
        distances = {node: float('inf') for node in graph}  # Distance from source
        distances[source] = 0

        # BFS to compute shortest paths and distances
        while queue:
            current = queue.popleft()
            stack.append(current)

            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):  # First visit
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

                if distances[neighbor] == distances[current] + 1:  # Shortest path
                    paths[neighbor].extend([path + [neighbor] for path in paths[current]])

        # Compute dependency scores
        dependency = {node: 0 for node in graph}

        # Reverse processing of nodes in stack
        while stack:
            node = stack.pop()
            for pred in paths[node]:
                if len(pred) > 1:  # Skip source node
                    parent = pred[-2]
                    dependency[parent] += (1 + dependency[node]) / len(paths[node])

            if node != source:
                betweenness[node] += dependency[node]

    # Normalize for undirected graphs
    normalization = 1 / ((len(graph) - 1) * (len(graph) - 2))
    for node in betweenness:
        betweenness[node] *= normalization

    return betweenness

import matplotlib.pyplot as plt

def plot_top_metric(centrality_metric_dict, metric, top_n=20):
    """
    Plots the distribution of the top N nodes by the specified centrality metric.
    
    param:
        betweenness_centrality: Dictionary {node: centrality_value}.
        metric: Name of the centrality metric to display on the plot.
        top_n: Number of top nodes to include in the plot (default: 20).
    """
    # Sort nodes by the metric in descending order
    sorted_centrality = sorted(centrality_metric_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Select the top N nodes
    top_nodes = sorted_centrality[:top_n]
    
    # Extract node names and their centrality values
    nodes = [node for node, centrality in top_nodes]
    centrality_values = [centrality for node, centrality in top_nodes]
    
    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.bar(nodes, centrality_values, color='skyblue')
    plt.xlabel("Nodes", fontsize=14)
    plt.ylabel(f'{metric}', fontsize=14)
    plt.title(f"Top {top_n} Nodes by {metric}", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def calculate_closeness_centrality(graph):
    """
    Calculate the closeness centrality for all nodes in a directed graph.
    The distance is measured as the number of hops (shortest paths) to each other node.
    
    param:
        graph: Dictionary representing the directed graph {node: [list_of_neighbors]}.
        Dictionary {node: closeness_centrality}.
    """
    closeness = {}

    for source in graph:
        # Step 1: Perform BFS to find shortest paths from 'source'
        distances = {node: float('inf') for node in graph}
        distances[source] = 0
        queue = deque([source])

        while queue:
            current = queue.popleft()

            for neighbor in graph[current]:
                # If neighbor hasn't been visited, update its distance and add it to the queue
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        # Step 2: Compute closeness centrality
        reachable_nodes = [dist for dist in distances.values() if dist != float('inf')]
        total_distance = sum(reachable_nodes)

        if total_distance > 0 and len(reachable_nodes) > 2:
            closeness[source] = (len(reachable_nodes) - 1) / total_distance
        else:
            closeness[source] = 0  # Centrality is 0 if the node is isolated or no paths exist

    return closeness

def numbers_neighbors(graph):
    """
    Calculate the out-degree (number of outgoing neighbors) for each node in a directed graph.

    param
        graph: A directed graph (DiGraph object).
        
    return:
        Dictionary {node: number_of_outgoing_neighbors}.
    """
    neighbors = {}
    for node in graph.nodes: 
        neighbors[node] = len(list(graph.successors(node)))
    
    return neighbors

def calculate_pagerank(graph, damping=0.85, max_iterations=100, tol=1.0e-6):
    """
    Compute the weighted PageRank of nodes in a directed, weighted graph.

    :param
        graph: Directed, weighted graph (networkx.DiGraph).
        damping: Damping factor (probability of following links, usually 0.85).
        max_iterations: Maximum number of iterations.
        tol: Tolerance for convergence (difference threshold).
    
    return:
        Dictionary {node: PageRank score}.
    """
    # Initialize PageRank to 1/N for all nodes
    num_nodes = len(graph)
    pagerank = {node: 1 / num_nodes for node in graph}

    # Handle dangling nodes (nodes with no outgoing edges)
    dangling_nodes = {node for node in graph if len(graph[node]) == 0}

    for _ in range(max_iterations):
        new_pagerank = {}
        # Compute the sum of dangling node contributions
        dangling_sum = damping * sum(pagerank[node] for node in dangling_nodes) / num_nodes

        for node in graph:
            # Start with the teleportation factor
            rank_sum = (1 - damping) / num_nodes + dangling_sum

            # Add contributions from neighbors (weighted by edge weights)
            for neighbor in graph.predecessors(node):  # Use predecessors for incoming edges
                weight = graph[neighbor][node].get("weight", 1)  # Get edge weight, default is 1
                total_weight = sum(graph[neighbor][n].get("weight", 1) for n in graph.successors(neighbor))
                if total_weight != 0:
                    rank_sum += damping * (pagerank[neighbor] * weight / total_weight)
                else:
                    pass
            new_pagerank[node] = rank_sum

        # Check for convergence
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in graph)
        if diff < tol:
            break

        pagerank = new_pagerank

    return pagerank


def analyze_centrality(G_dir, G_no_dir, airport):
    # Calculate the dictionaries for the centrality metrics of each airport
    betweenness_centr = calculate_betweenness_centrality(G_no_dir)
    closeness_centrality = calculate_closeness_centrality(G_no_dir)
    degree_centrality = numbers_neighbors(G_dir)
    pageranks = calculate_pagerank(G_dir)

    # Calculate centrality metrics for the airport
    betweenness = betweenness_centr[airport]
    closeness = closeness_centrality[airport]
    degree = degree_centrality[airport]
    pagerank = pageranks[airport]

    # Calculate percentiles
    betweenness_percentile = percentileofscore(list(betweenness_centr.values()), betweenness)
    closeness_percentile = percentileofscore(list(closeness_centrality.values()), closeness)
    degree_percentile = percentileofscore(list(degree_centrality.values()), degree)
    pagerank_percentile = percentileofscore(list(pageranks.values()), pagerank)

    # Organize metrics and percentiles in a dataframe
    centrality_df = pd.DataFrame({
        'Airport': [airport],
        'Betweenness Centrality': [betweenness],
        'Betweenness Percentile': [betweenness_percentile],
        'Closeness Centrality': [closeness],
        'Closeness Percentile': [closeness_percentile],
        'Degree Centrality': [degree],
        'Degree Percentile': [degree_percentile],
        'Pagerank': [pagerank],
        'Pagerank Percentile': [pagerank_percentile]
    })

    return centrality_df, betweenness_centr, closeness_centrality, degree_centrality, pageranks

def calculate_flow_centrality(graph):
    """
    Calculate the Flow Centrality for each node in the graph.

    Flow Centrality is a measure of a node's importance based on how often it 
    appears in the shortest paths between other nodes in the graph.

    Args:
        graph (dict): A dictionary representing the graph where each key is a node, 
                      and the value is a list of its neighbors.

    Returns:
        dict: A dictionary where keys are nodes, and values are the Flow Centrality 
              scores for each node.
    """
    # Initialize the flow centrality for each node to zero
    flow_centrality = {node: 0 for node in graph}

    # Iterate through each node as the source node
    for source in graph:
        # BFS setup for calculating the shortest paths from 'source' to all other nodes
        stack = []  # Stack to store the order of exploration
        paths = defaultdict(list)  # Dictionary to store the shortest paths to each node
        paths[source] = [[source]]  # The shortest path to the source is just the source itself
        queue = deque([source])  # Queue for BFS traversal
        distances = {node: float('inf') for node in graph}  # Initialize distances to infinity
        distances[source] = 0  # Distance to itself is zero

        # Perform BFS to calculate the shortest paths and distances from the source node
        while queue:
            current = queue.popleft()
            stack.append(current)

            # Explore each neighbor of the current node
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):  # First time visiting this neighbor
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

                # If this neighbor can be reached with the shortest path, update paths
                if distances[neighbor] == distances[current] + 1:
                    paths[neighbor].extend([path + [neighbor] for path in paths[current]])

        # For each target node, update the flow centrality based on shortest paths
        for target in graph:
            if source != target:  # Skip if the source is the same as the target
                for path in paths.get(target, []):
                    # Exclude the start and end nodes from the flow centrality calculation
                    for node in path[1:-1]:
                        flow_centrality[node] += 1

    # Normalize the flow centrality for undirected graphs
    normalization = 1 / ((len(graph) - 1) * (len(graph) - 2))
    for node in flow_centrality:
        flow_centrality[node] *= normalization

    return flow_centrality
