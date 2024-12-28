import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

def bfs_find_path(residual, source, sink, parent):
    """Breadth-First Search per trovare un cammino nel grafo residuo."""
    visited = {node: False for node in residual}
    queue = deque([source])
    visited[source] = True

    while queue:
        current = queue.popleft()

        for neighbor in residual[current]:
            if not visited[neighbor] and residual[current][neighbor] > 0:
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = current
                if neighbor == sink:
                    return True
    return False

def edmonds_karp_min_cut_custom(graph, source, sink):
    """Implementazione personalizzata dell'algoritmo Edmonds-Karp per calcolare il min-cut."""
    # Inizializza il grafo residuo
    residual = {u: {} for u in graph}
    for u, v, data in graph.edges(data=True):
        residual[u][v] = data['weight']
        residual[v][u] = data['weight']  # Grafo non orientato -> capacità simmetrica

    max_flow = 0
    parent = {}

    # Algoritmo Edmonds-Karp: trova il massimo flusso
    while bfs_find_path(residual, source, sink, parent):
        # Trova il flusso massimo possibile lungo il cammino trovato
        path_flow = float('Inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        # Aggiorna le capacità residue nel grafo
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u

        max_flow += path_flow

    # Trova i nodi raggiungibili dal source nel grafo residuo
    visited = {node: False for node in graph}
    queue = deque([source])
    visited[source] = True

    while queue:
        current = queue.popleft()
        for neighbor in residual[current]:
            if not visited[neighbor] and residual[current][neighbor] > 0:
                visited[neighbor] = True
                queue.append(neighbor)

    # Gli archi che attraversano la cut sono tra nodi raggiungibili e non raggiungibili
    cut_edges = []
    for u in visited:
        for v in graph[u]:
            if visited[u] and not visited[v]:
                cut_edges.append((u, v))

    # Dividi il grafo in due sottografi
    subgraph1 = set(u for u in visited if visited[u])
    subgraph2 = set(graph) - subgraph1

    # Calcola il peso totale del taglio
    cut_weight = sum(graph[u][v]['weight'] for u, v in cut_edges)

    return cut_edges, subgraph1, subgraph2, max_flow, cut_weight

def visualize_cut_result(graph, cut_edges, subgraph1, subgraph2):
    """Visualizza il risultato del min-cut con i sottografi separati."""
    plt.figure(figsize=(10, 7))
    
    # Crea due sottografi separati
    G1 = graph.subgraph(subgraph1)
    G2 = graph.subgraph(subgraph2)
    
    # Calcola il layout per l'intero grafo originale per mantenere la coerenza visiva
    pos = nx.spring_layout(graph)
    
    # Estrai solo le posizioni necessarie per ogni sottografo
    pos1 = {node: pos[node] for node in subgraph1}
    pos2 = {node: pos[node] for node in subgraph2}
    
    # Disegna i nodi e gli archi del primo sottografo
    nx.draw_networkx_nodes(G1, pos1, node_color='lightblue', label="Subgraph 1")
    nx.draw_networkx_edges(G1, pos1, edge_color='gray')
    
    # Disegna i nodi e gli archi del secondo sottografo
    nx.draw_networkx_nodes(G2, pos2, node_color='lightgreen', label="Subgraph 2")
    nx.draw_networkx_edges(G2, pos2, edge_color='gray')
    
    # Disegna gli archi del cut con un colore distintivo (es. rosso)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='red', width=2, style='dashed', label="Cut Edges")
    
    # Disegna le etichette per tutti i nodi
    nx.draw_networkx_labels(graph, pos)
    
    # Aggiungi le etichette dei pesi per gli archi di ogni sottografo
    edge_labels_G1 = nx.get_edge_attributes(G1, "weight")
    edge_labels_G2 = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G1, pos1, edge_labels=edge_labels_G1, font_color="blue")
    nx.draw_networkx_edge_labels(G2, pos2, edge_labels=edge_labels_G2, font_color="blue")
    
    # Aggiungi le etichette dei pesi per gli archi del cut
    edge_labels_cut = {edge: graph.edges[edge]["weight"] for edge in cut_edges if "weight" in graph.edges[edge]}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_cut, font_color="red")
    
    plt.legend()
    plt.title("Result of Min-Cut - Separated Subgraphs")
    plt.show()


def create_test_graph():
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

    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G