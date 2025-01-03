import random
import time
from collections import defaultdict

'''

Functions used for Part 5

'''

def compute_modularity(adj_list, communities, m):
    """Calculate the weighted modularity of the graph."""
    Q = 0.0
    for node in adj_list:
        for neighbor, weight in adj_list[node].items():
            if communities[node] == communities[neighbor]:  # Check if nodes are in the same community
                ki = sum(adj_list[node].values())  # Weighted degree of the node
                kj = sum(adj_list[neighbor].values())  # Weighted degree of the neighbor
                Q += weight - (ki * kj) / (2 * m)  # Modularity formula
    return Q / (2 * m)



def louvain_with_max_iterations(adj_list, max_iterations=600, print_limit=5, start_time=None):
    """Louvain algorithm with a maximum number of iterations."""
    # Initialize each node in a separate community
    communities = {node: i for i, node in enumerate(adj_list)}
    m = sum(sum(neighbors.values()) for neighbors in adj_list.values()) / 2  # Total weight of the graph

    print(f"Total nodes: {len(adj_list)}, Total graph weight (2m): {2 * m}")

    improvement = True
    iteration = 0
    modularity_history = []  # Track modularity improvements for debugging and analysis

    while iteration < max_iterations:
        iteration += 1
        improvement = False  # Reset improvement flag for this iteration
        nodes = list(adj_list.keys())
        random.shuffle(nodes)  # Shuffle the nodes to process them in a random order

        # Print only the first and last iterations based on the limit
        if iteration <= print_limit or iteration > max_iterations - print_limit:
            print(f"\nIteration {iteration} - Nodes: {len(nodes)}")

        for node in nodes:
            best_community = communities[node]  # Initialize with the current community
            best_increase = 0  # Initialize the best modularity increase as zero
            current_community = communities[node]  # Store the current community of the node

            # Compute the sum of weights for each neighboring community
            neighbor_communities = defaultdict(float)
            for neighbor, weight in adj_list[node].items():
                neighbor_communities[communities[neighbor]] += weight

            # Evaluate modularity gain for moving the node to each neighboring community
            for community, weight_sum in neighbor_communities.items():
                ki = sum(adj_list[node].values())  # Weighted degree of the node
                sigma_tot = sum(sum(adj_list[n].values()) for n in adj_list if communities[n] == community)
                delta_Q = (weight_sum - (ki * sigma_tot) / (2 * m))  # Modularity gain formula

                if delta_Q > best_increase:  # Check if this move improves modularity
                    best_community = community
                    best_increase = delta_Q

            # Update the community of the node if a better one was found
            if best_community != current_community:
                communities[node] = best_community
                improvement = True

        # Compute the modularity after the current iteration
        current_modularity = compute_modularity(adj_list, communities, m)
        modularity_history.append(current_modularity)  # Store modularity for analysis

        # Print only the first and last iterations based on the limit
        if iteration <= print_limit or iteration > max_iterations - print_limit:
            print(f"Iteration {iteration} - Modularity: {current_modularity:.6f}")
            print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    print(f"\nMaximum iterations ({max_iterations}) reached.")
    print("Algorithm completed.")

    # Group nodes by their community
    community_groups = defaultdict(list)
    for node, community in communities.items():
        community_groups[community].append(node)

    return list(community_groups.values())

def print_communities(communities):
    """
    Print all detected communities.
    """
    print(f"Total number of detected communities: {len(communities)}")
    for i, community in enumerate(communities):
        print(f"Community {i+1} ({len(community)} nodes):")
        print(", ".join(community))
        print("-" * 50)


def analyze_cities_communities(city_communities, city1, city2):
    """
    Analyze communities to check if two cities belong to the same community.

    Input:
        - city_communities: List of communities (of cities)
        - city1: Name of the first city
        - city2: Name of the second city

    Output:
        - Indicates whether city1 and city2 belong to the same community
    """
    for i, community in enumerate(city_communities):
        if city1 in community and city2 in community:
            print(f"{city1} and {city2} belong to the same community ({i + 1}).")
            return
    print(f"{city1} and {city2} do not belong to the same community.")


def analyze_community_statistics(df, communities):
    """Analyze statistics for the first 5 communities."""
    for i, community in enumerate(communities[:5]):  # Limit to the first 5 communities
        # Filter the DataFrame for nodes in the community
        community_df = df[df['Origin_airport'].isin(community) & df['Destination_airport'].isin(community)]

        # Calculate aggregated statistics
        total_flights = community_df['Flights'].sum()
        total_passengers = community_df['Passengers'].sum()
        avg_distance = community_df['Distance'].mean()

        # Print statistics for the community
        print(f"Community {i+1}:")
        print(f"  Number of nodes: {len(community)}")
        print(f"  Total flights: {total_flights}")
        print(f"  Total passengers: {total_passengers}")
        print(f"  Average distance: {avg_distance:.2f} km")
        print("-" * 50)

def analyze_geographic_distribution(df, communities):
    """Analyze the geographic distribution of the community."""
    for i, community in enumerate(communities):
        # Filter the DataFrame for nodes in the community
        community_df = df[df['Origin_airport'].isin(community)]

        # Get the cities and their counts
        cities = community_df['Origin_city'].value_counts()

        print(f"Community {i+1}:")
        print(f"  Top cities (by frequency):")
        print(cities.head(5))  # Display the top 5 cities
        print("-" * 50)


def label_propagation_weighted(adj_list_weighted):
    """
    Manual implementation of Label Propagation on a weighted graph.

    Input:
        - adj_list_weighted: Weighted adjacency dictionary {node: {neighbor: weight, ...}}
    Output:
        - communities: List of communities (each community is a list of nodes)
    """
    # 1. Initialize each node with a unique label
    labels = {node: node for node in adj_list_weighted}

    # 2. Iterate until convergence
    converged = False
    while not converged:
        converged = True
        nodes = list(adj_list_weighted.keys())
        random.shuffle(nodes)  # Random order of nodes

        for node in nodes:
            # Count weighted labels of neighbors
            neighbor_labels = {}
            for neighbor, weight in adj_list_weighted[node].items():
                label = labels[neighbor]
                if label not in neighbor_labels:
                    neighbor_labels[label] = 0
                neighbor_labels[label] += weight  # Sum of weights per label

            # Find the label with the maximum weight
            if neighbor_labels:
                new_label = max(neighbor_labels, key=neighbor_labels.get)

                # If the label changes, update and continue the iteration
                if labels[node] != new_label:
                    labels[node] = new_label
                    converged = False

    # 3. Group nodes by label
    communities = {}
    for node, label in labels.items():
        if label not in communities:
            communities[label] = []
        communities[label].append(node)

    return list(communities.values())


def analyze_communities_stats(communities, method_name):
    sizes = [len(community) for community in communities]
    print(f"\n[{method_name}]")
    print(f"Number of communities: {len(communities)}")
    print(f"Minimum community size: {min(sizes)}")
    print(f"Maximum community size: {max(sizes)}")
    print(f"Average community size: {sum(sizes) / len(sizes):.2f}")
