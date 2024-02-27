import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import csv
from itertools import combinations
from pyvis.network import Network
import plotly.graph_objects as go
from matplotlib.dates import AutoDateLocator

movielens_dir = '/Users/sylviagreene/Desktop/movienetwork/'

movie_file_name = movielens_dir + 'vertex-movies.csv'
actor_file_name = movielens_dir + 'vertex-actor.csv'
crew_file_name = movielens_dir +  'vertex-crew.csv'

cast_file_name = movielens_dir + 'edge-cast.csv'
credit_file_name = movielens_dir + 'edge-credit.csv'

movie_header = ["Id", "imdbId", "Label", "Year", "Genre"]
person_header = ["Id", "Label", "Gender"]

edge_header = ["Source", "Target", "Role", "Year"]

def export_list(out_file_name, data_list, header=[]):
    """Exports a list of lists to a csv file, with optional header"""
    print('Exporting to', out_file_name)
    if out_file_name in [movie_file_name, actor_file_name, crew_file_name, cast_file_name, credit_file_name]:
        raise NameError('DO not overwrite your raw data files. Use a different name than ' + out_file_name)
    file = open(out_file_name, 'w')
    writer = csv.writer(file)

    if len(header) > 0:
        writer.writerow(header)

    writer.writerows(data_list)

    file.close()


def get_data_from_file(file_name):
    """gets csv data from a file"""
    data_list = []

    with open(file_name, mode='r') as in_file:
        print('reading', file_name)
        csv_file = csv.reader(in_file)
        next(csv_file, None)  # skip the headers
        for data in csv_file:
            data_list.append(data)

    in_file.close()

    return data_list


def get_all_movies():
    return get_data_from_file(movie_file_name)


def get_all_cast_edges():
    return get_data_from_file(cast_file_name)


def get_all_credit_edges():
    return get_data_from_file(credit_file_name)


def get_data_map_from_file(file_name):
    """ Gets data from a csv file and returns a dictionary using the first entry """
    data_list = get_data_from_file(file_name)
    data_map = dict()

    for data in data_list:
        data_map[data[0]] = data

    return data_map

def get_actor_map():
    return get_data_map_from_file(actor_file_name)


def get_crew_map():
    return get_data_map_from_file(crew_file_name)

def get_edges_for_movie_list(movie_list, edge_list):
    # Returns a sublist of movie-role edges for movies in the movie list
    movie_id_set = set()
    for movie_data in movie_list:
        movie_id_set.add(movie_data[0])

    my_edge_list = []

    for edge in edge_list:
        if edge[0] in movie_id_set:
            my_edge_list.append(edge)

    return my_edge_list




def export_edges_to_csv(file_name, edge_list):
    """Exports a list of edges to a CSV file"""
    export_list(file_name, edge_list, edge_header)


def get_movies_for_years(start_year, end_year):
    """Gets all moves with the span of these years"""
    all_movie_list = get_all_movies()
    my_movie_list = [data for data in all_movie_list if int(start_year) <= int(data[3]) <= int(end_year)]


    return my_movie_list


def get_movies_for_meryl():
    edge_list = get_all_cast_edges()
    meryl_movies = [data for data in edge_list if data[1] == 'a5064']

    return meryl_movies

def get_movies_for_meryl_edges(start_year, end_year):
    """
    Gets movie edges for movies in which Meryl Streep has participated within the specified years.
    """
    all_cast_edges = get_all_cast_edges()
    meryl_movies = get_movies_for_meryl()
    meryl_edges = get_edges_for_movie_list(meryl_movies, all_cast_edges)
    
    filtered_movies = filter_movies_by_year(meryl_movies, start_year, end_year)
    filtered_edges = get_edges_for_movie_list(filtered_movies, all_cast_edges)

        # Print the number of different movies
    unique_movies = set(movie[0] for movie in filtered_edges)
    print(f"Number of different movies: {len(unique_movies)}")

    export_edges_to_csv("meryl_edges.csv", filtered_edges)
    return filtered_edges



def get_timestamp_edges(edge_list):
    """Converts a list of edges with year at index 3 into cumulative timestamp edges"""
    all_edge_map = dict()

    # create weighted edges for the edges in each year
    for edge in edge_list:
        key = edge[0] + '-' + edge[1]
        if not key in all_edge_map:
            all_edge_map[key] = dict()

        my_map = all_edge_map[key]
        year = int(edge[3])

        if not year in my_map:
            my_map[year] = 1
        else:
            my_map[year] = my_map[year] + 1

    # now make cumulative edges
    cumulative_edge_list = []
    for edge_key in all_edge_map:
        my_map = all_edge_map[edge_key]

        id_list = edge_key.split('-')

        year_list = list(my_map.keys())
        year_list.sort()

        total = 0

        for year in year_list:
            total = total + my_map[year]
            cumulative_edge_list.append([id_list[0], id_list[1], year, total, 'Undirected'])

    return cumulative_edge_list



def make_network_edges(role_list):
    """Converts movie-role edges into role-role edges"""
    movie_map = dict()
    for role in role_list:
        movie_id = role[0]
        person_id = role[1]

        if not movie_id in movie_map:
            year = role[-1]
            movie_map[movie_id] = [movie_id, year, []]

        data = movie_map[movie_id]
        data[2].append(person_id)

    edge_list = []
    for data in movie_map.values():
        people = data[2]
        pair_list = list(combinations(people, 2))

        for pair in pair_list:
            edge_list.append([pair[0], pair[1], data[0], data[1], 'Undirected'])
    return edge_list


def get_roles_for_edges(role_map, edge_list):
    """Returns a map with the matching roles (vertices) for ids in the edges of edge_list"""
    my_role_map = dict()

    for edge in edge_list:
        for id in edge[0:2]:
            if not id in my_role_map:
                my_role_map[id] = role_map[id]

    return my_role_map


def create_timestamp_network_1997_2005():
    '''An example that creates an actor-actor network where edges accumulate over time'''
    test_cast_file_name = movielens_dir + 'test-cast.csv'
    test_edge_file_name = movielens_dir + 'test-edges.csv'

    my_movies = get_movies_for_years(1997, 2005)
    all_cast_edges = get_all_cast_edges()
    my_cast_edges = get_edges_for_movie_list(my_movies, all_cast_edges)

    movie_data = make_network_edges(my_cast_edges)

    cumulative_edge_list = get_timestamp_edges(movie_data)

    all_actor_map = get_actor_map()

    my_cast_map = get_roles_for_edges(all_actor_map, cumulative_edge_list)

    print('role num', len(my_cast_map))
    print('edge num', len(cumulative_edge_list))

    G = nx.Graph()

    # Add nodes
    for actor_id, actor_data in my_cast_map.items():
        G.add_node(actor_id, label=actor_data[1])  

    # Add edges
    for edge in cumulative_edge_list:
        G.add_edge(edge[0], edge[1])




    export_list(test_edge_file_name, cumulative_edge_list, edge_header)
    export_list(test_cast_file_name, list(my_cast_map.values()), person_header)


def filter_movies_by_year(movie_list, start_year, end_year):
    return [data for data in movie_list if start_year <= int(data[3]) <= end_year]

def create_female_actor_graph(year1,year2, cast_edges, actor_map):
    filtered_movies = get_movies_for_years(year1, year2)

    # Get edges for the filtered movies
    filtered_cast_edges = get_edges_for_movie_list(filtered_movies, cast_edges)

    # Create the role-role edges
    movie_data = make_network_edges(filtered_cast_edges)

    # Convert movie-role edges into actor-actor edges
    cumulative_edge_list = get_timestamp_edges(movie_data)

    # Get roles for actor-actor edges
    actor_roles_map = get_roles_for_edges(actor_map, cumulative_edge_list)

    # Filter female actors
    female_actors = {k: v for k, v in actor_roles_map.items() if v[-1] == '1'}
    print("Number of female actors:", len(female_actors))
        # Check if the graph is not empty before computing eigenvector centrality
    if female_actors:
        G = nx.Graph()

        # Add nodes with labels
        for actor_id, actor_data in female_actors.items():
            G.add_node(actor_id, label=actor_data[1])

        labels_mapping = {actor_id: actor_data[1] for actor_id, actor_data in female_actors.items()}
        nx.set_node_attributes(G, labels_mapping, 'label')

        # Add edges only between female actors
        for edge in cumulative_edge_list:
            if edge[0] in female_actors and edge[1] in female_actors:
                G.add_edge(edge[0], edge[1])

        # Write the Gephi file or any other operations you need
        # gephi_file_name = 'female_actor_network_2010_2010.gexf'
        # nx.write_gexf(G, gephi_file_name)
        # print(f'Graph exported to {gephi_file_name}')

        return G  # Return the graph

    else:
        print("Graph is empty for the specified years.")
        return None




def create_meryl_list():
    meryl_nodes_file_name = movielens_dir + 'meryl-node.csv'
    meryl_edge_file_name = movielens_dir + 'meryl-edges.csv'
    all_cast_edges = get_all_cast_edges()
    movie_data = get_movies_for_meryl()
    my_cast_edges = get_edges_for_movie_list(movie_data, all_cast_edges)

    movie_data = make_network_edges(my_cast_edges)

    cumulative_edge_list = get_timestamp_edges(movie_data)

    all_actor_map = get_actor_map()

    my_cast_map = get_roles_for_edges(all_actor_map, cumulative_edge_list)

    print('role num', len(my_cast_map))
    print('edge num', len(cumulative_edge_list))

    G = nx.Graph()
    for actor_id, actor_data in my_cast_map.items():
        G.add_node(actor_id, label=actor_data[1])  

    # Add edges
    for edge in cumulative_edge_list:
        G.add_edge(edge[0], edge[1])
    # gephi_file_name = 'meryl_network.gexf'

    # # Write the Gephi file
    # # nx.write_gexf(G, gephi_file_name)

    # print(f'Graph exported to {gephi_file_name}')
    # Add nodes

    # export_list(meryl_edge_file_name, cumulative_edge_list, edge_header)
    # export_list(meryl_nodes_file_name, list(my_cast_map.values()), person_header)

def get_years_for_meryl():
    edge_list = get_all_cast_edges()
    meryl_movies = [data for data in edge_list if data[1] == 'a5064']
    
    # Extract unique years when Meryl Streep was in a movie
    meryl_years = set(data[3] for data in meryl_movies)
    
    return meryl_years

def main():
    # ...
    #   get_movies_for_meryl_edges(1977,2019)
    meryl_streep_node = "a5064" 
    all_movies = get_all_movies()
    all_cast_edges = get_all_cast_edges()
    all_actor_map = get_actor_map()
    female_actor_graph = create_female_actor_graph(1980,2000, all_cast_edges, all_actor_map)
    print("Eigenvector Centrality 1980-2000" )
    eigenvector_centrality = nx.eigenvector_centrality(female_actor_graph)
    print(eigenvector_centrality.get(meryl_streep_node, 0))
    print("Degree Centrality 1980-2000") 
    deg_centrality = nx.degree_centrality(female_actor_graph)
    print(deg_centrality[meryl_streep_node])
    # close_centrality = nx.closeness_centrality(female_actor_graph)
    # bet_centrality = nx.betweenness_centrality(female_actor_graph, normalized = True,endpoints=False)
    # print("Betweeness Centrality 1980-2000" )
    # print( bet_centrality(meryl_streep_node,0))
    # print("Closeness Centrality 1980-2000")
    # print( close_centrality(meryl_streep_node,0) )
    all_deg_centrality = []
    deg_centrality_scores = {}
    close_centrality_scores = {}
    bet_centrality_scores = {}
    pr_scores = {}
    eigenvector_centrality_scores = {}  

    meryl_years = get_years_for_meryl()

    for year in meryl_years:
        print(year)
        female_actor_graph = create_female_actor_graph(year, year, all_cast_edges, all_actor_map)
    
    # Calculate centrality measures
        deg_centrality = nx.degree_centrality(female_actor_graph)
        # close_centrality = nx.closeness_centrality(female_actor_graph)
        # bet_centrality = nx.betweenness_centrality(female_actor_graph, normalized=True, endpoints=False)
        # pr = nx.pagerank(female_actor_graph, alpha=0.8)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(female_actor_graph)
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = {node: 0 for node in female_actor_graph.nodes()}



    # Store the scores for Meryl Streep
        meryl_streep_node = "a5064"   
        deg_centrality_scores[year] = deg_centrality.get(meryl_streep_node, 0)
        # close_centrality_scores[year] = close_centrality.get(meryl_streep_node, 0)
        # bet_centrality_scores[year] = bet_centrality.get(meryl_streep_node, 0)
        # pr_scores[year] = pr.get(meryl_streep_node, 0)
        eigenvector_centrality_scores[year] = eigenvector_centrality.get(meryl_streep_node, 0)

    # Collect degree centralities for all nodes
        all_deg_centrality.extend(list(deg_centrality.values()))

# Plot the distribution of all degree centralities
    plt.hist(all_deg_centrality, bins=20, alpha=0.5, color='blue', edgecolor='black')
    plt.xlabel('Degree Centrality')
    plt.ylabel('Frequency')
    plt.title('Distribution of Degree Centralities')
    plt.show()

# Plot Meryl Streep's centrality measures compared to other nodes
    plt.plot(deg_centrality_scores.keys(), deg_centrality_scores.values(), label='Meryl Streep Degree Centrality')
    # plt.plot(close_centrality_scores.keys(), close_centrality_scores.values(), label='Meryl Streep Closeness Centrality')
    # plt.plot(bet_centrality_scores.keys(), bet_centrality_scores.values(), label='Meryl Streep Betweenness Centrality')
    # plt.plot(pr_scores.keys(), pr_scores.values(), label='Meryl Streep PageRank')
    plt.plot(eigenvector_centrality_scores.keys(), eigenvector_centrality_scores.values(), label='Meryl Streep Eigenvector Centrality')

# Add labels and legend
    plt.xlabel('Year')
    plt.ylabel('Centrality Score')
    plt.legend()
    plt.title('Meryl Streep Centrality Scores Compared to Other Nodes')
    plt.show()

    plt.hist(all_deg_centrality, bins=30, alpha=0.5, color='blue', edgecolor='black')
    plt.xlabel('Degree Centrality')
    plt.ylabel('Frequency')
    plt.title('Distribution of Degree Centralities')
    plt.show()

    # Plot Meryl Streep's eigenvector centrality compared to other nodes with larger bins
    plt.hist(eigenvector_centrality_scores.values(), bins=30, alpha=0.5, color='green', edgecolor='black')
    plt.xlabel('Eigenvector Centrality Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Eigenvector Centrality Scores')
    plt.show()
    print("Number of movies:", len(all_movies))
    print("Number of cast edges:", len(all_cast_edges))
    print("Number of actors:", len(all_actor_map))


    # Initialize dictionaries to store centrality rankings
    deg_centrality_ranks = {}
    close_centrality_ranks = {}
    bet_centrality_ranks = {}
    pr_ranks = {}
    eigenvector_centrality_ranks = {}

    # Loop through the years
    for year in sorted(map(int, meryl_years)):
        print(get_years_for_meryl)
        female_actor_graph = create_female_actor_graph(year, year, all_cast_edges, all_actor_map)

        # Calculate centrality measures
        deg_centrality = nx.degree_centrality(female_actor_graph)
        close_centrality = nx.closeness_centrality(female_actor_graph)
        # bet_centrality = nx.betweenness_centrality(female_actor_graph, normalized=True, endpoints=False)
        # pr = nx.pagerank(female_actor_graph, alpha=0.8)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(female_actor_graph)
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = {node: 0 for node in female_actor_graph.nodes()}

        # Sort nodes by centrality values in descending order
        sorted_deg_centrality = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)
        sorted_close_centrality = sorted(close_centrality.items(), key=lambda x: x[1], reverse=True)
        # sorted_bet_centrality = sorted(bet_centrality.items(), key=lambda x: x[1], reverse=True)
        # sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        sorted_eigenvector_centrality = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

        # Find Meryl Streep's ranking for each centrality measure
        meryl_streep_node = "a5064"
        meryl_deg_rank = next((i + 1 for i, (node, _) in enumerate(sorted_deg_centrality) if node == meryl_streep_node), None)
        meryl_close_rank = next((i + 1 for i, (node, _) in enumerate(sorted_close_centrality) if node == meryl_streep_node), None)
        # meryl_bet_rank = next((i + 1 for i, (node, _) in enumerate(sorted_bet_centrality) if node == meryl_streep_node), None)
        # meryl_pr_rank = next((i + 1 for i, (node, _) in enumerate(sorted_pr) if node == meryl_streep_node), None)
        meryl_eigenvector_rank = next((i + 1 for i, (node, _) in enumerate(sorted_eigenvector_centrality) if node == meryl_streep_node), None)

        # Store Meryl Streep's ranking for each centrality measure
        deg_centrality_ranks[year] = meryl_deg_rank
        close_centrality_ranks[year] = meryl_close_rank
        # bet_centrality_ranks[year] = meryl_bet_rank
        # pr_ranks[year] = meryl_pr_rank
        eigenvector_centrality_ranks[year] = meryl_eigenvector_rank
        num_nodes_deg_centrality = len(female_actor_graph.nodes())
        num_nodes_eigenvector_centrality = len(female_actor_graph.nodes())
        num_nodes_close_centrality = len(female_actor_graph.nodes())

        norm_deg_rank = meryl_deg_rank / num_nodes_deg_centrality if meryl_deg_rank is not None else None
        norm_eigenvector_rank = meryl_eigenvector_rank / num_nodes_eigenvector_centrality if meryl_eigenvector_rank is not None else None
        norm_close_rank = meryl_close_rank/num_nodes_close_centrality if meryl_close_rank is not None else None
        # Store normalized rankings
        deg_centrality_ranks[year] = norm_deg_rank
        eigenvector_centrality_ranks[year] = norm_eigenvector_rank
        close_centrality_ranks[year] = norm_close_rank

    # Plotting Meryl Streep's normalized ranking over the years with inverted y-axis
    plt.plot(deg_centrality_ranks.keys(), deg_centrality_ranks.values(), label='Degree Centrality')
    plt.plot(eigenvector_centrality_ranks.keys(), eigenvector_centrality_ranks.values(), label='Eigenvector Centrality')
    plt.plot(close_centrality_ranks.keys(), close_centrality_ranks.values(), label = "Closeness Centrality ")
    # Invert the y-axis
    plt.gca().invert_yaxis()


    # Add labels and legend
    plt.xlabel('Year')
    plt.ylabel('Normalized Ranking')
    plt.legend()
    plt.title('Meryl Streep Centrality Normalized Ranking Over Time')
    plt.xticks(list(deg_centrality_ranks.keys()), rotation=45, ha='right')
    plt.show()
    bar_width = 0.35
    years = sorted(map(int, meryl_years))
    plt.bar([year - bar_width/2 for year in years], deg_centrality_ranks, width=bar_width, alpha=0.5, color='blue', label='Degree Centrality')
    plt.bar([year + bar_width/2 for year in years], eigenvector_centrality_ranks, width=bar_width, alpha=0.5, color='green', label='Eigenvector Centrality')

    # Add labels and legend
    plt.xlabel('Year')
    plt.ylabel('Ranking')
    plt.legend()
    plt.title('Bar Chart of Meryl Streep\'s Centrality Rankings Over Time')

    plt.show()
    


# ...

if __name__ == "__main__":
    main()