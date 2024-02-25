#import pandas as pd
import networkx as nx
#import matplotlib.pyplot as plt
import csv
from itertools import combinations
#from pyvis.network import Network
#import plotly.graph_objects as go

movielens_dir = '/Users/devynfleming/Downloads/movienetwork 2/'

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






def get_movies_for_years(start_year, end_year):
    """Gets all moves with the span of these years"""
    all_movie_list = get_all_movies()
    my_movie_list = [data for data in all_movie_list if start_year <= int(data[3]) <= end_year]

    return my_movie_list


def get_movies_for_meryl():
    edge_list = get_all_cast_edges()
    meryl_movies = [data for data in edge_list if data[1] == 'a5064']

    return meryl_movies

def get_movies_for_anne():
    edge_list = get_all_cast_edges()
    anne_movies = [data for data in edge_list if data[1] == 'a1813']

    return anne_movies

def get_movies_for_julie():
    edge_list = get_all_cast_edges()
    julie_movies = [data for data in edge_list if data[1] == 'a5823']

    return julie_movies

def get_edges_for_movie_list(movie_list, edge_list):
    """Returns the sublist of movie-role edges for movies in the movie_list"""
    movie_id_set = set()
    for movie_data in movie_list:
        movie_id_set.add(movie_data[0])

    my_edge_list = []

    for edge in edge_list:
        if edge[0] in movie_id_set:
            my_edge_list.append(edge)

    return my_edge_list



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

def create_female_actor_graph(movie_list, cast_edges, actor_map):
    filtered_movies = get_movies_for_years(2010, 2010)

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

    G = nx.Graph()

    # Add nodes with labels
    for actor_id, actor_data in female_actors.items():
        G.add_node(actor_id, label=actor_data[1])

    # Set the 'label' attribute directly for each node
    labels_mapping = {actor_id: actor_data[1] for actor_id, actor_data in female_actors.items()}
    nx.set_node_attributes(G, labels_mapping, 'label')

    # Add edges only between female actors
    for edge in cumulative_edge_list:
        if edge[0] in female_actors and edge[1] in female_actors:
            G.add_edge(edge[0], edge[1])

    gephi_file_name = 'female_actor_network_2010_2010Take2.gexf'

    # Write the Gephi file
    nx.write_gexf(G, gephi_file_name)

    print(f'Graph exported to {gephi_file_name}')

    test_cast_file_name = movielens_dir + 'female-2010-cast.csv'
    test_edge_file_name = movielens_dir + 'female-2010-edges.csv'

    export_list(test_edge_file_name, cumulative_edge_list, edge_header)
    export_list(test_cast_file_name, list(female_actors.values()), person_header)

    return G  # Add this line to return the graph



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
    gephi_file_name = 'meryl_network.gexf'

    # Write the Gephi file
    nx.write_gexf(G, gephi_file_name)

    print(f'Graph exported to {gephi_file_name}')
    # Add nodes

    export_list(meryl_edge_file_name, cumulative_edge_list, edge_header)
    export_list(meryl_nodes_file_name, list(my_cast_map.values()), person_header)

def create_julie_list():
    julie_nodes_file_name = movielens_dir + 'julie-node.csv'
    julie_edge_file_name = movielens_dir + 'julie-edges.csv'
    all_cast_edges = get_all_cast_edges()
    movie_data = get_movies_for_julie()
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
    gephi_file_name = 'julie_network.gexf'

    # Write the Gephi file
    nx.write_gexf(G, gephi_file_name)

    print(f'Graph exported to {gephi_file_name}')
    # Add nodes

    export_list(julie_edge_file_name, cumulative_edge_list, edge_header)
    export_list(julie_nodes_file_name, list(my_cast_map.values()), person_header)

def create_anne_list():
    anne_nodes_file_name = movielens_dir + 'anne-node.csv'
    anne_edge_file_name = movielens_dir + 'anne-edges.csv'
    all_cast_edges = get_all_cast_edges()
    movie_data = get_movies_for_anne()
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
    gephi_file_name = 'anne_network.gexf'

    # Write the Gephi file
    nx.write_gexf(G, gephi_file_name)

    print(f'Graph exported to {gephi_file_name}')
    # Add nodes

    export_list(anne_edge_file_name, cumulative_edge_list, edge_header)
    export_list(anne_nodes_file_name, list(my_cast_map.values()), person_header)

def main():
    # ...

    m = get_movies_for_meryl()
    n = get_movies_for_anne()
    j = get_movies_for_julie()
    i = 0
    for movies in m :
        print(i)
        i = i+1
    create_meryl_list()

    for movies in n :
        print(i)
        i = i+1
    create_anne_list()

    for movies in j :
        print(i)
        i = i+1
    create_julie_list()
#     all_movies = get_all_movies()
#     all_cast_edges = get_all_cast_edges()
#     all_actor_map = get_actor_map()

#     print("Number of movies:", len(all_movies))
#     print("Number of cast edges:", len(all_cast_edges))
#     print("Number of actors:", len(all_actor_map))
# # Call the function and store the returned graph
#     female_actor_graph = create_female_actor_graph(all_movies, all_cast_edges, all_actor_map)

# # Check the number of nodes and edges
#     if female_actor_graph:
#         print("Number of nodes in female actor graph:", len(female_actor_graph.nodes))
#         print("Number of edges in female actor graph:", len(female_actor_graph.edges))
#     else:
#         print("Graph creation failed.")

#         print("Number of nodes in female actor graph:", len(female_actor_graph.nodes))
#         print("Number of edges in female actor graph:", len(female_actor_graph.edges))







# ...

if __name__ == "__main__":
    main()
