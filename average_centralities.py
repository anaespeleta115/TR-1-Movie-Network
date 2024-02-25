import csv
from itertools import combinations
from operator import itemgetter
from matplotlib import pyplot as plt
import networkx as nx
from numpy import take



movie_file_name = 'vertex-movies.csv'
actor_file_name = 'vertex-actor.csv'
crew_file_name = 'vertex-crew.csv'

cast_file_name = 'edge-cast.csv'
credit_file_name = 'edge-credit.csv'

movie_header = ["Id", "imdbId", "Label", "Year", "Genre"]
person_header = ["Id", "Label", "Gender"]

edge_header = ["Source", "Target", "Role", "Year"]

#####################


def export_list(out_file_name, data_list, header=[]):
    """Exports a list of lists to a csv file, with optional header"""
    # print('Exporting to', out_file_name)
    if out_file_name in [movie_file_name, actor_file_name, crew_file_name, cast_file_name, credit_file_name]:
        raise NameError('DO not overwrite your raw data files. Use a different name than ' + out_file_name)
    with open(out_file_name, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        if len(header) > 0:
            writer.writerow(header)
        writer.writerows(data_list)
    
    file.close()


def get_data_from_file(file_name):
    """gets csv data from a file"""
    data_list = []

    with open(file_name, mode='r', encoding='utf-8') as in_file:
        # print('reading', file_name)
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


def get_movies_for_year(year):
    """Gets all moves with the span of these years"""
    all_movie_list = get_all_movies()
    my_movie_list = [data for data in all_movie_list if year == int(data[3])]

    return my_movie_list


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
            if id in role_map:  # Check if id exists in role_map                
                if not id in my_role_map:
                    my_role_map[id] = role_map[id]

    return my_role_map



def create_female_network(G, year): # edit here when using a year range
    '''An example that creates an actor-actor network where edges accumulate over time'''

    test_cast_file_name = 'test-cast.csv'
    test_edge_file_name = 'test-edges.csv'


    # Filter movies for the year 2010
    # my_movies = get_movies_for_years(start_year, end_year)
    my_movies = get_movies_for_year(year)

    
    # Get all cast edges and filter for movies in 2010
    all_cast_edges = get_all_cast_edges()
    my_cast_edges = get_edges_for_movie_list(my_movies, all_cast_edges)
    
    # Create network edges from cast edges
    movie_data = make_network_edges(my_cast_edges)
    
    # Convert to cumulative timestamp edges
    cumulative_edge_list = get_timestamp_edges(movie_data)
    
    # Get a map of all actors
    all_actor_map = get_actor_map()

    
    # Filter for only female actors
    female_actor_map = {actor_id: actor_data for actor_id, actor_data in all_actor_map.items() if actor_data[2] == '1'}
    

    # Get roles for female actors in cumulative edges
    my_cast_map = get_roles_for_edges(female_actor_map, cumulative_edge_list)
    
    # print('role num', len(my_cast_map))
    # print('edge num', len(cumulative_edge_list))

    # Export the data to CSV files
    export_list(test_edge_file_name, cumulative_edge_list, edge_header)
    export_list(test_cast_file_name, list(my_cast_map.values()), person_header)

    
    # Add nodes for female actors
    for actor_id, actor_data in my_cast_map.items():
        # print(actor_data)
        G.add_node(actor_id, label=actor_data[1])  # Assuming 'Label' is the actor's name
    

    # Add edges between female actors
    for edge in cumulative_edge_list:
        actor1_id, actor2_id = edge[0], edge[1]
        actor1_data, actor2_data = my_cast_map.get(actor1_id), my_cast_map.get(actor2_id)
        
        if actor1_data is not None and actor2_data is not None and actor1_data[2] == '1' and actor2_data[2] == '1':  # Both actors are female
            G.add_edge(actor1_id, actor2_id)

    
    # Export the network graph to a GEXF file
    # gephi_file_name = 'actor_actor_network_1980_2000_female.gexf'
    # nx.write_gexf(G, gephi_file_name)
    # print(f'Graph exported to {gephi_file_name}')

    return G


################## CENTRALITY MEASURES ####################



def get_degree(G, actor_name):
    
    degree_dict = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_dict, 'degree')

    
    for k, v in degree_dict.items():
        if((G.nodes[k])['label'] == actor_name):
            print("Got degree")
            return(v)        




def get_eigenvector_centrality(G, actor_name):
    

    eigen_dict = nx.eigenvector_centrality(G)
    nx.set_node_attributes(G, eigen_dict, 'eigen_centrality')

    for k, v in eigen_dict.items():
        if((G.nodes[k])['label'] == actor_name):
            print("Got eigenvector centrality")
            return(v)     



def get_betweenness(G, actor_name):
    
    betweenness_dict = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, betweenness_dict, 'betweenness')
    

    for k, v in betweenness_dict.items():
        if((G.nodes[k])['label'] == actor_name):
            print("Got betweenness")
            return(v)           


    
def get_stats(G, actor_name):
    deg = get_degree(G, actor_name)
    bet = get_betweenness(G, actor_name)
    eigen = get_eigenvector_centrality(G, actor_name)
    # print("Degree: ", deg, ", Betweenness: ", "{:.4f}".format(bet), ", Eigenvector Centrality: ", "{:.5f}".format(eigen))
    return deg, bet, eigen


def get_avg_centralities(G):
    
    # Get average degree
    degree_dict = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_dict, 'degree')
    avg_deg = 0
    total = 0
    
    for v in degree_dict.values():
        avg_deg += v
        total+= 1
    
    avg_deg = avg_deg/total

    # Get average betweenness
    betweenness_dict = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, betweenness_dict, 'degree')
    avg_bet = 0
    total_bet = 0
    
    for v in betweenness_dict.values():
        avg_bet += v
        total_bet += 1
    
    avg_bet = avg_bet/total_bet

    # Get average eigenvector
    eigenvector_dict = nx.eigenvector_centrality(G, max_iter=500)
    nx.set_node_attributes(G, eigenvector_dict, 'degree')
    avg_eigen = 0
    total_eigen = 0
    
    for v in eigenvector_dict.values():
        avg_eigen += v
        total_eigen += 1
    
    avg_eigen = avg_eigen/total_eigen

    return(avg_deg, avg_bet, avg_eigen)
 
    





def main():
    # Create a network graph
    
    G = nx.Graph()


    method = input("Run on terminal? ")
    if method == "yes":
        # actor_name = input("Enter the name of the actor you wish to analyze: ")
        # years_string = input("Enter the set of years of networks you wish to analyze (separate by spaces): ")
        year1 = input("Enter a start year: ")
        year2 = input("Enter an end year: ")
        # years = [int(num_str) for num_str in years_string.split()]

        centralities_over_time = {
            "degree": [],
            "betweenness": [],
            "eigenvector": []
        }

        years = []
        

        for i in range(int(year1), int(year2)+1):
            movie_network = create_female_network(G, i)
            print("Getting stats for: ", i)
            # deg, bet, eigen = get_stats(movie_network, actor_name)
            deg, bet, eigen = get_avg_centralities(movie_network)
            centralities_over_time["degree"].append(deg)
            centralities_over_time["betweenness"].append(bet)
            centralities_over_time["eigenvector"].append(eigen)
            years.append(i) 
            # print("centralities over time (degree): ", centralities_over_time["degree"])
            # print("centralities over time (betweenness): ", centralities_over_time["betweenness"])
            # print("centralities over time (eigenvector): ", centralities_over_time["eigenvector"])


            
    # Plotting
        plt.figure(figsize=(6, 4))
        plt.plot(years, centralities_over_time["degree"], 'ro', label="Degree Centrality")
        plt.xlabel("Year")
        plt.ylabel("Degree Centrality")
        plt.title(f"Degree Centrality over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(years, centralities_over_time['betweenness'], 'bo', label="Betweenness Centrality")
        plt.xlabel("Year")
        plt.ylabel("Betweenness Centrality")
        plt.title(f"Betweenness Centrality over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(years, centralities_over_time['eigenvector'], 'go', label="Eigenvector Centrality")
        plt.xlabel("Year")
        plt.ylabel("Eigenvector Centrality")
        plt.title(f"Eigenvector Centrality over Time")
        plt.legend()
        plt.grid(True)
        plt.show()    
        

if __name__ == '__main__':
    main()


