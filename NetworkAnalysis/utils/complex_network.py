import itertools

import networkx as nx
from tqdm import tqdm
from utils import load_data, cal_average_path_length
import random


class ComplexNetwork:
    def __init__(self):
        self.raw_data = load_data()
        self.network = self.__generate_network()

    def __generate_network(self):
        data = self.raw_data["result"]
        network = nx.DiGraph(name="WYY")
        nodes = [user["id"] for user in data]
        network.add_nodes_from(nodes)
        for user in tqdm(data):
            follows = user['follows'][:]
            for follow_id in user['follows']:
                if follow_id not in nodes:
                    follows.remove(follow_id)
            edges = itertools.product([user["id"]], follows)
            network.add_edges_from(edges)
        return network

    def get_network_params(self):
        net_clustering_coefficient = nx.average_clustering(self.network)
        net_coreness = max(nx.core_number(self.network).values())
        average_shortest_path_length = round(cal_average_path_length(nx.shortest_paths.shortest_path(self.network)), 3)
        return {
            'net_clustering_coefficient': round(net_clustering_coefficient, 3),
            'net_coreness': net_coreness,
            'average_shortest_path_length': average_shortest_path_length,
        }

    def generate_graph_data(self):
        graph_data = {}
        nodes = []
        edges = []
        degree_map = nx.degree(self.network)
        name_map = self.raw_data["name_map"]
        for node_id in self.network.nodes:
            temp = {"id": node_id, "node_name": name_map[node_id], "node_degree": degree_map[node_id]}
            # temp = {"id": str(node_id)}
            nodes.append(temp)
        for edge in self.network.edges:
            temp = {"source": str(edge[0]), "target": str(edge[1]), "source_name": name_map[edge[0]],
                    "target_name": name_map[edge[1]]}
            edges.append(temp)

        # print(len(nodes), len(edges))
        graph_data["nodes"] = nodes
        graph_data["edges"] = edges
        return graph_data

    def net_attack(self, method="random"):
        method = method.lower()
        assert method == "random" or method == "intention"
        # before_shortest_avg_path = nx.average_shortest_path_length(self.network)
        before_max_connection = len(max(nx.weakly_connected_components(self.network), key=len))

        if method == "random":
            attack_node = random.choice(list(self.network.nodes))
        elif method == "intention":
            node_list = self.network.degree
            node_list = sorted(list(node_list), key=lambda x: -x[1])
            # print(node_list)
            attack_node = node_list[0][0]
        else:
            raise Exception("Wrong attack operation!")
        self.network.remove_node(attack_node)

        # after_shortest_avg_path = nx.average_shortest_path_length(self.network)
        after_max_connection = len(max(nx.weakly_connected_components(self.network), key=len))
        return {
            # "before_shortest_avg_path": before_shortest_avg_path,
            # "after_shortest_avg_path": after_shortest_avg_path,
            "removed_node": attack_node,
            "connection_ratio": after_max_connection / before_max_connection
        }

    def __find_max_connection(self):
        pass

    def node_distribution(self):
        node_list = self.network.degree
        node_dis_dic = {}
        for _, degree in list(node_list):
            if str(degree) in node_dis_dic.keys():
                node_dis_dic[str(degree)] += 1
            else:
                node_dis_dic[str(degree)] = 1
        x_labels = list(node_dis_dic.keys())
        y_labels = list(node_dis_dic.values())
        return {
            "x_axis_name": "node degree",
            "x_axis_label": x_labels,
            "y_axis_name": "count",
            "graph_data": y_labels,
        }

    def raw_repr(self):
        print(self.raw_data.keys())

    def get_node_params(self, node_id):
        node_clustering_coefficient = nx.clustering(self.network, node_id)
        node_degree = nx.degree(self.network, node_id)
        node_coreness = nx.core_number(self.network)[node_id]
        node_betweenness = nx.betweenness_centrality(self.network)[node_id]
        return {
            'node_clustering_coefficient': round(node_clustering_coefficient, 5),
            'node_degree': node_degree,
            'node_coreness': node_coreness,
            'node_betweenness': round(node_betweenness, 5),
        }

    def get_edge_params(self, edge_key):
        edge_betweenness = nx.edge_betweenness_centrality(self.network)[edge_key]
        return {
            'edge_betweenness': round(edge_betweenness, 5),
        }


if __name__ == "__main__":
    G = ComplexNetwork()
    # print(G.node_distribution())
    # print(G.get_network_params())
    # p = G.get_node_params(1918244613)
    # print(p)
    print(G.net_attack(method="intention"))
    print(G.net_attack(method="intention"))
    # print(G.net_attack())
    # p = G.get_node_params(1918244613)
    # print(p)
    # print(G.network.degree)
    # print(G.node_distribution())
