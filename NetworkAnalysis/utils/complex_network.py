import itertools

import networkx as nx
from tqdm import tqdm
from .utils import load_data, cal_average_path_length


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
            temp = {"source": str(edge[0]), "target": str(edge[1]), "source_name": name_map[edge[0]], "target_name": name_map[edge[1]]}
            edges.append(temp)

        # print(len(nodes), len(edges))
        graph_data["nodes"] = nodes
        graph_data["edges"] = edges
        return graph_data

    def attack(self):
        pass

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
    cn = ComplexNetwork()
    cn.get_network_params()
    p = cn.get_node_params(1918244613)
    print(p)