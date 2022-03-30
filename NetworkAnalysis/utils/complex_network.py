import itertools
import random
from collections import Counter

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community

from tqdm import tqdm

from .utils import load_data, cal_average_path_length, get_combo_name, get_combos_arr
from .influence_maximization import unweighted_ic, weighted_ic


class ComplexNetwork:
    def __init__(self):
        self.raw_data = load_data()
        self.network = self.__generate_network()
        self.communinty_method_map = {
            'girvan_newman': self.get_community_by_girvan_newman,
            'greedy_modularity': self.get_community_by_greedy_modularity,
            'spectral_partition': self.get_community_by_spectral_partition,
        }

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

    def retrieve(self):
        self.network = self.__generate_network()  # 重置数据

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

    def net_attack(self, method="random", attack_times=1):
        if float(attack_times) < 1:
            attack_times = round(float(attack_times) * len(self.network.nodes))
        elif int(attack_times) >= len(self.network.nodes):
            attack_times = len(self.network.nodes) - 1
        else:
            attack_times = int(attack_times)
        # before_shortest_avg_path = nx.average_shortest_path_length(self.network)
        before_max_connection = len(max(nx.weakly_connected_components(self.network), key=len))

        if method == "random":
            attacked_nodes = random.sample(list(self.network.nodes), attack_times)
        elif method == "intention":
            node_list = self.network.degree
            node_list = sorted(list(node_list), key=lambda x: -x[1])
            # print(node_list)
            attacked_nodes = [i[0] for i in node_list[:attack_times]]
        else:
            raise Exception("Wrong attack operation!")

        print(attacked_nodes)
        attacked_nodes_list = []
        for each_node in attacked_nodes:
            tmp_dic = {
                "attacked_node_name": self.raw_data["name_map"][each_node],
                "attacked_node_degree": nx.degree(self.network)[each_node]
            }
            attacked_nodes_list.append(tmp_dic)

        self.network.remove_nodes_from(attacked_nodes)

        # after_shortest_avg_path = nx.average_shortest_path_length(self.network)
        after_max_connection = len(max(nx.weakly_connected_components(self.network), key=len))
        return {
            # "before_shortest_avg_path": before_shortest_avg_path,
            # "after_shortest_avg_path": after_shortest_avg_path,
            "attacked_nodes": attacked_nodes_list,
            "connection_ratio": round(after_max_connection / before_max_connection, 3)
        }

    def __find_max_connection(self):
        pass

    def get_node_distribution(self):
        counts = Counter(d for n, d in self.network.degree())
        x_labels = list(range(max(counts) + 1))
        y_labels = [counts.get(i, 0) for i in x_labels]
        return {
            "x_axis_name": "Node Degree",
            "x_axis_label": x_labels,
            "y_axis_name": "Count",
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

    def get_community_by_girvan_newman(self, community_num):
        """
        girvan_newman方法，社区检测
        @params community_num 期望划分成的社区数量
        """
        communities_generator = community.girvan_newman(self.network)
        for com in itertools.islice(communities_generator, community_num):
            if len(com) == community_num:
                res = tuple(sorted(c) for c in com)
                break
        return res

    def get_community_by_greedy_modularity(self, community_num):
        """
        greedy modularity方法，社区检测
        @params community_num 期望划分成的社区数量
        """
        communities = community.greedy_modularity_communities(self.network, best_n=community_num)
        res = []
        for com in communities:
            res.append(list(com))
        return res

    def get_community_by_spectral_partition(self, community_num=2):
        """
        naive spectral clustering方法 社区检测
        """
        adj_matrix = nx.to_numpy_array(self.network)
        deg_matrix = np.sum(adj_matrix, axis=1)
        lap_matrix = np.diag(deg_matrix) - adj_matrix

        eigen_values, eigen_vectors = np.linalg.eig(lap_matrix)
        lambda2_pos = list(eigen_values).index(sorted(eigen_values)[1])

        v = eigen_vectors.T[lambda2_pos]
        pos = []
        neg = []
        # 根据0二分
        for idx, val in zip(range(len(v)), v):
            if val > 0:
                pos.append(idx)
            else:
                neg.append(idx)
        pos_id = [list(self.network.nodes)[i] for i in pos]
        neg_id = [list(self.network.nodes)[i] for i in neg]
        return [pos_id, neg_id]

    def generate_graph_data_with_community(self, community_num, method_name):
        method = self.communinty_method_map[method_name]
        communities = method(community_num)

        graph_data = {}
        nodes = []
        edges = []
        degree_map = nx.degree(self.network)
        name_map = self.raw_data["name_map"]
        for node_id in self.network.nodes:
            temp = {
                "id": str(node_id),
                "node_name": name_map[node_id],
                "node_degree": degree_map[node_id],
                "comboId": get_combo_name(communities, node_id)
            }
            # temp = {"id": str(node_id)}
            nodes.append(temp)
        for edge in self.network.edges:
            temp = {"source": str(edge[0]), "target": str(edge[1]), "source_name": name_map[edge[0]],
                    "target_name": name_map[edge[1]]}
            edges.append(temp)

        # print(len(nodes), len(edges))
        graph_data["nodes"] = nodes
        graph_data["edges"] = edges
        graph_data["combos"] = get_combos_arr(len(communities))
        return graph_data

    def greedy_ic(self, num_of_seed, case='unweighted', epochs=5, p=0.5):
        """
        Influence Maximization 贪心算法
        :param num_of_seed: 待选定的种子节点数量
        :param case: 1-unweighted, 2-weighted
        :param p: 无权重独立级联模型传播概率
        :param epochs: 每轮迭代次数
        :return: 种子节点集合
        """
        seed = []
        result = []
        nodes = list(self.network.nodes)
        ic_func = unweighted_ic if case == 'unweighted' else weighted_ic
        for i in range(num_of_seed):
            best_inf, best_inf_node = 0, 0
            for v in set(nodes) - set(seed):
                # 计算边际影响增益
                inf_score = ic_func(self.network, seed + [v], epochs, p)
                if inf_score > best_inf:
                    best_inf, best_inf_node = inf_score, v
            seed.append(best_inf_node)
            result.append((best_inf_node, best_inf))  # 总增益
        return result

    def heuristic_ic(self, num_of_seed, case='unweighted', epochs=5, p=0.5, heuristic=1):
        """
        Influence Maximization 贪心算法
        :param num_of_seed: 待选定的种子节点数量
        :param case: 1-unweighted, 2-weighted
        :param p: 无权重独立级联模型传播概率
        :param epochs: 每轮迭代模拟的次数
        :param heuristic: 1-degree_centrality, 2-distance_centrality, 3-random_nodes
        :return: 种子节点集合
        """
        seed = []
        result = []
        candidates = []
        nodes = list(self.network.nodes)
        ic_func = unweighted_ic if case == 'unweighted' else weighted_ic
        if heuristic == 1:
            # 选择degree最大的节点
            node_list = sorted(list(nx.degree(self.network)), key=lambda x: -x[1])
            candidates = [i[0] for i in node_list[:num_of_seed]]
        elif heuristic == 2:
            # 选择最靠近网络中心的节点，根据pagerank或hits得分排序
            node_list = sorted(nx.pagerank(self.network).items(), key=lambda kv: -kv[1])
            candidates = [i[0] for i in node_list[:num_of_seed]]
        else:
            # 随机选择节点
            node_list = random.sample(list(nx.degree(self.network)), num_of_seed)
            candidates = [i[0] for i in node_list]

        for v in candidates:
            # 计算边际影响增益
            inf_score = ic_func(self.network, seed + [v], epochs, p)
            seed.append(v)
            result.append((v, inf_score))  # 总增益
        return result

    def get_influence_comparison_data(self, case, epochs=5, p=0.5):
        num_of_seed = 30
        legend = ['greedy', 'high degree', 'central', 'random']
        x_axis_label = list(range(1, 31))
        if case == 'unweighted':
            title = 'p = ' + str(p)
        else:
            title = 'p = 1/deg(node)'
        # 生成数据
        graph_data = []
        # greedy
        res_greedy = self.greedy_ic(num_of_seed, case=case, epochs=epochs, p=p)
        graph_data.append({
            'name': 'greedy',
            'type': 'line',
            'data': [item[1] for item in res_greedy]
        })
        # heuristic
        res_degree = self.heuristic_ic(num_of_seed, case=case, epochs=epochs, p=p, heuristic=1)
        res_central = self.heuristic_ic(num_of_seed, case=case, epochs=epochs, p=p, heuristic=2)
        res_random = self.heuristic_ic(num_of_seed, case=case, epochs=epochs, p=p, heuristic=3)
        graph_data.append({
            'name': 'high degree',
            'type': 'line',
            'data': [item[1] for item in res_degree]
        })
        graph_data.append({
            'name': 'central',
            'type': 'line',
            'data': [item[1] for item in res_central]
        })
        graph_data.append({
            'name': 'random',
            'type': 'line',
            'data': [item[1] for item in res_random]
        })
        return {
            "x_axis_name": "Seed Set Size",
            "x_axis_label": x_axis_label,
            "y_axis_name": "Active Set Size",
            "legend": legend,
            "graph_data": graph_data,
        }


if __name__ == "__main__":
    G = ComplexNetwork()
    # print(G.node_distribution())
    # print(G.get_network_params())

    # g1 = community.girvan_newman(G.network)
    # for com in itertools.islice(g1, 4):
    #     c = tuple(sorted(c) for c in com)
    #
    g2 = community.greedy_modularity_communities(G.network, best_n=5)

    # g3 = G.get_community_by_spectral_partition()

    # res1 = G.heuristic_ic(4, heuristic=1, case=1)
    res2 = G.heuristic_ic(4, heuristic=2, case=1)
    # res3 = G.heuristic_ic(4, heuristic=3, case=1)
    # print(res1)
    print(res2)
    # print(res3)
