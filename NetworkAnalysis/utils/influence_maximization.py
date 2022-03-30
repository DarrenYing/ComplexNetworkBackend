"""
Influence Maximization
Independent Cascading Model
"""

import networkx as nx
from copy import deepcopy
import random
import tqdm


def unweighted_ic(graph, seed, epochs=5, p=0.5):
    """
    无权重IC模型
    :param graph: networkx的图
    :param seed: 激活节点或者激活节点集合
    :param p: 独立级联激活概率值
    :param epochs: 迭代次数
    :return: 被激活节点平均个数
    """
    count = 0
    for i in range(epochs):
        used_nodes = deepcopy(seed)
        activated_nodes = deepcopy(seed)
        tmp_seed = deepcopy(seed)
        cur_activated_nodes = []
        while tmp_seed:
            for v in tmp_seed:
                for w in nx.neighbors(graph, v):
                    if w not in used_nodes:
                        if random.random() < p:
                            cur_activated_nodes.append(w)
                        used_nodes.append(w)
            tmp_seed = cur_activated_nodes
            activated_nodes.extend(cur_activated_nodes)
            cur_activated_nodes = []
        count += len(activated_nodes)

    return count / epochs


def weighted_ic(graph, seed, epochs=5, *args):
    """
    带权重IC模型
    :param graph: networkx的图
    :param seed: 激活节点或者激活节点集合
    :param epochs: 迭代次数
    :param args: 占位，和unweighted_ic接口一致
    :return: 被激活节点平均个数
    """
    count = 0
    for i in range(epochs):
        used_nodes = deepcopy(seed)
        activated_nodes = deepcopy(seed)
        tmp_seed = deepcopy(seed)
        cur_activated_nodes = []
        while tmp_seed:
            for v in tmp_seed:
                for w in nx.neighbors(graph, v):
                    if w not in used_nodes:
                        if random.random() * nx.degree(graph, w) < 1:
                            cur_activated_nodes.append(w)
                        used_nodes.append(w)
            tmp_seed = cur_activated_nodes
            activated_nodes.extend(cur_activated_nodes)
            cur_activated_nodes = []
        count += len(activated_nodes)

    return count / epochs
