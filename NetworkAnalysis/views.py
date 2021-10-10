import json

from .utils.complex_network import ComplexNetwork
from django.http import JsonResponse

# global network
network = ComplexNetwork()


def get_graph_data(request):
    graph_data = network.generate_graph_data()
    response = {'data': graph_data, 'code': 20000}
    return JsonResponse(response)


def get_network_attributes(request):
    net_attributes = network.get_network_params()
    response = {'data': net_attributes, 'code': 20000}
    return JsonResponse(response)


def get_node_attributes(request):
    request_data = json.loads(request.body)
    node_id = int(request_data['node_id'])
    node_attributes = network.get_node_params(node_id)
    response = {'data': node_attributes, 'code': 20000}
    return JsonResponse(response)


def get_edge_attributes(request):
    request_data = json.loads(request.body)
    source_id = int(request_data['source_id'])
    target_id = int(request_data['target_id'])
    edge_attributes = network.get_edge_params((source_id, target_id))
    response = {'data': edge_attributes, 'code': 20000}
    return JsonResponse(response)


def get_node_distribution_data(request):
    node_distribution = network.node_distribution()
    response = {"data": node_distribution, "code": 20000}
    return JsonResponse(response)


def attack_network(request):
    request_data = json.loads(request.body)
    attack_method = request_data["method"]  # 前端给出攻击方式
    evaluation = network.net_attack(method=attack_method)
    new_net_attributes = network.get_network_params()
    response = {'data': {
        "new_network": new_net_attributes,
        "robust_evaluation": evaluation,
    },
        'code': 20000}
    return JsonResponse(response)
