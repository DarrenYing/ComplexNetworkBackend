from django.urls import path

from . import views

urlpatterns = [
    path('graph-data', views.get_graph_data),
    path('network-attr', views.get_network_attributes),
    path('node-attr', views.get_node_attributes),
    path('edge-attr', views.get_edge_attributes),
    path('node-distribution', views.get_node_distribution_data),
    path('network-attack', views.attack_network),
    path('network-retrieve', views.retrieve_network),
]