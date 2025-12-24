## Lazy NetworkX proxy
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None

if TYPE_CHECKING:
    from ..graph import AnnNet


class _LazyNXProxyAutogen:
    def AmbiguousSolution(self, *args, **kwargs):
        return self.__getattr__("AmbiguousSolution")(*args, **kwargs)

    def ArborescenceIterator(self, *args, **kwargs):
        return self.__getattr__("ArborescenceIterator")(*args, **kwargs)

    def DiGraph(self, *args, **kwargs):
        return self.__getattr__("DiGraph")(*args, **kwargs)

    def EdgePartition(self, *args, **kwargs):
        return self.__getattr__("EdgePartition")(*args, **kwargs)

    def ExceededMaxIterations(self, *args, **kwargs):
        return self.__getattr__("ExceededMaxIterations")(*args, **kwargs)

    def Graph(self, *args, **kwargs):
        return self.__getattr__("Graph")(*args, **kwargs)

    def GraphMLReader(self, *args, **kwargs):
        return self.__getattr__("GraphMLReader")(*args, **kwargs)

    def GraphMLWriter(self, *args, **kwargs):
        return self.__getattr__("GraphMLWriter")(*args, **kwargs)

    def HasACycle(self, *args, **kwargs):
        return self.__getattr__("HasACycle")(*args, **kwargs)

    def LCF_graph(self, *args, **kwargs):
        return self.__getattr__("LCF_graph")(*args, **kwargs)

    def LFR_benchmark_graph(self, *args, **kwargs):
        return self.__getattr__("LFR_benchmark_graph")(*args, **kwargs)

    def MultiDiGraph(self, *args, **kwargs):
        return self.__getattr__("MultiDiGraph")(*args, **kwargs)

    def MultiGraph(self, *args, **kwargs):
        return self.__getattr__("MultiGraph")(*args, **kwargs)

    def NetworkXAlgorithmError(self, *args, **kwargs):
        return self.__getattr__("NetworkXAlgorithmError")(*args, **kwargs)

    def NetworkXError(self, *args, **kwargs):
        return self.__getattr__("NetworkXError")(*args, **kwargs)

    def NetworkXException(self, *args, **kwargs):
        return self.__getattr__("NetworkXException")(*args, **kwargs)

    def NetworkXNoCycle(self, *args, **kwargs):
        return self.__getattr__("NetworkXNoCycle")(*args, **kwargs)

    def NetworkXNoPath(self, *args, **kwargs):
        return self.__getattr__("NetworkXNoPath")(*args, **kwargs)

    def NetworkXNotImplemented(self, *args, **kwargs):
        return self.__getattr__("NetworkXNotImplemented")(*args, **kwargs)

    def NetworkXPointlessConcept(self, *args, **kwargs):
        return self.__getattr__("NetworkXPointlessConcept")(*args, **kwargs)

    def NetworkXTreewidthBoundExceeded(self, *args, **kwargs):
        return self.__getattr__("NetworkXTreewidthBoundExceeded")(*args, **kwargs)

    def NetworkXUnbounded(self, *args, **kwargs):
        return self.__getattr__("NetworkXUnbounded")(*args, **kwargs)

    def NetworkXUnfeasible(self, *args, **kwargs):
        return self.__getattr__("NetworkXUnfeasible")(*args, **kwargs)

    def NodeNotFound(self, *args, **kwargs):
        return self.__getattr__("NodeNotFound")(*args, **kwargs)

    def NotATree(self, *args, **kwargs):
        return self.__getattr__("NotATree")(*args, **kwargs)

    def PlanarEmbedding(self, *args, **kwargs):
        return self.__getattr__("PlanarEmbedding")(*args, **kwargs)

    def PowerIterationFailedConvergence(self, *args, **kwargs):
        return self.__getattr__("PowerIterationFailedConvergence")(*args, **kwargs)

    def SpanningTreeIterator(self, *args, **kwargs):
        return self.__getattr__("SpanningTreeIterator")(*args, **kwargs)

    def adamic_adar_index(self, *args, **kwargs):
        return self.__getattr__("adamic_adar_index")(*args, **kwargs)

    def add_cycle(self, *args, **kwargs):
        return self.__getattr__("add_cycle")(*args, **kwargs)

    def add_path(self, *args, **kwargs):
        return self.__getattr__("add_path")(*args, **kwargs)

    def add_star(self, *args, **kwargs):
        return self.__getattr__("add_star")(*args, **kwargs)

    def adjacency_data(self, *args, **kwargs):
        return self.__getattr__("adjacency_data")(*args, **kwargs)

    def adjacency_graph(self, *args, **kwargs):
        return self.__getattr__("adjacency_graph")(*args, **kwargs)

    def adjacency_matrix(self, *args, **kwargs):
        return self.__getattr__("adjacency_matrix")(*args, **kwargs)

    def adjacency_spectrum(self, *args, **kwargs):
        return self.__getattr__("adjacency_spectrum")(*args, **kwargs)

    def algebraic_connectivity(self, *args, **kwargs):
        return self.__getattr__("algebraic_connectivity")(*args, **kwargs)

    def all_neighbors(self, *args, **kwargs):
        return self.__getattr__("all_neighbors")(*args, **kwargs)

    def all_node_cuts(self, *args, **kwargs):
        return self.__getattr__("all_node_cuts")(*args, **kwargs)

    def all_pairs_all_shortest_paths(self, *args, **kwargs):
        return self.__getattr__("all_pairs_all_shortest_paths")(*args, **kwargs)

    def all_pairs_bellman_ford_path(self, *args, **kwargs):
        return self.__getattr__("all_pairs_bellman_ford_path")(*args, **kwargs)

    def all_pairs_bellman_ford_path_length(self, *args, **kwargs):
        return self.__getattr__("all_pairs_bellman_ford_path_length")(*args, **kwargs)

    def all_pairs_dijkstra(self, *args, **kwargs):
        return self.__getattr__("all_pairs_dijkstra")(*args, **kwargs)

    def all_pairs_dijkstra_path(self, *args, **kwargs):
        return self.__getattr__("all_pairs_dijkstra_path")(*args, **kwargs)

    def all_pairs_dijkstra_path_length(self, *args, **kwargs):
        return self.__getattr__("all_pairs_dijkstra_path_length")(*args, **kwargs)

    def all_pairs_lowest_common_ancestor(self, *args, **kwargs):
        return self.__getattr__("all_pairs_lowest_common_ancestor")(*args, **kwargs)

    def all_pairs_node_connectivity(self, *args, **kwargs):
        return self.__getattr__("all_pairs_node_connectivity")(*args, **kwargs)

    def all_pairs_shortest_path(self, *args, **kwargs):
        return self.__getattr__("all_pairs_shortest_path")(*args, **kwargs)

    def all_pairs_shortest_path_length(self, *args, **kwargs):
        return self.__getattr__("all_pairs_shortest_path_length")(*args, **kwargs)

    def all_shortest_paths(self, *args, **kwargs):
        return self.__getattr__("all_shortest_paths")(*args, **kwargs)

    def all_simple_edge_paths(self, *args, **kwargs):
        return self.__getattr__("all_simple_edge_paths")(*args, **kwargs)

    def all_simple_paths(self, *args, **kwargs):
        return self.__getattr__("all_simple_paths")(*args, **kwargs)

    def all_topological_sorts(self, *args, **kwargs):
        return self.__getattr__("all_topological_sorts")(*args, **kwargs)

    def all_triads(self, *args, **kwargs):
        return self.__getattr__("all_triads")(*args, **kwargs)

    def alternating_havel_hakimi_graph(self, *args, **kwargs):
        return self.__getattr__("alternating_havel_hakimi_graph")(*args, **kwargs)

    def ancestors(self, *args, **kwargs):
        return self.__getattr__("ancestors")(*args, **kwargs)

    def antichains(self, *args, **kwargs):
        return self.__getattr__("antichains")(*args, **kwargs)

    def apply_matplotlib_colors(self, *args, **kwargs):
        return self.__getattr__("apply_matplotlib_colors")(*args, **kwargs)

    def approximate_current_flow_betweenness_centrality(self, *args, **kwargs):
        return self.__getattr__("approximate_current_flow_betweenness_centrality")(*args, **kwargs)

    def arf_layout(self, *args, **kwargs):
        return self.__getattr__("arf_layout")(*args, **kwargs)

    def articulation_points(self, *args, **kwargs):
        return self.__getattr__("articulation_points")(*args, **kwargs)

    def asadpour_atsp(self, *args, **kwargs):
        return self.__getattr__("asadpour_atsp")(*args, **kwargs)

    def astar_path(self, *args, **kwargs):
        return self.__getattr__("astar_path")(*args, **kwargs)

    def astar_path_length(self, *args, **kwargs):
        return self.__getattr__("astar_path_length")(*args, **kwargs)

    def asyn_fluidc(self, *args, **kwargs):
        return self.__getattr__("asyn_fluidc")(*args, **kwargs)

    def asyn_lpa_communities(self, *args, **kwargs):
        return self.__getattr__("asyn_lpa_communities")(*args, **kwargs)

    def attr_matrix(self, *args, **kwargs):
        return self.__getattr__("attr_matrix")(*args, **kwargs)

    def attr_sparse_matrix(self, *args, **kwargs):
        return self.__getattr__("attr_sparse_matrix")(*args, **kwargs)

    def attracting_components(self, *args, **kwargs):
        return self.__getattr__("attracting_components")(*args, **kwargs)

    def attribute_assortativity_coefficient(self, *args, **kwargs):
        return self.__getattr__("attribute_assortativity_coefficient")(*args, **kwargs)

    def attribute_mixing_dict(self, *args, **kwargs):
        return self.__getattr__("attribute_mixing_dict")(*args, **kwargs)

    def attribute_mixing_matrix(self, *args, **kwargs):
        return self.__getattr__("attribute_mixing_matrix")(*args, **kwargs)

    def average_clustering(self, *args, **kwargs):
        return self.__getattr__("average_clustering")(*args, **kwargs)

    def average_degree_connectivity(self, *args, **kwargs):
        return self.__getattr__("average_degree_connectivity")(*args, **kwargs)

    def average_neighbor_degree(self, *args, **kwargs):
        return self.__getattr__("average_neighbor_degree")(*args, **kwargs)

    def average_node_connectivity(self, *args, **kwargs):
        return self.__getattr__("average_node_connectivity")(*args, **kwargs)

    def average_shortest_path_length(self, *args, **kwargs):
        return self.__getattr__("average_shortest_path_length")(*args, **kwargs)

    def balanced_tree(self, *args, **kwargs):
        return self.__getattr__("balanced_tree")(*args, **kwargs)

    def barabasi_albert_graph(self, *args, **kwargs):
        return self.__getattr__("barabasi_albert_graph")(*args, **kwargs)

    def barbell_graph(self, *args, **kwargs):
        return self.__getattr__("barbell_graph")(*args, **kwargs)

    def barycenter(self, *args, **kwargs):
        return self.__getattr__("barycenter")(*args, **kwargs)

    def bellman_ford_path(self, *args, **kwargs):
        return self.__getattr__("bellman_ford_path")(*args, **kwargs)

    def bellman_ford_path_length(self, *args, **kwargs):
        return self.__getattr__("bellman_ford_path_length")(*args, **kwargs)

    def bellman_ford_predecessor_and_distance(self, *args, **kwargs):
        return self.__getattr__("bellman_ford_predecessor_and_distance")(*args, **kwargs)

    def bethe_hessian_matrix(self, *args, **kwargs):
        return self.__getattr__("bethe_hessian_matrix")(*args, **kwargs)

    def bethe_hessian_spectrum(self, *args, **kwargs):
        return self.__getattr__("bethe_hessian_spectrum")(*args, **kwargs)

    def betweenness_centrality(self, *args, **kwargs):
        return self.__getattr__("betweenness_centrality")(*args, **kwargs)

    def betweenness_centrality_subset(self, *args, **kwargs):
        return self.__getattr__("betweenness_centrality_subset")(*args, **kwargs)

    def bfs_beam_edges(self, *args, **kwargs):
        return self.__getattr__("bfs_beam_edges")(*args, **kwargs)

    def bfs_edges(self, *args, **kwargs):
        return self.__getattr__("bfs_edges")(*args, **kwargs)

    def bfs_labeled_edges(self, *args, **kwargs):
        return self.__getattr__("bfs_labeled_edges")(*args, **kwargs)

    def bfs_layers(self, *args, **kwargs):
        return self.__getattr__("bfs_layers")(*args, **kwargs)

    def bfs_layout(self, *args, **kwargs):
        return self.__getattr__("bfs_layout")(*args, **kwargs)

    def bfs_predecessors(self, *args, **kwargs):
        return self.__getattr__("bfs_predecessors")(*args, **kwargs)

    def bfs_successors(self, *args, **kwargs):
        return self.__getattr__("bfs_successors")(*args, **kwargs)

    def bfs_tree(self, *args, **kwargs):
        return self.__getattr__("bfs_tree")(*args, **kwargs)

    def biadjacency_matrix(self, *args, **kwargs):
        return self.__getattr__("biadjacency_matrix")(*args, **kwargs)

    def biconnected_component_edges(self, *args, **kwargs):
        return self.__getattr__("biconnected_component_edges")(*args, **kwargs)

    def biconnected_components(self, *args, **kwargs):
        return self.__getattr__("biconnected_components")(*args, **kwargs)

    def bidirectional_dijkstra(self, *args, **kwargs):
        return self.__getattr__("bidirectional_dijkstra")(*args, **kwargs)

    def bidirectional_shortest_path(self, *args, **kwargs):
        return self.__getattr__("bidirectional_shortest_path")(*args, **kwargs)

    def binomial_graph(self, *args, **kwargs):
        return self.__getattr__("binomial_graph")(*args, **kwargs)

    def binomial_tree(self, *args, **kwargs):
        return self.__getattr__("binomial_tree")(*args, **kwargs)

    def bipartite_layout(self, *args, **kwargs):
        return self.__getattr__("bipartite_layout")(*args, **kwargs)

    def birank(self, *args, **kwargs):
        return self.__getattr__("birank")(*args, **kwargs)

    def boundary_expansion(self, *args, **kwargs):
        return self.__getattr__("boundary_expansion")(*args, **kwargs)

    def boykov_kolmogorov(self, *args, **kwargs):
        return self.__getattr__("boykov_kolmogorov")(*args, **kwargs)

    def bridges(self, *args, **kwargs):
        return self.__getattr__("bridges")(*args, **kwargs)

    def build_flow_dict(self, *args, **kwargs):
        return self.__getattr__("build_flow_dict")(*args, **kwargs)

    def build_residual_network(self, *args, **kwargs):
        return self.__getattr__("build_residual_network")(*args, **kwargs)

    def bull_graph(self, *args, **kwargs):
        return self.__getattr__("bull_graph")(*args, **kwargs)

    def capacity_scaling(self, *args, **kwargs):
        return self.__getattr__("capacity_scaling")(*args, **kwargs)

    def cartesian_product(self, *args, **kwargs):
        return self.__getattr__("cartesian_product")(*args, **kwargs)

    def caveman_graph(self, *args, **kwargs):
        return self.__getattr__("caveman_graph")(*args, **kwargs)

    def cd_index(self, *args, **kwargs):
        return self.__getattr__("cd_index")(*args, **kwargs)

    def center(self, *args, **kwargs):
        return self.__getattr__("center")(*args, **kwargs)

    def chain_decomposition(self, *args, **kwargs):
        return self.__getattr__("chain_decomposition")(*args, **kwargs)

    def check_planarity(self, *args, **kwargs):
        return self.__getattr__("check_planarity")(*args, **kwargs)

    def chordal_cycle_graph(self, *args, **kwargs):
        return self.__getattr__("chordal_cycle_graph")(*args, **kwargs)

    def chordal_graph_cliques(self, *args, **kwargs):
        return self.__getattr__("chordal_graph_cliques")(*args, **kwargs)

    def chordal_graph_treewidth(self, *args, **kwargs):
        return self.__getattr__("chordal_graph_treewidth")(*args, **kwargs)

    def chordless_cycles(self, *args, **kwargs):
        return self.__getattr__("chordless_cycles")(*args, **kwargs)

    def christofides(self, *args, **kwargs):
        return self.__getattr__("christofides")(*args, **kwargs)

    def chromatic_polynomial(self, *args, **kwargs):
        return self.__getattr__("chromatic_polynomial")(*args, **kwargs)

    def chvatal_graph(self, *args, **kwargs):
        return self.__getattr__("chvatal_graph")(*args, **kwargs)

    def circulant_graph(self, *args, **kwargs):
        return self.__getattr__("circulant_graph")(*args, **kwargs)

    def circular_ladder_graph(self, *args, **kwargs):
        return self.__getattr__("circular_ladder_graph")(*args, **kwargs)

    def circular_layout(self, *args, **kwargs):
        return self.__getattr__("circular_layout")(*args, **kwargs)

    def clique_removal(self, *args, **kwargs):
        return self.__getattr__("clique_removal")(*args, **kwargs)

    def closeness_centrality(self, *args, **kwargs):
        return self.__getattr__("closeness_centrality")(*args, **kwargs)

    def closeness_vitality(self, *args, **kwargs):
        return self.__getattr__("closeness_vitality")(*args, **kwargs)

    def clustering(self, *args, **kwargs):
        return self.__getattr__("clustering")(*args, **kwargs)

    def cn_soundarajan_hopcroft(self, *args, **kwargs):
        return self.__getattr__("cn_soundarajan_hopcroft")(*args, **kwargs)

    def collaboration_weighted_projected_graph(self, *args, **kwargs):
        return self.__getattr__("collaboration_weighted_projected_graph")(*args, **kwargs)

    def color(self, *args, **kwargs):
        return self.__getattr__("color")(*args, **kwargs)

    def combinatorial_embedding_to_pos(self, *args, **kwargs):
        return self.__getattr__("combinatorial_embedding_to_pos")(*args, **kwargs)

    def common_neighbor_centrality(self, *args, **kwargs):
        return self.__getattr__("common_neighbor_centrality")(*args, **kwargs)

    def common_neighbors(self, *args, **kwargs):
        return self.__getattr__("common_neighbors")(*args, **kwargs)

    def communicability(self, *args, **kwargs):
        return self.__getattr__("communicability")(*args, **kwargs)

    def communicability_betweenness_centrality(self, *args, **kwargs):
        return self.__getattr__("communicability_betweenness_centrality")(*args, **kwargs)

    def communicability_exp(self, *args, **kwargs):
        return self.__getattr__("communicability_exp")(*args, **kwargs)

    def complement(self, *args, **kwargs):
        return self.__getattr__("complement")(*args, **kwargs)

    def complete_bipartite_graph(self, *args, **kwargs):
        return self.__getattr__("complete_bipartite_graph")(*args, **kwargs)

    def complete_graph(self, *args, **kwargs):
        return self.__getattr__("complete_graph")(*args, **kwargs)

    def complete_multipartite_graph(self, *args, **kwargs):
        return self.__getattr__("complete_multipartite_graph")(*args, **kwargs)

    def complete_to_chordal_graph(self, *args, **kwargs):
        return self.__getattr__("complete_to_chordal_graph")(*args, **kwargs)

    def compose(self, *args, **kwargs):
        return self.__getattr__("compose")(*args, **kwargs)

    def compose_all(self, *args, **kwargs):
        return self.__getattr__("compose_all")(*args, **kwargs)

    def compute_v_structures(self, *args, **kwargs):
        return self.__getattr__("compute_v_structures")(*args, **kwargs)

    def condensation(self, *args, **kwargs):
        return self.__getattr__("condensation")(*args, **kwargs)

    def conductance(self, *args, **kwargs):
        return self.__getattr__("conductance")(*args, **kwargs)

    def config(self, *args, **kwargs):
        return self.__getattr__("config")(*args, **kwargs)

    def configuration_model(self, *args, **kwargs):
        return self.__getattr__("configuration_model")(*args, **kwargs)

    def connected_caveman_graph(self, *args, **kwargs):
        return self.__getattr__("connected_caveman_graph")(*args, **kwargs)

    def connected_components(self, *args, **kwargs):
        return self.__getattr__("connected_components")(*args, **kwargs)

    def connected_dominating_set(self, *args, **kwargs):
        return self.__getattr__("connected_dominating_set")(*args, **kwargs)

    def connected_double_edge_swap(self, *args, **kwargs):
        return self.__getattr__("connected_double_edge_swap")(*args, **kwargs)

    def connected_watts_strogatz_graph(self, *args, **kwargs):
        return self.__getattr__("connected_watts_strogatz_graph")(*args, **kwargs)

    def constraint(self, *args, **kwargs):
        return self.__getattr__("constraint")(*args, **kwargs)

    def contracted_edge(self, *args, **kwargs):
        return self.__getattr__("contracted_edge")(*args, **kwargs)

    def contracted_nodes(self, *args, **kwargs):
        return self.__getattr__("contracted_nodes")(*args, **kwargs)

    def convert_node_labels_to_integers(self, *args, **kwargs):
        return self.__getattr__("convert_node_labels_to_integers")(*args, **kwargs)

    def core_number(self, *args, **kwargs):
        return self.__getattr__("core_number")(*args, **kwargs)

    def corona_product(self, *args, **kwargs):
        return self.__getattr__("corona_product")(*args, **kwargs)

    def cost_of_flow(self, *args, **kwargs):
        return self.__getattr__("cost_of_flow")(*args, **kwargs)

    def could_be_isomorphic(self, *args, **kwargs):
        return self.__getattr__("could_be_isomorphic")(*args, **kwargs)

    def create_empty_copy(self, *args, **kwargs):
        return self.__getattr__("create_empty_copy")(*args, **kwargs)

    def cubical_graph(self, *args, **kwargs):
        return self.__getattr__("cubical_graph")(*args, **kwargs)

    def current_flow_betweenness_centrality(self, *args, **kwargs):
        return self.__getattr__("current_flow_betweenness_centrality")(*args, **kwargs)

    def current_flow_betweenness_centrality_subset(self, *args, **kwargs):
        return self.__getattr__("current_flow_betweenness_centrality_subset")(*args, **kwargs)

    def current_flow_closeness_centrality(self, *args, **kwargs):
        return self.__getattr__("current_flow_closeness_centrality")(*args, **kwargs)

    def cut_size(self, *args, **kwargs):
        return self.__getattr__("cut_size")(*args, **kwargs)

    def cycle_basis(self, *args, **kwargs):
        return self.__getattr__("cycle_basis")(*args, **kwargs)

    def cycle_graph(self, *args, **kwargs):
        return self.__getattr__("cycle_graph")(*args, **kwargs)

    def cytoscape_data(self, *args, **kwargs):
        return self.__getattr__("cytoscape_data")(*args, **kwargs)

    def cytoscape_graph(self, *args, **kwargs):
        return self.__getattr__("cytoscape_graph")(*args, **kwargs)

    def dag_longest_path(self, *args, **kwargs):
        return self.__getattr__("dag_longest_path")(*args, **kwargs)

    def dag_longest_path_length(self, *args, **kwargs):
        return self.__getattr__("dag_longest_path_length")(*args, **kwargs)

    def dag_to_branching(self, *args, **kwargs):
        return self.__getattr__("dag_to_branching")(*args, **kwargs)

    def davis_southern_women_graph(self, *args, **kwargs):
        return self.__getattr__("davis_southern_women_graph")(*args, **kwargs)

    def dedensify(self, *args, **kwargs):
        return self.__getattr__("dedensify")(*args, **kwargs)

    def degree(self, *args, **kwargs):
        return self.__getattr__("degree")(*args, **kwargs)

    def degree_assortativity_coefficient(self, *args, **kwargs):
        return self.__getattr__("degree_assortativity_coefficient")(*args, **kwargs)

    def degree_centrality(self, *args, **kwargs):
        return self.__getattr__("degree_centrality")(*args, **kwargs)

    def degree_histogram(self, *args, **kwargs):
        return self.__getattr__("degree_histogram")(*args, **kwargs)

    def degree_mixing_dict(self, *args, **kwargs):
        return self.__getattr__("degree_mixing_dict")(*args, **kwargs)

    def degree_mixing_matrix(self, *args, **kwargs):
        return self.__getattr__("degree_mixing_matrix")(*args, **kwargs)

    def degree_pearson_correlation_coefficient(self, *args, **kwargs):
        return self.__getattr__("degree_pearson_correlation_coefficient")(*args, **kwargs)

    def degree_sequence_tree(self, *args, **kwargs):
        return self.__getattr__("degree_sequence_tree")(*args, **kwargs)

    def degrees(self, *args, **kwargs):
        return self.__getattr__("degrees")(*args, **kwargs)

    def dense_gnm_random_graph(self, *args, **kwargs):
        return self.__getattr__("dense_gnm_random_graph")(*args, **kwargs)

    def densest_subgraph(self, *args, **kwargs):
        return self.__getattr__("densest_subgraph")(*args, **kwargs)

    def density(self, *args, **kwargs):
        return self.__getattr__("density")(*args, **kwargs)

    def desargues_graph(self, *args, **kwargs):
        return self.__getattr__("desargues_graph")(*args, **kwargs)

    def descendants(self, *args, **kwargs):
        return self.__getattr__("descendants")(*args, **kwargs)

    def descendants_at_distance(self, *args, **kwargs):
        return self.__getattr__("descendants_at_distance")(*args, **kwargs)

    def dfs_edges(self, *args, **kwargs):
        return self.__getattr__("dfs_edges")(*args, **kwargs)

    def dfs_labeled_edges(self, *args, **kwargs):
        return self.__getattr__("dfs_labeled_edges")(*args, **kwargs)

    def dfs_postorder_nodes(self, *args, **kwargs):
        return self.__getattr__("dfs_postorder_nodes")(*args, **kwargs)

    def dfs_predecessors(self, *args, **kwargs):
        return self.__getattr__("dfs_predecessors")(*args, **kwargs)

    def dfs_preorder_nodes(self, *args, **kwargs):
        return self.__getattr__("dfs_preorder_nodes")(*args, **kwargs)

    def dfs_successors(self, *args, **kwargs):
        return self.__getattr__("dfs_successors")(*args, **kwargs)

    def dfs_tree(self, *args, **kwargs):
        return self.__getattr__("dfs_tree")(*args, **kwargs)

    def diameter(self, *args, **kwargs):
        return self.__getattr__("diameter")(*args, **kwargs)

    def diamond_graph(self, *args, **kwargs):
        return self.__getattr__("diamond_graph")(*args, **kwargs)

    def difference(self, *args, **kwargs):
        return self.__getattr__("difference")(*args, **kwargs)

    def dijkstra_path(self, *args, **kwargs):
        return self.__getattr__("dijkstra_path")(*args, **kwargs)

    def dijkstra_path_length(self, *args, **kwargs):
        return self.__getattr__("dijkstra_path_length")(*args, **kwargs)

    def dijkstra_predecessor_and_distance(self, *args, **kwargs):
        return self.__getattr__("dijkstra_predecessor_and_distance")(*args, **kwargs)

    def dinitz(self, *args, **kwargs):
        return self.__getattr__("dinitz")(*args, **kwargs)

    def directed_combinatorial_laplacian_matrix(self, *args, **kwargs):
        return self.__getattr__("directed_combinatorial_laplacian_matrix")(*args, **kwargs)

    def directed_configuration_model(self, *args, **kwargs):
        return self.__getattr__("directed_configuration_model")(*args, **kwargs)

    def directed_edge_swap(self, *args, **kwargs):
        return self.__getattr__("directed_edge_swap")(*args, **kwargs)

    def directed_havel_hakimi_graph(self, *args, **kwargs):
        return self.__getattr__("directed_havel_hakimi_graph")(*args, **kwargs)

    def directed_joint_degree_graph(self, *args, **kwargs):
        return self.__getattr__("directed_joint_degree_graph")(*args, **kwargs)

    def directed_laplacian_matrix(self, *args, **kwargs):
        return self.__getattr__("directed_laplacian_matrix")(*args, **kwargs)

    def directed_modularity_matrix(self, *args, **kwargs):
        return self.__getattr__("directed_modularity_matrix")(*args, **kwargs)

    def disjoint_union(self, *args, **kwargs):
        return self.__getattr__("disjoint_union")(*args, **kwargs)

    def disjoint_union_all(self, *args, **kwargs):
        return self.__getattr__("disjoint_union_all")(*args, **kwargs)

    def dispersion(self, *args, **kwargs):
        return self.__getattr__("dispersion")(*args, **kwargs)

    def display(self, *args, **kwargs):
        return self.__getattr__("display")(*args, **kwargs)

    def dodecahedral_graph(self, *args, **kwargs):
        return self.__getattr__("dodecahedral_graph")(*args, **kwargs)

    def dominance_frontiers(self, *args, **kwargs):
        return self.__getattr__("dominance_frontiers")(*args, **kwargs)

    def dominating_set(self, *args, **kwargs):
        return self.__getattr__("dominating_set")(*args, **kwargs)

    def dorogovtsev_goltsev_mendes_graph(self, *args, **kwargs):
        return self.__getattr__("dorogovtsev_goltsev_mendes_graph")(*args, **kwargs)

    def double_edge_swap(self, *args, **kwargs):
        return self.__getattr__("double_edge_swap")(*args, **kwargs)

    def draw(self, *args, **kwargs):
        return self.__getattr__("draw")(*args, **kwargs)

    def draw_bipartite(self, *args, **kwargs):
        return self.__getattr__("draw_bipartite")(*args, **kwargs)

    def draw_circular(self, *args, **kwargs):
        return self.__getattr__("draw_circular")(*args, **kwargs)

    def draw_forceatlas2(self, *args, **kwargs):
        return self.__getattr__("draw_forceatlas2")(*args, **kwargs)

    def draw_kamada_kawai(self, *args, **kwargs):
        return self.__getattr__("draw_kamada_kawai")(*args, **kwargs)

    def draw_networkx(self, *args, **kwargs):
        return self.__getattr__("draw_networkx")(*args, **kwargs)

    def draw_networkx_edge_labels(self, *args, **kwargs):
        return self.__getattr__("draw_networkx_edge_labels")(*args, **kwargs)

    def draw_networkx_edges(self, *args, **kwargs):
        return self.__getattr__("draw_networkx_edges")(*args, **kwargs)

    def draw_networkx_labels(self, *args, **kwargs):
        return self.__getattr__("draw_networkx_labels")(*args, **kwargs)

    def draw_networkx_nodes(self, *args, **kwargs):
        return self.__getattr__("draw_networkx_nodes")(*args, **kwargs)

    def draw_planar(self, *args, **kwargs):
        return self.__getattr__("draw_planar")(*args, **kwargs)

    def draw_random(self, *args, **kwargs):
        return self.__getattr__("draw_random")(*args, **kwargs)

    def draw_shell(self, *args, **kwargs):
        return self.__getattr__("draw_shell")(*args, **kwargs)

    def draw_spectral(self, *args, **kwargs):
        return self.__getattr__("draw_spectral")(*args, **kwargs)

    def draw_spring(self, *args, **kwargs):
        return self.__getattr__("draw_spring")(*args, **kwargs)

    def dual_barabasi_albert_graph(self, *args, **kwargs):
        return self.__getattr__("dual_barabasi_albert_graph")(*args, **kwargs)

    def duplication_divergence_graph(self, *args, **kwargs):
        return self.__getattr__("duplication_divergence_graph")(*args, **kwargs)

    def eccentricity(self, *args, **kwargs):
        return self.__getattr__("eccentricity")(*args, **kwargs)

    def edge_betweenness_centrality(self, *args, **kwargs):
        return self.__getattr__("edge_betweenness_centrality")(*args, **kwargs)

    def edge_betweenness_centrality_subset(self, *args, **kwargs):
        return self.__getattr__("edge_betweenness_centrality_subset")(*args, **kwargs)

    def edge_betweenness_partition(self, *args, **kwargs):
        return self.__getattr__("edge_betweenness_partition")(*args, **kwargs)

    def edge_bfs(self, *args, **kwargs):
        return self.__getattr__("edge_bfs")(*args, **kwargs)

    def edge_boundary(self, *args, **kwargs):
        return self.__getattr__("edge_boundary")(*args, **kwargs)

    def edge_connectivity(self, *args, **kwargs):
        return self.__getattr__("edge_connectivity")(*args, **kwargs)

    def edge_current_flow_betweenness_centrality(self, *args, **kwargs):
        return self.__getattr__("edge_current_flow_betweenness_centrality")(*args, **kwargs)

    def edge_current_flow_betweenness_centrality_subset(self, *args, **kwargs):
        return self.__getattr__("edge_current_flow_betweenness_centrality_subset")(*args, **kwargs)

    def edge_current_flow_betweenness_partition(self, *args, **kwargs):
        return self.__getattr__("edge_current_flow_betweenness_partition")(*args, **kwargs)

    def edge_dfs(self, *args, **kwargs):
        return self.__getattr__("edge_dfs")(*args, **kwargs)

    def edge_disjoint_paths(self, *args, **kwargs):
        return self.__getattr__("edge_disjoint_paths")(*args, **kwargs)

    def edge_expansion(self, *args, **kwargs):
        return self.__getattr__("edge_expansion")(*args, **kwargs)

    def edge_load_centrality(self, *args, **kwargs):
        return self.__getattr__("edge_load_centrality")(*args, **kwargs)

    def edge_subgraph(self, *args, **kwargs):
        return self.__getattr__("edge_subgraph")(*args, **kwargs)

    def edges(self, *args, **kwargs):
        return self.__getattr__("edges")(*args, **kwargs)

    def edmonds_karp(self, *args, **kwargs):
        return self.__getattr__("edmonds_karp")(*args, **kwargs)

    def effective_graph_resistance(self, *args, **kwargs):
        return self.__getattr__("effective_graph_resistance")(*args, **kwargs)

    def effective_size(self, *args, **kwargs):
        return self.__getattr__("effective_size")(*args, **kwargs)

    def efficiency(self, *args, **kwargs):
        return self.__getattr__("efficiency")(*args, **kwargs)

    def ego_graph(self, *args, **kwargs):
        return self.__getattr__("ego_graph")(*args, **kwargs)

    def eigenvector_centrality(self, *args, **kwargs):
        return self.__getattr__("eigenvector_centrality")(*args, **kwargs)

    def eigenvector_centrality_numpy(self, *args, **kwargs):
        return self.__getattr__("eigenvector_centrality_numpy")(*args, **kwargs)

    def empty_graph(self, *args, **kwargs):
        return self.__getattr__("empty_graph")(*args, **kwargs)

    def enumerate_all_cliques(self, *args, **kwargs):
        return self.__getattr__("enumerate_all_cliques")(*args, **kwargs)

    def eppstein_matching(self, *args, **kwargs):
        return self.__getattr__("eppstein_matching")(*args, **kwargs)

    def equitable_color(self, *args, **kwargs):
        return self.__getattr__("equitable_color")(*args, **kwargs)

    def equivalence_classes(self, *args, **kwargs):
        return self.__getattr__("equivalence_classes")(*args, **kwargs)

    def erdos_renyi_graph(self, *args, **kwargs):
        return self.__getattr__("erdos_renyi_graph")(*args, **kwargs)

    def estrada_index(self, *args, **kwargs):
        return self.__getattr__("estrada_index")(*args, **kwargs)

    def eulerian_circuit(self, *args, **kwargs):
        return self.__getattr__("eulerian_circuit")(*args, **kwargs)

    def eulerian_path(self, *args, **kwargs):
        return self.__getattr__("eulerian_path")(*args, **kwargs)

    def eulerize(self, *args, **kwargs):
        return self.__getattr__("eulerize")(*args, **kwargs)

    def expected_degree_graph(self, *args, **kwargs):
        return self.__getattr__("expected_degree_graph")(*args, **kwargs)

    def extended_barabasi_albert_graph(self, *args, **kwargs):
        return self.__getattr__("extended_barabasi_albert_graph")(*args, **kwargs)

    def fast_could_be_isomorphic(self, *args, **kwargs):
        return self.__getattr__("fast_could_be_isomorphic")(*args, **kwargs)

    def fast_gnp_random_graph(self, *args, **kwargs):
        return self.__getattr__("fast_gnp_random_graph")(*args, **kwargs)

    def fast_label_propagation_communities(self, *args, **kwargs):
        return self.__getattr__("fast_label_propagation_communities")(*args, **kwargs)

    def faster_could_be_isomorphic(self, *args, **kwargs):
        return self.__getattr__("faster_could_be_isomorphic")(*args, **kwargs)

    def fiedler_vector(self, *args, **kwargs):
        return self.__getattr__("fiedler_vector")(*args, **kwargs)

    def find_asteroidal_triple(self, *args, **kwargs):
        return self.__getattr__("find_asteroidal_triple")(*args, **kwargs)

    def find_cliques(self, *args, **kwargs):
        return self.__getattr__("find_cliques")(*args, **kwargs)

    def find_cliques_recursive(self, *args, **kwargs):
        return self.__getattr__("find_cliques_recursive")(*args, **kwargs)

    def find_cycle(self, *args, **kwargs):
        return self.__getattr__("find_cycle")(*args, **kwargs)

    def find_induced_nodes(self, *args, **kwargs):
        return self.__getattr__("find_induced_nodes")(*args, **kwargs)

    def find_minimal_d_separator(self, *args, **kwargs):
        return self.__getattr__("find_minimal_d_separator")(*args, **kwargs)

    def find_negative_cycle(self, *args, **kwargs):
        return self.__getattr__("find_negative_cycle")(*args, **kwargs)

    def florentine_families_graph(self, *args, **kwargs):
        return self.__getattr__("florentine_families_graph")(*args, **kwargs)

    def flow_hierarchy(self, *args, **kwargs):
        return self.__getattr__("flow_hierarchy")(*args, **kwargs)

    def floyd_warshall(self, *args, **kwargs):
        return self.__getattr__("floyd_warshall")(*args, **kwargs)

    def floyd_warshall_numpy(self, *args, **kwargs):
        return self.__getattr__("floyd_warshall_numpy")(*args, **kwargs)

    def floyd_warshall_predecessor_and_distance(self, *args, **kwargs):
        return self.__getattr__("floyd_warshall_predecessor_and_distance")(*args, **kwargs)

    def forceatlas2_layout(self, *args, **kwargs):
        return self.__getattr__("forceatlas2_layout")(*args, **kwargs)

    def freeze(self, *args, **kwargs):
        return self.__getattr__("freeze")(*args, **kwargs)

    def from_biadjacency_matrix(self, *args, **kwargs):
        return self.__getattr__("from_biadjacency_matrix")(*args, **kwargs)

    def from_dict_of_dicts(self, *args, **kwargs):
        return self.__getattr__("from_dict_of_dicts")(*args, **kwargs)

    def from_dict_of_lists(self, *args, **kwargs):
        return self.__getattr__("from_dict_of_lists")(*args, **kwargs)

    def from_edgelist(self, *args, **kwargs):
        return self.__getattr__("from_edgelist")(*args, **kwargs)

    def from_graph6_bytes(self, *args, **kwargs):
        return self.__getattr__("from_graph6_bytes")(*args, **kwargs)

    def from_nested_tuple(self, *args, **kwargs):
        return self.__getattr__("from_nested_tuple")(*args, **kwargs)

    def from_numpy_array(self, *args, **kwargs):
        return self.__getattr__("from_numpy_array")(*args, **kwargs)

    def from_pandas_adjacency(self, *args, **kwargs):
        return self.__getattr__("from_pandas_adjacency")(*args, **kwargs)

    def from_pandas_edgelist(self, *args, **kwargs):
        return self.__getattr__("from_pandas_edgelist")(*args, **kwargs)

    def from_prufer_sequence(self, *args, **kwargs):
        return self.__getattr__("from_prufer_sequence")(*args, **kwargs)

    def from_scipy_sparse_array(self, *args, **kwargs):
        return self.__getattr__("from_scipy_sparse_array")(*args, **kwargs)

    def from_sparse6_bytes(self, *args, **kwargs):
        return self.__getattr__("from_sparse6_bytes")(*args, **kwargs)

    def frozen(self, *args, **kwargs):
        return self.__getattr__("frozen")(*args, **kwargs)

    def frucht_graph(self, *args, **kwargs):
        return self.__getattr__("frucht_graph")(*args, **kwargs)

    def fruchterman_reingold_layout(self, *args, **kwargs):
        return self.__getattr__("fruchterman_reingold_layout")(*args, **kwargs)

    def full_join(self, *args, **kwargs):
        return self.__getattr__("full_join")(*args, **kwargs)

    def full_rary_tree(self, *args, **kwargs):
        return self.__getattr__("full_rary_tree")(*args, **kwargs)

    def gaussian_random_partition_graph(self, *args, **kwargs):
        return self.__getattr__("gaussian_random_partition_graph")(*args, **kwargs)

    def general_random_intersection_graph(self, *args, **kwargs):
        return self.__getattr__("general_random_intersection_graph")(*args, **kwargs)

    def generalized_degree(self, *args, **kwargs):
        return self.__getattr__("generalized_degree")(*args, **kwargs)

    def generate_adjlist(self, *args, **kwargs):
        return self.__getattr__("generate_adjlist")(*args, **kwargs)

    def generate_edgelist(self, *args, **kwargs):
        return self.__getattr__("generate_edgelist")(*args, **kwargs)

    def generate_gexf(self, *args, **kwargs):
        return self.__getattr__("generate_gexf")(*args, **kwargs)

    def generate_gml(self, *args, **kwargs):
        return self.__getattr__("generate_gml")(*args, **kwargs)

    def generate_graphml(self, *args, **kwargs):
        return self.__getattr__("generate_graphml")(*args, **kwargs)

    def generate_multiline_adjlist(self, *args, **kwargs):
        return self.__getattr__("generate_multiline_adjlist")(*args, **kwargs)

    def generate_network_text(self, *args, **kwargs):
        return self.__getattr__("generate_network_text")(*args, **kwargs)

    def generate_pajek(self, *args, **kwargs):
        return self.__getattr__("generate_pajek")(*args, **kwargs)

    def generate_random_paths(self, *args, **kwargs):
        return self.__getattr__("generate_random_paths")(*args, **kwargs)

    def generic_bfs_edges(self, *args, **kwargs):
        return self.__getattr__("generic_bfs_edges")(*args, **kwargs)

    def generic_weighted_projected_graph(self, *args, **kwargs):
        return self.__getattr__("generic_weighted_projected_graph")(*args, **kwargs)

    def geographical_threshold_graph(self, *args, **kwargs):
        return self.__getattr__("geographical_threshold_graph")(*args, **kwargs)

    def geometric_edges(self, *args, **kwargs):
        return self.__getattr__("geometric_edges")(*args, **kwargs)

    def geometric_soft_configuration_graph(self, *args, **kwargs):
        return self.__getattr__("geometric_soft_configuration_graph")(*args, **kwargs)

    def get_edge_attributes(self, *args, **kwargs):
        return self.__getattr__("get_edge_attributes")(*args, **kwargs)

    def get_node_attributes(self, *args, **kwargs):
        return self.__getattr__("get_node_attributes")(*args, **kwargs)

    def girth(self, *args, **kwargs):
        return self.__getattr__("girth")(*args, **kwargs)

    def girvan_newman(self, *args, **kwargs):
        return self.__getattr__("girvan_newman")(*args, **kwargs)

    def global_efficiency(self, *args, **kwargs):
        return self.__getattr__("global_efficiency")(*args, **kwargs)

    def global_parameters(self, *args, **kwargs):
        return self.__getattr__("global_parameters")(*args, **kwargs)

    def global_reaching_centrality(self, *args, **kwargs):
        return self.__getattr__("global_reaching_centrality")(*args, **kwargs)

    def gn_graph(self, *args, **kwargs):
        return self.__getattr__("gn_graph")(*args, **kwargs)

    def gnc_graph(self, *args, **kwargs):
        return self.__getattr__("gnc_graph")(*args, **kwargs)

    def gnm_random_graph(self, *args, **kwargs):
        return self.__getattr__("gnm_random_graph")(*args, **kwargs)

    def gnmk_random_graph(self, *args, **kwargs):
        return self.__getattr__("gnmk_random_graph")(*args, **kwargs)

    def gnp_random_graph(self, *args, **kwargs):
        return self.__getattr__("gnp_random_graph")(*args, **kwargs)

    def gnr_graph(self, *args, **kwargs):
        return self.__getattr__("gnr_graph")(*args, **kwargs)

    def goldberg_radzik(self, *args, **kwargs):
        return self.__getattr__("goldberg_radzik")(*args, **kwargs)

    def gomory_hu_tree(self, *args, **kwargs):
        return self.__getattr__("gomory_hu_tree")(*args, **kwargs)

    def google_matrix(self, *args, **kwargs):
        return self.__getattr__("google_matrix")(*args, **kwargs)

    def graph_atlas(self, *args, **kwargs):
        return self.__getattr__("graph_atlas")(*args, **kwargs)

    def graph_atlas_g(self, *args, **kwargs):
        return self.__getattr__("graph_atlas_g")(*args, **kwargs)

    def graph_edit_distance(self, *args, **kwargs):
        return self.__getattr__("graph_edit_distance")(*args, **kwargs)

    def greedy_color(self, *args, **kwargs):
        return self.__getattr__("greedy_color")(*args, **kwargs)

    def greedy_modularity_communities(self, *args, **kwargs):
        return self.__getattr__("greedy_modularity_communities")(*args, **kwargs)

    def greedy_source_expansion(self, *args, **kwargs):
        return self.__getattr__("greedy_source_expansion")(*args, **kwargs)

    def greedy_tsp(self, *args, **kwargs):
        return self.__getattr__("greedy_tsp")(*args, **kwargs)

    def grid_2d_graph(self, *args, **kwargs):
        return self.__getattr__("grid_2d_graph")(*args, **kwargs)

    def grid_graph(self, *args, **kwargs):
        return self.__getattr__("grid_graph")(*args, **kwargs)

    def group_betweenness_centrality(self, *args, **kwargs):
        return self.__getattr__("group_betweenness_centrality")(*args, **kwargs)

    def group_closeness_centrality(self, *args, **kwargs):
        return self.__getattr__("group_closeness_centrality")(*args, **kwargs)

    def group_degree_centrality(self, *args, **kwargs):
        return self.__getattr__("group_degree_centrality")(*args, **kwargs)

    def group_in_degree_centrality(self, *args, **kwargs):
        return self.__getattr__("group_in_degree_centrality")(*args, **kwargs)

    def group_out_degree_centrality(self, *args, **kwargs):
        return self.__getattr__("group_out_degree_centrality")(*args, **kwargs)

    def gutman_index(self, *args, **kwargs):
        return self.__getattr__("gutman_index")(*args, **kwargs)

    def harmonic_centrality(self, *args, **kwargs):
        return self.__getattr__("harmonic_centrality")(*args, **kwargs)

    def harmonic_diameter(self, *args, **kwargs):
        return self.__getattr__("harmonic_diameter")(*args, **kwargs)

    def has_bridges(self, *args, **kwargs):
        return self.__getattr__("has_bridges")(*args, **kwargs)

    def has_eulerian_path(self, *args, **kwargs):
        return self.__getattr__("has_eulerian_path")(*args, **kwargs)

    def has_path(self, *args, **kwargs):
        return self.__getattr__("has_path")(*args, **kwargs)

    def havel_hakimi_graph(self, *args, **kwargs):
        return self.__getattr__("havel_hakimi_graph")(*args, **kwargs)

    def heawood_graph(self, *args, **kwargs):
        return self.__getattr__("heawood_graph")(*args, **kwargs)

    def hexagonal_lattice_graph(self, *args, **kwargs):
        return self.__getattr__("hexagonal_lattice_graph")(*args, **kwargs)

    def hits(self, *args, **kwargs):
        return self.__getattr__("hits")(*args, **kwargs)

    def hkn_harary_graph(self, *args, **kwargs):
        return self.__getattr__("hkn_harary_graph")(*args, **kwargs)

    def hnm_harary_graph(self, *args, **kwargs):
        return self.__getattr__("hnm_harary_graph")(*args, **kwargs)

    def hoffman_singleton_graph(self, *args, **kwargs):
        return self.__getattr__("hoffman_singleton_graph")(*args, **kwargs)

    def hopcroft_karp_matching(self, *args, **kwargs):
        return self.__getattr__("hopcroft_karp_matching")(*args, **kwargs)

    def house_graph(self, *args, **kwargs):
        return self.__getattr__("house_graph")(*args, **kwargs)

    def house_x_graph(self, *args, **kwargs):
        return self.__getattr__("house_x_graph")(*args, **kwargs)

    def hypercube_graph(self, *args, **kwargs):
        return self.__getattr__("hypercube_graph")(*args, **kwargs)

    def icosahedral_graph(self, *args, **kwargs):
        return self.__getattr__("icosahedral_graph")(*args, **kwargs)

    def identified_nodes(self, *args, **kwargs):
        return self.__getattr__("identified_nodes")(*args, **kwargs)

    def immediate_dominators(self, *args, **kwargs):
        return self.__getattr__("immediate_dominators")(*args, **kwargs)

    def in_degree_centrality(self, *args, **kwargs):
        return self.__getattr__("in_degree_centrality")(*args, **kwargs)

    def incidence_matrix(self, *args, **kwargs):
        return self.__getattr__("incidence_matrix")(*args, **kwargs)

    def incremental_closeness_centrality(self, *args, **kwargs):
        return self.__getattr__("incremental_closeness_centrality")(*args, **kwargs)

    def induced_subgraph(self, *args, **kwargs):
        return self.__getattr__("induced_subgraph")(*args, **kwargs)

    def information_centrality(self, *args, **kwargs):
        return self.__getattr__("information_centrality")(*args, **kwargs)

    def intersection(self, *args, **kwargs):
        return self.__getattr__("intersection")(*args, **kwargs)

    def intersection_all(self, *args, **kwargs):
        return self.__getattr__("intersection_all")(*args, **kwargs)

    def intersection_array(self, *args, **kwargs):
        return self.__getattr__("intersection_array")(*args, **kwargs)

    def interval_graph(self, *args, **kwargs):
        return self.__getattr__("interval_graph")(*args, **kwargs)

    def inverse_line_graph(self, *args, **kwargs):
        return self.__getattr__("inverse_line_graph")(*args, **kwargs)

    def is_aperiodic(self, *args, **kwargs):
        return self.__getattr__("is_aperiodic")(*args, **kwargs)

    def is_arborescence(self, *args, **kwargs):
        return self.__getattr__("is_arborescence")(*args, **kwargs)

    def is_at_free(self, *args, **kwargs):
        return self.__getattr__("is_at_free")(*args, **kwargs)

    def is_attracting_component(self, *args, **kwargs):
        return self.__getattr__("is_attracting_component")(*args, **kwargs)

    def is_biconnected(self, *args, **kwargs):
        return self.__getattr__("is_biconnected")(*args, **kwargs)

    def is_bipartite(self, *args, **kwargs):
        return self.__getattr__("is_bipartite")(*args, **kwargs)

    def is_bipartite_node_set(self, *args, **kwargs):
        return self.__getattr__("is_bipartite_node_set")(*args, **kwargs)

    def is_branching(self, *args, **kwargs):
        return self.__getattr__("is_branching")(*args, **kwargs)

    def is_chordal(self, *args, **kwargs):
        return self.__getattr__("is_chordal")(*args, **kwargs)

    def is_connected(self, *args, **kwargs):
        return self.__getattr__("is_connected")(*args, **kwargs)

    def is_connected_dominating_set(self, *args, **kwargs):
        return self.__getattr__("is_connected_dominating_set")(*args, **kwargs)

    def is_d_separator(self, *args, **kwargs):
        return self.__getattr__("is_d_separator")(*args, **kwargs)

    def is_digraphical(self, *args, **kwargs):
        return self.__getattr__("is_digraphical")(*args, **kwargs)

    def is_directed(self, *args, **kwargs):
        return self.__getattr__("is_directed")(*args, **kwargs)

    def is_directed_acyclic_graph(self, *args, **kwargs):
        return self.__getattr__("is_directed_acyclic_graph")(*args, **kwargs)

    def is_distance_regular(self, *args, **kwargs):
        return self.__getattr__("is_distance_regular")(*args, **kwargs)

    def is_dominating_set(self, *args, **kwargs):
        return self.__getattr__("is_dominating_set")(*args, **kwargs)

    def is_edge_cover(self, *args, **kwargs):
        return self.__getattr__("is_edge_cover")(*args, **kwargs)

    def is_empty(self, *args, **kwargs):
        return self.__getattr__("is_empty")(*args, **kwargs)

    def is_eulerian(self, *args, **kwargs):
        return self.__getattr__("is_eulerian")(*args, **kwargs)

    def is_forest(self, *args, **kwargs):
        return self.__getattr__("is_forest")(*args, **kwargs)

    def is_frozen(self, *args, **kwargs):
        return self.__getattr__("is_frozen")(*args, **kwargs)

    def is_graphical(self, *args, **kwargs):
        return self.__getattr__("is_graphical")(*args, **kwargs)

    def is_isolate(self, *args, **kwargs):
        return self.__getattr__("is_isolate")(*args, **kwargs)

    def is_isomorphic(self, *args, **kwargs):
        return self.__getattr__("is_isomorphic")(*args, **kwargs)

    def is_k_edge_connected(self, *args, **kwargs):
        return self.__getattr__("is_k_edge_connected")(*args, **kwargs)

    def is_k_regular(self, *args, **kwargs):
        return self.__getattr__("is_k_regular")(*args, **kwargs)

    def is_kl_connected(self, *args, **kwargs):
        return self.__getattr__("is_kl_connected")(*args, **kwargs)

    def is_matching(self, *args, **kwargs):
        return self.__getattr__("is_matching")(*args, **kwargs)

    def is_maximal_matching(self, *args, **kwargs):
        return self.__getattr__("is_maximal_matching")(*args, **kwargs)

    def is_minimal_d_separator(self, *args, **kwargs):
        return self.__getattr__("is_minimal_d_separator")(*args, **kwargs)

    def is_multigraphical(self, *args, **kwargs):
        return self.__getattr__("is_multigraphical")(*args, **kwargs)

    def is_negatively_weighted(self, *args, **kwargs):
        return self.__getattr__("is_negatively_weighted")(*args, **kwargs)

    def is_partition(self, *args, **kwargs):
        return self.__getattr__("is_partition")(*args, **kwargs)

    def is_path(self, *args, **kwargs):
        return self.__getattr__("is_path")(*args, **kwargs)

    def is_perfect_matching(self, *args, **kwargs):
        return self.__getattr__("is_perfect_matching")(*args, **kwargs)

    def is_planar(self, *args, **kwargs):
        return self.__getattr__("is_planar")(*args, **kwargs)

    def is_pseudographical(self, *args, **kwargs):
        return self.__getattr__("is_pseudographical")(*args, **kwargs)

    def is_regular(self, *args, **kwargs):
        return self.__getattr__("is_regular")(*args, **kwargs)

    def is_regular_expander(self, *args, **kwargs):
        return self.__getattr__("is_regular_expander")(*args, **kwargs)

    def is_semiconnected(self, *args, **kwargs):
        return self.__getattr__("is_semiconnected")(*args, **kwargs)

    def is_semieulerian(self, *args, **kwargs):
        return self.__getattr__("is_semieulerian")(*args, **kwargs)

    def is_simple_path(self, *args, **kwargs):
        return self.__getattr__("is_simple_path")(*args, **kwargs)

    def is_strongly_connected(self, *args, **kwargs):
        return self.__getattr__("is_strongly_connected")(*args, **kwargs)

    def is_strongly_regular(self, *args, **kwargs):
        return self.__getattr__("is_strongly_regular")(*args, **kwargs)

    def is_tournament(self, *args, **kwargs):
        return self.__getattr__("is_tournament")(*args, **kwargs)

    def is_tree(self, *args, **kwargs):
        return self.__getattr__("is_tree")(*args, **kwargs)

    def is_triad(self, *args, **kwargs):
        return self.__getattr__("is_triad")(*args, **kwargs)

    def is_valid_degree_sequence_erdos_gallai(self, *args, **kwargs):
        return self.__getattr__("is_valid_degree_sequence_erdos_gallai")(*args, **kwargs)

    def is_valid_degree_sequence_havel_hakimi(self, *args, **kwargs):
        return self.__getattr__("is_valid_degree_sequence_havel_hakimi")(*args, **kwargs)

    def is_valid_directed_joint_degree(self, *args, **kwargs):
        return self.__getattr__("is_valid_directed_joint_degree")(*args, **kwargs)

    def is_valid_joint_degree(self, *args, **kwargs):
        return self.__getattr__("is_valid_joint_degree")(*args, **kwargs)

    def is_weakly_connected(self, *args, **kwargs):
        return self.__getattr__("is_weakly_connected")(*args, **kwargs)

    def is_weighted(self, *args, **kwargs):
        return self.__getattr__("is_weighted")(*args, **kwargs)

    def isolates(self, *args, **kwargs):
        return self.__getattr__("isolates")(*args, **kwargs)

    def jaccard_coefficient(self, *args, **kwargs):
        return self.__getattr__("jaccard_coefficient")(*args, **kwargs)

    def johnson(self, *args, **kwargs):
        return self.__getattr__("johnson")(*args, **kwargs)

    def join_trees(self, *args, **kwargs):
        return self.__getattr__("join_trees")(*args, **kwargs)

    def joint_degree_graph(self, *args, **kwargs):
        return self.__getattr__("joint_degree_graph")(*args, **kwargs)

    def junction_tree(self, *args, **kwargs):
        return self.__getattr__("junction_tree")(*args, **kwargs)

    def k_clique_communities(self, *args, **kwargs):
        return self.__getattr__("k_clique_communities")(*args, **kwargs)

    def k_components(self, *args, **kwargs):
        return self.__getattr__("k_components")(*args, **kwargs)

    def k_core(self, *args, **kwargs):
        return self.__getattr__("k_core")(*args, **kwargs)

    def k_corona(self, *args, **kwargs):
        return self.__getattr__("k_corona")(*args, **kwargs)

    def k_crust(self, *args, **kwargs):
        return self.__getattr__("k_crust")(*args, **kwargs)

    def k_edge_augmentation(self, *args, **kwargs):
        return self.__getattr__("k_edge_augmentation")(*args, **kwargs)

    def k_edge_components(self, *args, **kwargs):
        return self.__getattr__("k_edge_components")(*args, **kwargs)

    def k_edge_subgraphs(self, *args, **kwargs):
        return self.__getattr__("k_edge_subgraphs")(*args, **kwargs)

    def k_factor(self, *args, **kwargs):
        return self.__getattr__("k_factor")(*args, **kwargs)

    def k_random_intersection_graph(self, *args, **kwargs):
        return self.__getattr__("k_random_intersection_graph")(*args, **kwargs)

    def k_shell(self, *args, **kwargs):
        return self.__getattr__("k_shell")(*args, **kwargs)

    def k_truss(self, *args, **kwargs):
        return self.__getattr__("k_truss")(*args, **kwargs)

    def kamada_kawai_layout(self, *args, **kwargs):
        return self.__getattr__("kamada_kawai_layout")(*args, **kwargs)

    def karate_club_graph(self, *args, **kwargs):
        return self.__getattr__("karate_club_graph")(*args, **kwargs)

    def katz_centrality(self, *args, **kwargs):
        return self.__getattr__("katz_centrality")(*args, **kwargs)

    def katz_centrality_numpy(self, *args, **kwargs):
        return self.__getattr__("katz_centrality_numpy")(*args, **kwargs)

    def kemeny_constant(self, *args, **kwargs):
        return self.__getattr__("kemeny_constant")(*args, **kwargs)

    def kernighan_lin_bisection(self, *args, **kwargs):
        return self.__getattr__("kernighan_lin_bisection")(*args, **kwargs)

    def kl_connected_subgraph(self, *args, **kwargs):
        return self.__getattr__("kl_connected_subgraph")(*args, **kwargs)

    def kneser_graph(self, *args, **kwargs):
        return self.__getattr__("kneser_graph")(*args, **kwargs)

    def kosaraju_strongly_connected_components(self, *args, **kwargs):
        return self.__getattr__("kosaraju_strongly_connected_components")(*args, **kwargs)

    def krackhardt_kite_graph(self, *args, **kwargs):
        return self.__getattr__("krackhardt_kite_graph")(*args, **kwargs)

    def label_propagation_communities(self, *args, **kwargs):
        return self.__getattr__("label_propagation_communities")(*args, **kwargs)

    def ladder_graph(self, *args, **kwargs):
        return self.__getattr__("ladder_graph")(*args, **kwargs)

    def laplacian_centrality(self, *args, **kwargs):
        return self.__getattr__("laplacian_centrality")(*args, **kwargs)

    def laplacian_matrix(self, *args, **kwargs):
        return self.__getattr__("laplacian_matrix")(*args, **kwargs)

    def laplacian_spectrum(self, *args, **kwargs):
        return self.__getattr__("laplacian_spectrum")(*args, **kwargs)

    def large_clique_size(self, *args, **kwargs):
        return self.__getattr__("large_clique_size")(*args, **kwargs)

    def latapy_clustering(self, *args, **kwargs):
        return self.__getattr__("latapy_clustering")(*args, **kwargs)

    def lattice_reference(self, *args, **kwargs):
        return self.__getattr__("lattice_reference")(*args, **kwargs)

    def leiden_communities(self, *args, **kwargs):
        return self.__getattr__("leiden_communities")(*args, **kwargs)

    def leiden_partitions(self, *args, **kwargs):
        return self.__getattr__("leiden_partitions")(*args, **kwargs)

    def les_miserables_graph(self, *args, **kwargs):
        return self.__getattr__("les_miserables_graph")(*args, **kwargs)

    def lexicographic_product(self, *args, **kwargs):
        return self.__getattr__("lexicographic_product")(*args, **kwargs)

    def lexicographical_topological_sort(self, *args, **kwargs):
        return self.__getattr__("lexicographical_topological_sort")(*args, **kwargs)

    def line_graph(self, *args, **kwargs):
        return self.__getattr__("line_graph")(*args, **kwargs)

    def load_centrality(self, *args, **kwargs):
        return self.__getattr__("load_centrality")(*args, **kwargs)

    def local_bridges(self, *args, **kwargs):
        return self.__getattr__("local_bridges")(*args, **kwargs)

    def local_constraint(self, *args, **kwargs):
        return self.__getattr__("local_constraint")(*args, **kwargs)

    def local_efficiency(self, *args, **kwargs):
        return self.__getattr__("local_efficiency")(*args, **kwargs)

    def local_node_connectivity(self, *args, **kwargs):
        return self.__getattr__("local_node_connectivity")(*args, **kwargs)

    def local_reaching_centrality(self, *args, **kwargs):
        return self.__getattr__("local_reaching_centrality")(*args, **kwargs)

    def lollipop_graph(self, *args, **kwargs):
        return self.__getattr__("lollipop_graph")(*args, **kwargs)

    def louvain_communities(self, *args, **kwargs):
        return self.__getattr__("louvain_communities")(*args, **kwargs)

    def louvain_partitions(self, *args, **kwargs):
        return self.__getattr__("louvain_partitions")(*args, **kwargs)

    def lowest_common_ancestor(self, *args, **kwargs):
        return self.__getattr__("lowest_common_ancestor")(*args, **kwargs)

    def lukes_partitioning(self, *args, **kwargs):
        return self.__getattr__("lukes_partitioning")(*args, **kwargs)

    def make_clique_bipartite(self, *args, **kwargs):
        return self.__getattr__("make_clique_bipartite")(*args, **kwargs)

    def make_max_clique_graph(self, *args, **kwargs):
        return self.__getattr__("make_max_clique_graph")(*args, **kwargs)

    def margulis_gabber_galil_graph(self, *args, **kwargs):
        return self.__getattr__("margulis_gabber_galil_graph")(*args, **kwargs)

    def max_clique(self, *args, **kwargs):
        return self.__getattr__("max_clique")(*args, **kwargs)

    def max_flow_min_cost(self, *args, **kwargs):
        return self.__getattr__("max_flow_min_cost")(*args, **kwargs)

    def max_weight_clique(self, *args, **kwargs):
        return self.__getattr__("max_weight_clique")(*args, **kwargs)

    def max_weight_matching(self, *args, **kwargs):
        return self.__getattr__("max_weight_matching")(*args, **kwargs)

    def maximal_extendability(self, *args, **kwargs):
        return self.__getattr__("maximal_extendability")(*args, **kwargs)

    def maximal_independent_set(self, *args, **kwargs):
        return self.__getattr__("maximal_independent_set")(*args, **kwargs)

    def maximal_matching(self, *args, **kwargs):
        return self.__getattr__("maximal_matching")(*args, **kwargs)

    def maximum_branching(self, *args, **kwargs):
        return self.__getattr__("maximum_branching")(*args, **kwargs)

    def maximum_flow(self, *args, **kwargs):
        return self.__getattr__("maximum_flow")(*args, **kwargs)

    def maximum_flow_value(self, *args, **kwargs):
        return self.__getattr__("maximum_flow_value")(*args, **kwargs)

    def maximum_independent_set(self, *args, **kwargs):
        return self.__getattr__("maximum_independent_set")(*args, **kwargs)

    def maximum_matching(self, *args, **kwargs):
        return self.__getattr__("maximum_matching")(*args, **kwargs)

    def maximum_spanning_arborescence(self, *args, **kwargs):
        return self.__getattr__("maximum_spanning_arborescence")(*args, **kwargs)

    def maximum_spanning_edges(self, *args, **kwargs):
        return self.__getattr__("maximum_spanning_edges")(*args, **kwargs)

    def maximum_spanning_tree(self, *args, **kwargs):
        return self.__getattr__("maximum_spanning_tree")(*args, **kwargs)

    def maybe_regular_expander(self, *args, **kwargs):
        return self.__getattr__("maybe_regular_expander")(*args, **kwargs)

    def metric_closure(self, *args, **kwargs):
        return self.__getattr__("metric_closure")(*args, **kwargs)

    def min_cost_flow(self, *args, **kwargs):
        return self.__getattr__("min_cost_flow")(*args, **kwargs)

    def min_cost_flow_cost(self, *args, **kwargs):
        return self.__getattr__("min_cost_flow_cost")(*args, **kwargs)

    def min_edge_cover(self, *args, **kwargs):
        return self.__getattr__("min_edge_cover")(*args, **kwargs)

    def min_edge_dominating_set(self, *args, **kwargs):
        return self.__getattr__("min_edge_dominating_set")(*args, **kwargs)

    def min_maximal_matching(self, *args, **kwargs):
        return self.__getattr__("min_maximal_matching")(*args, **kwargs)

    def min_weight_matching(self, *args, **kwargs):
        return self.__getattr__("min_weight_matching")(*args, **kwargs)

    def min_weighted_dominating_set(self, *args, **kwargs):
        return self.__getattr__("min_weighted_dominating_set")(*args, **kwargs)

    def min_weighted_vertex_cover(self, *args, **kwargs):
        return self.__getattr__("min_weighted_vertex_cover")(*args, **kwargs)

    def minimum_branching(self, *args, **kwargs):
        return self.__getattr__("minimum_branching")(*args, **kwargs)

    def minimum_cut(self, *args, **kwargs):
        return self.__getattr__("minimum_cut")(*args, **kwargs)

    def minimum_cut_value(self, *args, **kwargs):
        return self.__getattr__("minimum_cut_value")(*args, **kwargs)

    def minimum_cycle_basis(self, *args, **kwargs):
        return self.__getattr__("minimum_cycle_basis")(*args, **kwargs)

    def minimum_edge_cut(self, *args, **kwargs):
        return self.__getattr__("minimum_edge_cut")(*args, **kwargs)

    def minimum_node_cut(self, *args, **kwargs):
        return self.__getattr__("minimum_node_cut")(*args, **kwargs)

    def minimum_spanning_arborescence(self, *args, **kwargs):
        return self.__getattr__("minimum_spanning_arborescence")(*args, **kwargs)

    def minimum_spanning_edges(self, *args, **kwargs):
        return self.__getattr__("minimum_spanning_edges")(*args, **kwargs)

    def minimum_spanning_tree(self, *args, **kwargs):
        return self.__getattr__("minimum_spanning_tree")(*args, **kwargs)

    def minimum_weight_full_matching(self, *args, **kwargs):
        return self.__getattr__("minimum_weight_full_matching")(*args, **kwargs)

    def mixing_dict(self, *args, **kwargs):
        return self.__getattr__("mixing_dict")(*args, **kwargs)

    def mixing_expansion(self, *args, **kwargs):
        return self.__getattr__("mixing_expansion")(*args, **kwargs)

    def modular_product(self, *args, **kwargs):
        return self.__getattr__("modular_product")(*args, **kwargs)

    def modularity(self, *args, **kwargs):
        return self.__getattr__("modularity")(*args, **kwargs)

    def modularity_matrix(self, *args, **kwargs):
        return self.__getattr__("modularity_matrix")(*args, **kwargs)

    def modularity_spectrum(self, *args, **kwargs):
        return self.__getattr__("modularity_spectrum")(*args, **kwargs)

    def moebius_kantor_graph(self, *args, **kwargs):
        return self.__getattr__("moebius_kantor_graph")(*args, **kwargs)

    def moral_graph(self, *args, **kwargs):
        return self.__getattr__("moral_graph")(*args, **kwargs)

    def multi_source_dijkstra(self, *args, **kwargs):
        return self.__getattr__("multi_source_dijkstra")(*args, **kwargs)

    def multi_source_dijkstra_path(self, *args, **kwargs):
        return self.__getattr__("multi_source_dijkstra_path")(*args, **kwargs)

    def multi_source_dijkstra_path_length(self, *args, **kwargs):
        return self.__getattr__("multi_source_dijkstra_path_length")(*args, **kwargs)

    def multipartite_layout(self, *args, **kwargs):
        return self.__getattr__("multipartite_layout")(*args, **kwargs)

    def mycielski_graph(self, *args, **kwargs):
        return self.__getattr__("mycielski_graph")(*args, **kwargs)

    def mycielskian(self, *args, **kwargs):
        return self.__getattr__("mycielskian")(*args, **kwargs)

    def naive_greedy_modularity_communities(self, *args, **kwargs):
        return self.__getattr__("naive_greedy_modularity_communities")(*args, **kwargs)

    def navigable_small_world_graph(self, *args, **kwargs):
        return self.__getattr__("navigable_small_world_graph")(*args, **kwargs)

    def negative_edge_cycle(self, *args, **kwargs):
        return self.__getattr__("negative_edge_cycle")(*args, **kwargs)

    def neighbors(self, *args, **kwargs):
        return self.__getattr__("neighbors")(*args, **kwargs)

    def network_simplex(self, *args, **kwargs):
        return self.__getattr__("network_simplex")(*args, **kwargs)

    def newman_watts_strogatz_graph(self, *args, **kwargs):
        return self.__getattr__("newman_watts_strogatz_graph")(*args, **kwargs)

    def node_attribute_xy(self, *args, **kwargs):
        return self.__getattr__("node_attribute_xy")(*args, **kwargs)

    def node_boundary(self, *args, **kwargs):
        return self.__getattr__("node_boundary")(*args, **kwargs)

    def node_clique_number(self, *args, **kwargs):
        return self.__getattr__("node_clique_number")(*args, **kwargs)

    def node_connected_component(self, *args, **kwargs):
        return self.__getattr__("node_connected_component")(*args, **kwargs)

    def node_connectivity(self, *args, **kwargs):
        return self.__getattr__("node_connectivity")(*args, **kwargs)

    def node_degree_xy(self, *args, **kwargs):
        return self.__getattr__("node_degree_xy")(*args, **kwargs)

    def node_disjoint_paths(self, *args, **kwargs):
        return self.__getattr__("node_disjoint_paths")(*args, **kwargs)

    def node_expansion(self, *args, **kwargs):
        return self.__getattr__("node_expansion")(*args, **kwargs)

    def node_link_data(self, *args, **kwargs):
        return self.__getattr__("node_link_data")(*args, **kwargs)

    def node_link_graph(self, *args, **kwargs):
        return self.__getattr__("node_link_graph")(*args, **kwargs)

    def node_redundancy(self, *args, **kwargs):
        return self.__getattr__("node_redundancy")(*args, **kwargs)

    def nodes(self, *args, **kwargs):
        return self.__getattr__("nodes")(*args, **kwargs)

    def nodes_with_selfloops(self, *args, **kwargs):
        return self.__getattr__("nodes_with_selfloops")(*args, **kwargs)

    def non_edges(self, *args, **kwargs):
        return self.__getattr__("non_edges")(*args, **kwargs)

    def non_neighbors(self, *args, **kwargs):
        return self.__getattr__("non_neighbors")(*args, **kwargs)

    def non_randomness(self, *args, **kwargs):
        return self.__getattr__("non_randomness")(*args, **kwargs)

    def nonisomorphic_trees(self, *args, **kwargs):
        return self.__getattr__("nonisomorphic_trees")(*args, **kwargs)

    def normalized_cut_size(self, *args, **kwargs):
        return self.__getattr__("normalized_cut_size")(*args, **kwargs)

    def normalized_laplacian_matrix(self, *args, **kwargs):
        return self.__getattr__("normalized_laplacian_matrix")(*args, **kwargs)

    def normalized_laplacian_spectrum(self, *args, **kwargs):
        return self.__getattr__("normalized_laplacian_spectrum")(*args, **kwargs)

    def not_implemented_for(self, *args, **kwargs):
        return self.__getattr__("not_implemented_for")(*args, **kwargs)

    def null_graph(self, *args, **kwargs):
        return self.__getattr__("null_graph")(*args, **kwargs)

    def number_attracting_components(self, *args, **kwargs):
        return self.__getattr__("number_attracting_components")(*args, **kwargs)

    def number_connected_components(self, *args, **kwargs):
        return self.__getattr__("number_connected_components")(*args, **kwargs)

    def number_of_cliques(self, *args, **kwargs):
        return self.__getattr__("number_of_cliques")(*args, **kwargs)

    def number_of_edges(self, *args, **kwargs):
        return self.__getattr__("number_of_edges")(*args, **kwargs)

    def number_of_isolates(self, *args, **kwargs):
        return self.__getattr__("number_of_isolates")(*args, **kwargs)

    def number_of_nodes(self, *args, **kwargs):
        return self.__getattr__("number_of_nodes")(*args, **kwargs)

    def number_of_nonisomorphic_trees(self, *args, **kwargs):
        return self.__getattr__("number_of_nonisomorphic_trees")(*args, **kwargs)

    def number_of_selfloops(self, *args, **kwargs):
        return self.__getattr__("number_of_selfloops")(*args, **kwargs)

    def number_of_spanning_trees(self, *args, **kwargs):
        return self.__getattr__("number_of_spanning_trees")(*args, **kwargs)

    def number_of_walks(self, *args, **kwargs):
        return self.__getattr__("number_of_walks")(*args, **kwargs)

    def number_strongly_connected_components(self, *args, **kwargs):
        return self.__getattr__("number_strongly_connected_components")(*args, **kwargs)

    def number_weakly_connected_components(self, *args, **kwargs):
        return self.__getattr__("number_weakly_connected_components")(*args, **kwargs)

    def numeric_assortativity_coefficient(self, *args, **kwargs):
        return self.__getattr__("numeric_assortativity_coefficient")(*args, **kwargs)

    def octahedral_graph(self, *args, **kwargs):
        return self.__getattr__("octahedral_graph")(*args, **kwargs)

    def omega(self, *args, **kwargs):
        return self.__getattr__("omega")(*args, **kwargs)

    def one_exchange(self, *args, **kwargs):
        return self.__getattr__("one_exchange")(*args, **kwargs)

    def onion_layers(self, *args, **kwargs):
        return self.__getattr__("onion_layers")(*args, **kwargs)

    def optimal_edit_paths(self, *args, **kwargs):
        return self.__getattr__("optimal_edit_paths")(*args, **kwargs)

    def optimize_edit_paths(self, *args, **kwargs):
        return self.__getattr__("optimize_edit_paths")(*args, **kwargs)

    def optimize_graph_edit_distance(self, *args, **kwargs):
        return self.__getattr__("optimize_graph_edit_distance")(*args, **kwargs)

    def out_degree_centrality(self, *args, **kwargs):
        return self.__getattr__("out_degree_centrality")(*args, **kwargs)

    def overall_reciprocity(self, *args, **kwargs):
        return self.__getattr__("overall_reciprocity")(*args, **kwargs)

    def overlap_weighted_projected_graph(self, *args, **kwargs):
        return self.__getattr__("overlap_weighted_projected_graph")(*args, **kwargs)

    def pagerank(self, *args, **kwargs):
        return self.__getattr__("pagerank")(*args, **kwargs)

    def pairwise(self, *args, **kwargs):
        return self.__getattr__("pairwise")(*args, **kwargs)

    def paley_graph(self, *args, **kwargs):
        return self.__getattr__("paley_graph")(*args, **kwargs)

    def panther_similarity(self, *args, **kwargs):
        return self.__getattr__("panther_similarity")(*args, **kwargs)

    def pappus_graph(self, *args, **kwargs):
        return self.__getattr__("pappus_graph")(*args, **kwargs)

    def parse_adjlist(self, *args, **kwargs):
        return self.__getattr__("parse_adjlist")(*args, **kwargs)

    def parse_edgelist(self, *args, **kwargs):
        return self.__getattr__("parse_edgelist")(*args, **kwargs)

    def parse_gml(self, *args, **kwargs):
        return self.__getattr__("parse_gml")(*args, **kwargs)

    def parse_graphml(self, *args, **kwargs):
        return self.__getattr__("parse_graphml")(*args, **kwargs)

    def parse_leda(self, *args, **kwargs):
        return self.__getattr__("parse_leda")(*args, **kwargs)

    def parse_multiline_adjlist(self, *args, **kwargs):
        return self.__getattr__("parse_multiline_adjlist")(*args, **kwargs)

    def parse_pajek(self, *args, **kwargs):
        return self.__getattr__("parse_pajek")(*args, **kwargs)

    def partial_duplication_graph(self, *args, **kwargs):
        return self.__getattr__("partial_duplication_graph")(*args, **kwargs)

    def partition_quality(self, *args, **kwargs):
        return self.__getattr__("partition_quality")(*args, **kwargs)

    def partition_spanning_tree(self, *args, **kwargs):
        return self.__getattr__("partition_spanning_tree")(*args, **kwargs)

    def path_graph(self, *args, **kwargs):
        return self.__getattr__("path_graph")(*args, **kwargs)

    def path_weight(self, *args, **kwargs):
        return self.__getattr__("path_weight")(*args, **kwargs)

    def percolation_centrality(self, *args, **kwargs):
        return self.__getattr__("percolation_centrality")(*args, **kwargs)

    def periphery(self, *args, **kwargs):
        return self.__getattr__("periphery")(*args, **kwargs)

    def petersen_graph(self, *args, **kwargs):
        return self.__getattr__("petersen_graph")(*args, **kwargs)

    def planar_layout(self, *args, **kwargs):
        return self.__getattr__("planar_layout")(*args, **kwargs)

    def planted_partition_graph(self, *args, **kwargs):
        return self.__getattr__("planted_partition_graph")(*args, **kwargs)

    def power(self, *args, **kwargs):
        return self.__getattr__("power")(*args, **kwargs)

    def powerlaw_cluster_graph(self, *args, **kwargs):
        return self.__getattr__("powerlaw_cluster_graph")(*args, **kwargs)

    def predecessor(self, *args, **kwargs):
        return self.__getattr__("predecessor")(*args, **kwargs)

    def preferential_attachment(self, *args, **kwargs):
        return self.__getattr__("preferential_attachment")(*args, **kwargs)

    def preferential_attachment_graph(self, *args, **kwargs):
        return self.__getattr__("preferential_attachment_graph")(*args, **kwargs)

    def prefix_tree(self, *args, **kwargs):
        return self.__getattr__("prefix_tree")(*args, **kwargs)

    def prefix_tree_recursive(self, *args, **kwargs):
        return self.__getattr__("prefix_tree_recursive")(*args, **kwargs)

    def preflow_push(self, *args, **kwargs):
        return self.__getattr__("preflow_push")(*args, **kwargs)

    def projected_graph(self, *args, **kwargs):
        return self.__getattr__("projected_graph")(*args, **kwargs)

    def prominent_group(self, *args, **kwargs):
        return self.__getattr__("prominent_group")(*args, **kwargs)

    def quotient_graph(self, *args, **kwargs):
        return self.__getattr__("quotient_graph")(*args, **kwargs)

    def ra_index_soundarajan_hopcroft(self, *args, **kwargs):
        return self.__getattr__("ra_index_soundarajan_hopcroft")(*args, **kwargs)

    def radius(self, *args, **kwargs):
        return self.__getattr__("radius")(*args, **kwargs)

    def ramsey_R2(self, *args, **kwargs):
        return self.__getattr__("ramsey_R2")(*args, **kwargs)

    def random_clustered_graph(self, *args, **kwargs):
        return self.__getattr__("random_clustered_graph")(*args, **kwargs)

    def random_cograph(self, *args, **kwargs):
        return self.__getattr__("random_cograph")(*args, **kwargs)

    def random_degree_sequence_graph(self, *args, **kwargs):
        return self.__getattr__("random_degree_sequence_graph")(*args, **kwargs)

    def random_geometric_graph(self, *args, **kwargs):
        return self.__getattr__("random_geometric_graph")(*args, **kwargs)

    def random_graph(self, *args, **kwargs):
        return self.__getattr__("random_graph")(*args, **kwargs)

    def random_internet_as_graph(self, *args, **kwargs):
        return self.__getattr__("random_internet_as_graph")(*args, **kwargs)

    def random_k_out_graph(self, *args, **kwargs):
        return self.__getattr__("random_k_out_graph")(*args, **kwargs)

    def random_kernel_graph(self, *args, **kwargs):
        return self.__getattr__("random_kernel_graph")(*args, **kwargs)

    def random_labeled_rooted_forest(self, *args, **kwargs):
        return self.__getattr__("random_labeled_rooted_forest")(*args, **kwargs)

    def random_labeled_rooted_tree(self, *args, **kwargs):
        return self.__getattr__("random_labeled_rooted_tree")(*args, **kwargs)

    def random_labeled_tree(self, *args, **kwargs):
        return self.__getattr__("random_labeled_tree")(*args, **kwargs)

    def random_layout(self, *args, **kwargs):
        return self.__getattr__("random_layout")(*args, **kwargs)

    def random_lobster(self, *args, **kwargs):
        return self.__getattr__("random_lobster")(*args, **kwargs)

    def random_partition_graph(self, *args, **kwargs):
        return self.__getattr__("random_partition_graph")(*args, **kwargs)

    def random_powerlaw_tree(self, *args, **kwargs):
        return self.__getattr__("random_powerlaw_tree")(*args, **kwargs)

    def random_powerlaw_tree_sequence(self, *args, **kwargs):
        return self.__getattr__("random_powerlaw_tree_sequence")(*args, **kwargs)

    def random_reference(self, *args, **kwargs):
        return self.__getattr__("random_reference")(*args, **kwargs)

    def random_regular_expander_graph(self, *args, **kwargs):
        return self.__getattr__("random_regular_expander_graph")(*args, **kwargs)

    def random_regular_graph(self, *args, **kwargs):
        return self.__getattr__("random_regular_graph")(*args, **kwargs)

    def random_shell_graph(self, *args, **kwargs):
        return self.__getattr__("random_shell_graph")(*args, **kwargs)

    def random_spanning_tree(self, *args, **kwargs):
        return self.__getattr__("random_spanning_tree")(*args, **kwargs)

    def random_unlabeled_rooted_forest(self, *args, **kwargs):
        return self.__getattr__("random_unlabeled_rooted_forest")(*args, **kwargs)

    def random_unlabeled_rooted_tree(self, *args, **kwargs):
        return self.__getattr__("random_unlabeled_rooted_tree")(*args, **kwargs)

    def random_unlabeled_tree(self, *args, **kwargs):
        return self.__getattr__("random_unlabeled_tree")(*args, **kwargs)

    def randomized_partitioning(self, *args, **kwargs):
        return self.__getattr__("randomized_partitioning")(*args, **kwargs)

    def read_adjlist(self, *args, **kwargs):
        return self.__getattr__("read_adjlist")(*args, **kwargs)

    def read_edgelist(self, *args, **kwargs):
        return self.__getattr__("read_edgelist")(*args, **kwargs)

    def read_gexf(self, *args, **kwargs):
        return self.__getattr__("read_gexf")(*args, **kwargs)

    def read_gml(self, *args, **kwargs):
        return self.__getattr__("read_gml")(*args, **kwargs)

    def read_graph6(self, *args, **kwargs):
        return self.__getattr__("read_graph6")(*args, **kwargs)

    def read_graphml(self, *args, **kwargs):
        return self.__getattr__("read_graphml")(*args, **kwargs)

    def read_leda(self, *args, **kwargs):
        return self.__getattr__("read_leda")(*args, **kwargs)

    def read_multiline_adjlist(self, *args, **kwargs):
        return self.__getattr__("read_multiline_adjlist")(*args, **kwargs)

    def read_pajek(self, *args, **kwargs):
        return self.__getattr__("read_pajek")(*args, **kwargs)

    def read_sparse6(self, *args, **kwargs):
        return self.__getattr__("read_sparse6")(*args, **kwargs)

    def read_weighted_edgelist(self, *args, **kwargs):
        return self.__getattr__("read_weighted_edgelist")(*args, **kwargs)

    def reciprocity(self, *args, **kwargs):
        return self.__getattr__("reciprocity")(*args, **kwargs)

    def reconstruct_path(self, *args, **kwargs):
        return self.__getattr__("reconstruct_path")(*args, **kwargs)

    def recursive_simple_cycles(self, *args, **kwargs):
        return self.__getattr__("recursive_simple_cycles")(*args, **kwargs)

    def relabel_gexf_graph(self, *args, **kwargs):
        return self.__getattr__("relabel_gexf_graph")(*args, **kwargs)

    def relabel_nodes(self, *args, **kwargs):
        return self.__getattr__("relabel_nodes")(*args, **kwargs)

    def relaxed_caveman_graph(self, *args, **kwargs):
        return self.__getattr__("relaxed_caveman_graph")(*args, **kwargs)

    def remove_edge_attributes(self, *args, **kwargs):
        return self.__getattr__("remove_edge_attributes")(*args, **kwargs)

    def remove_node_attributes(self, *args, **kwargs):
        return self.__getattr__("remove_node_attributes")(*args, **kwargs)

    def rescale_layout(self, *args, **kwargs):
        return self.__getattr__("rescale_layout")(*args, **kwargs)

    def rescale_layout_dict(self, *args, **kwargs):
        return self.__getattr__("rescale_layout_dict")(*args, **kwargs)

    def resistance_distance(self, *args, **kwargs):
        return self.__getattr__("resistance_distance")(*args, **kwargs)

    def resource_allocation_index(self, *args, **kwargs):
        return self.__getattr__("resource_allocation_index")(*args, **kwargs)

    def restricted_view(self, *args, **kwargs):
        return self.__getattr__("restricted_view")(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        return self.__getattr__("reverse")(*args, **kwargs)

    def reverse_havel_hakimi_graph(self, *args, **kwargs):
        return self.__getattr__("reverse_havel_hakimi_graph")(*args, **kwargs)

    def reverse_view(self, *args, **kwargs):
        return self.__getattr__("reverse_view")(*args, **kwargs)

    def rich_club_coefficient(self, *args, **kwargs):
        return self.__getattr__("rich_club_coefficient")(*args, **kwargs)

    def ring_of_cliques(self, *args, **kwargs):
        return self.__getattr__("ring_of_cliques")(*args, **kwargs)

    def robins_alexander_clustering(self, *args, **kwargs):
        return self.__getattr__("robins_alexander_clustering")(*args, **kwargs)

    def rooted_product(self, *args, **kwargs):
        return self.__getattr__("rooted_product")(*args, **kwargs)

    def s_metric(self, *args, **kwargs):
        return self.__getattr__("s_metric")(*args, **kwargs)

    def scale_free_graph(self, *args, **kwargs):
        return self.__getattr__("scale_free_graph")(*args, **kwargs)

    def schultz_index(self, *args, **kwargs):
        return self.__getattr__("schultz_index")(*args, **kwargs)

    def second_order_centrality(self, *args, **kwargs):
        return self.__getattr__("second_order_centrality")(*args, **kwargs)

    def sedgewick_maze_graph(self, *args, **kwargs):
        return self.__getattr__("sedgewick_maze_graph")(*args, **kwargs)

    def selfloop_edges(self, *args, **kwargs):
        return self.__getattr__("selfloop_edges")(*args, **kwargs)

    def set_edge_attributes(self, *args, **kwargs):
        return self.__getattr__("set_edge_attributes")(*args, **kwargs)

    def set_node_attributes(self, *args, **kwargs):
        return self.__getattr__("set_node_attributes")(*args, **kwargs)

    def sets(self, *args, **kwargs):
        return self.__getattr__("sets")(*args, **kwargs)

    def shell_layout(self, *args, **kwargs):
        return self.__getattr__("shell_layout")(*args, **kwargs)

    def shortest_augmenting_path(self, *args, **kwargs):
        return self.__getattr__("shortest_augmenting_path")(*args, **kwargs)

    def shortest_path(self, *args, **kwargs):
        return self.__getattr__("shortest_path")(*args, **kwargs)

    def shortest_path_length(self, *args, **kwargs):
        return self.__getattr__("shortest_path_length")(*args, **kwargs)

    def shortest_simple_paths(self, *args, **kwargs):
        return self.__getattr__("shortest_simple_paths")(*args, **kwargs)

    def sigma(self, *args, **kwargs):
        return self.__getattr__("sigma")(*args, **kwargs)

    def simple_cycles(self, *args, **kwargs):
        return self.__getattr__("simple_cycles")(*args, **kwargs)

    def simrank_similarity(self, *args, **kwargs):
        return self.__getattr__("simrank_similarity")(*args, **kwargs)

    def simulated_annealing_tsp(self, *args, **kwargs):
        return self.__getattr__("simulated_annealing_tsp")(*args, **kwargs)

    def single_source_all_shortest_paths(self, *args, **kwargs):
        return self.__getattr__("single_source_all_shortest_paths")(*args, **kwargs)

    def single_source_bellman_ford(self, *args, **kwargs):
        return self.__getattr__("single_source_bellman_ford")(*args, **kwargs)

    def single_source_bellman_ford_path(self, *args, **kwargs):
        return self.__getattr__("single_source_bellman_ford_path")(*args, **kwargs)

    def single_source_bellman_ford_path_length(self, *args, **kwargs):
        return self.__getattr__("single_source_bellman_ford_path_length")(*args, **kwargs)

    def single_source_dijkstra(self, *args, **kwargs):
        return self.__getattr__("single_source_dijkstra")(*args, **kwargs)

    def single_source_dijkstra_path(self, *args, **kwargs):
        return self.__getattr__("single_source_dijkstra_path")(*args, **kwargs)

    def single_source_dijkstra_path_length(self, *args, **kwargs):
        return self.__getattr__("single_source_dijkstra_path_length")(*args, **kwargs)

    def single_source_shortest_path(self, *args, **kwargs):
        return self.__getattr__("single_source_shortest_path")(*args, **kwargs)

    def single_source_shortest_path_length(self, *args, **kwargs):
        return self.__getattr__("single_source_shortest_path_length")(*args, **kwargs)

    def single_target_shortest_path(self, *args, **kwargs):
        return self.__getattr__("single_target_shortest_path")(*args, **kwargs)

    def single_target_shortest_path_length(self, *args, **kwargs):
        return self.__getattr__("single_target_shortest_path_length")(*args, **kwargs)

    def snap_aggregation(self, *args, **kwargs):
        return self.__getattr__("snap_aggregation")(*args, **kwargs)

    def soft_random_geometric_graph(self, *args, **kwargs):
        return self.__getattr__("soft_random_geometric_graph")(*args, **kwargs)

    def spanner(self, *args, **kwargs):
        return self.__getattr__("spanner")(*args, **kwargs)

    def spectral_bipartivity(self, *args, **kwargs):
        return self.__getattr__("spectral_bipartivity")(*args, **kwargs)

    def spectral_bisection(self, *args, **kwargs):
        return self.__getattr__("spectral_bisection")(*args, **kwargs)

    def spectral_graph_forge(self, *args, **kwargs):
        return self.__getattr__("spectral_graph_forge")(*args, **kwargs)

    def spectral_layout(self, *args, **kwargs):
        return self.__getattr__("spectral_layout")(*args, **kwargs)

    def spectral_ordering(self, *args, **kwargs):
        return self.__getattr__("spectral_ordering")(*args, **kwargs)

    def spiral_layout(self, *args, **kwargs):
        return self.__getattr__("spiral_layout")(*args, **kwargs)

    def spring_layout(self, *args, **kwargs):
        return self.__getattr__("spring_layout")(*args, **kwargs)

    def square_clustering(self, *args, **kwargs):
        return self.__getattr__("square_clustering")(*args, **kwargs)

    def star_graph(self, *args, **kwargs):
        return self.__getattr__("star_graph")(*args, **kwargs)

    def steiner_tree(self, *args, **kwargs):
        return self.__getattr__("steiner_tree")(*args, **kwargs)

    def stochastic_block_model(self, *args, **kwargs):
        return self.__getattr__("stochastic_block_model")(*args, **kwargs)

    def stochastic_graph(self, *args, **kwargs):
        return self.__getattr__("stochastic_graph")(*args, **kwargs)

    def stoer_wagner(self, *args, **kwargs):
        return self.__getattr__("stoer_wagner")(*args, **kwargs)

    def strong_product(self, *args, **kwargs):
        return self.__getattr__("strong_product")(*args, **kwargs)

    def strongly_connected_components(self, *args, **kwargs):
        return self.__getattr__("strongly_connected_components")(*args, **kwargs)

    def subgraph(self, *args, **kwargs):
        return self.__getattr__("subgraph")(*args, **kwargs)

    def subgraph_centrality(self, *args, **kwargs):
        return self.__getattr__("subgraph_centrality")(*args, **kwargs)

    def subgraph_centrality_exp(self, *args, **kwargs):
        return self.__getattr__("subgraph_centrality_exp")(*args, **kwargs)

    def subgraph_view(self, *args, **kwargs):
        return self.__getattr__("subgraph_view")(*args, **kwargs)

    def sudoku_graph(self, *args, **kwargs):
        return self.__getattr__("sudoku_graph")(*args, **kwargs)

    def symmetric_difference(self, *args, **kwargs):
        return self.__getattr__("symmetric_difference")(*args, **kwargs)

    def tadpole_graph(self, *args, **kwargs):
        return self.__getattr__("tadpole_graph")(*args, **kwargs)

    def tensor_product(self, *args, **kwargs):
        return self.__getattr__("tensor_product")(*args, **kwargs)

    def tetrahedral_graph(self, *args, **kwargs):
        return self.__getattr__("tetrahedral_graph")(*args, **kwargs)

    def threshold_accepting_tsp(self, *args, **kwargs):
        return self.__getattr__("threshold_accepting_tsp")(*args, **kwargs)

    def thresholded_random_geometric_graph(self, *args, **kwargs):
        return self.__getattr__("thresholded_random_geometric_graph")(*args, **kwargs)

    def to_dict_of_dicts(self, *args, **kwargs):
        return self.__getattr__("to_dict_of_dicts")(*args, **kwargs)

    def to_dict_of_lists(self, *args, **kwargs):
        return self.__getattr__("to_dict_of_lists")(*args, **kwargs)

    def to_directed(self, *args, **kwargs):
        return self.__getattr__("to_directed")(*args, **kwargs)

    def to_edgelist(self, *args, **kwargs):
        return self.__getattr__("to_edgelist")(*args, **kwargs)

    def to_graph6_bytes(self, *args, **kwargs):
        return self.__getattr__("to_graph6_bytes")(*args, **kwargs)

    def to_latex(self, *args, **kwargs):
        return self.__getattr__("to_latex")(*args, **kwargs)

    def to_latex_raw(self, *args, **kwargs):
        return self.__getattr__("to_latex_raw")(*args, **kwargs)

    def to_nested_tuple(self, *args, **kwargs):
        return self.__getattr__("to_nested_tuple")(*args, **kwargs)

    def to_networkx_graph(self, *args, **kwargs):
        return self.__getattr__("to_networkx_graph")(*args, **kwargs)

    def to_numpy_array(self, *args, **kwargs):
        return self.__getattr__("to_numpy_array")(*args, **kwargs)

    def to_pandas_adjacency(self, *args, **kwargs):
        return self.__getattr__("to_pandas_adjacency")(*args, **kwargs)

    def to_pandas_edgelist(self, *args, **kwargs):
        return self.__getattr__("to_pandas_edgelist")(*args, **kwargs)

    def to_prufer_sequence(self, *args, **kwargs):
        return self.__getattr__("to_prufer_sequence")(*args, **kwargs)

    def to_scipy_sparse_array(self, *args, **kwargs):
        return self.__getattr__("to_scipy_sparse_array")(*args, **kwargs)

    def to_sparse6_bytes(self, *args, **kwargs):
        return self.__getattr__("to_sparse6_bytes")(*args, **kwargs)

    def to_undirected(self, *args, **kwargs):
        return self.__getattr__("to_undirected")(*args, **kwargs)

    def to_vertex_cover(self, *args, **kwargs):
        return self.__getattr__("to_vertex_cover")(*args, **kwargs)

    def topological_generations(self, *args, **kwargs):
        return self.__getattr__("topological_generations")(*args, **kwargs)

    def topological_sort(self, *args, **kwargs):
        return self.__getattr__("topological_sort")(*args, **kwargs)

    def transitive_closure(self, *args, **kwargs):
        return self.__getattr__("transitive_closure")(*args, **kwargs)

    def transitive_closure_dag(self, *args, **kwargs):
        return self.__getattr__("transitive_closure_dag")(*args, **kwargs)

    def transitive_reduction(self, *args, **kwargs):
        return self.__getattr__("transitive_reduction")(*args, **kwargs)

    def transitivity(self, *args, **kwargs):
        return self.__getattr__("transitivity")(*args, **kwargs)

    def traveling_salesman_problem(self, *args, **kwargs):
        return self.__getattr__("traveling_salesman_problem")(*args, **kwargs)

    def tree_all_pairs_lowest_common_ancestor(self, *args, **kwargs):
        return self.__getattr__("tree_all_pairs_lowest_common_ancestor")(*args, **kwargs)

    def tree_broadcast_center(self, *args, **kwargs):
        return self.__getattr__("tree_broadcast_center")(*args, **kwargs)

    def tree_broadcast_time(self, *args, **kwargs):
        return self.__getattr__("tree_broadcast_time")(*args, **kwargs)

    def tree_data(self, *args, **kwargs):
        return self.__getattr__("tree_data")(*args, **kwargs)

    def tree_graph(self, *args, **kwargs):
        return self.__getattr__("tree_graph")(*args, **kwargs)

    def treewidth_min_degree(self, *args, **kwargs):
        return self.__getattr__("treewidth_min_degree")(*args, **kwargs)

    def treewidth_min_fill_in(self, *args, **kwargs):
        return self.__getattr__("treewidth_min_fill_in")(*args, **kwargs)

    def triad_graph(self, *args, **kwargs):
        return self.__getattr__("triad_graph")(*args, **kwargs)

    def triad_type(self, *args, **kwargs):
        return self.__getattr__("triad_type")(*args, **kwargs)

    def triadic_census(self, *args, **kwargs):
        return self.__getattr__("triadic_census")(*args, **kwargs)

    def triads_by_type(self, *args, **kwargs):
        return self.__getattr__("triads_by_type")(*args, **kwargs)

    def triangles(self, *args, **kwargs):
        return self.__getattr__("triangles")(*args, **kwargs)

    def triangular_lattice_graph(self, *args, **kwargs):
        return self.__getattr__("triangular_lattice_graph")(*args, **kwargs)

    def trivial_graph(self, *args, **kwargs):
        return self.__getattr__("trivial_graph")(*args, **kwargs)

    def trophic_differences(self, *args, **kwargs):
        return self.__getattr__("trophic_differences")(*args, **kwargs)

    def trophic_incoherence_parameter(self, *args, **kwargs):
        return self.__getattr__("trophic_incoherence_parameter")(*args, **kwargs)

    def trophic_levels(self, *args, **kwargs):
        return self.__getattr__("trophic_levels")(*args, **kwargs)

    def truncated_cube_graph(self, *args, **kwargs):
        return self.__getattr__("truncated_cube_graph")(*args, **kwargs)

    def truncated_tetrahedron_graph(self, *args, **kwargs):
        return self.__getattr__("truncated_tetrahedron_graph")(*args, **kwargs)

    def turan_graph(self, *args, **kwargs):
        return self.__getattr__("turan_graph")(*args, **kwargs)

    def tutte_graph(self, *args, **kwargs):
        return self.__getattr__("tutte_graph")(*args, **kwargs)

    def tutte_polynomial(self, *args, **kwargs):
        return self.__getattr__("tutte_polynomial")(*args, **kwargs)

    def uniform_random_intersection_graph(self, *args, **kwargs):
        return self.__getattr__("uniform_random_intersection_graph")(*args, **kwargs)

    def union(self, *args, **kwargs):
        return self.__getattr__("union")(*args, **kwargs)

    def union_all(self, *args, **kwargs):
        return self.__getattr__("union_all")(*args, **kwargs)

    def vf2pp_all_isomorphisms(self, *args, **kwargs):
        return self.__getattr__("vf2pp_all_isomorphisms")(*args, **kwargs)

    def vf2pp_is_isomorphic(self, *args, **kwargs):
        return self.__getattr__("vf2pp_is_isomorphic")(*args, **kwargs)

    def vf2pp_isomorphism(self, *args, **kwargs):
        return self.__getattr__("vf2pp_isomorphism")(*args, **kwargs)

    def visibility_graph(self, *args, **kwargs):
        return self.__getattr__("visibility_graph")(*args, **kwargs)

    def volume(self, *args, **kwargs):
        return self.__getattr__("volume")(*args, **kwargs)

    def voronoi_cells(self, *args, **kwargs):
        return self.__getattr__("voronoi_cells")(*args, **kwargs)

    def voterank(self, *args, **kwargs):
        return self.__getattr__("voterank")(*args, **kwargs)

    def watts_strogatz_graph(self, *args, **kwargs):
        return self.__getattr__("watts_strogatz_graph")(*args, **kwargs)

    def waxman_graph(self, *args, **kwargs):
        return self.__getattr__("waxman_graph")(*args, **kwargs)

    def weakly_connected_components(self, *args, **kwargs):
        return self.__getattr__("weakly_connected_components")(*args, **kwargs)

    def weighted_projected_graph(self, *args, **kwargs):
        return self.__getattr__("weighted_projected_graph")(*args, **kwargs)

    def weisfeiler_lehman_graph_hash(self, *args, **kwargs):
        return self.__getattr__("weisfeiler_lehman_graph_hash")(*args, **kwargs)

    def weisfeiler_lehman_subgraph_hashes(self, *args, **kwargs):
        return self.__getattr__("weisfeiler_lehman_subgraph_hashes")(*args, **kwargs)

    def wheel_graph(self, *args, **kwargs):
        return self.__getattr__("wheel_graph")(*args, **kwargs)

    def wiener_index(self, *args, **kwargs):
        return self.__getattr__("wiener_index")(*args, **kwargs)

    def windmill_graph(self, *args, **kwargs):
        return self.__getattr__("windmill_graph")(*args, **kwargs)

    def within_inter_cluster(self, *args, **kwargs):
        return self.__getattr__("within_inter_cluster")(*args, **kwargs)

    def write_adjlist(self, *args, **kwargs):
        return self.__getattr__("write_adjlist")(*args, **kwargs)

    def write_edgelist(self, *args, **kwargs):
        return self.__getattr__("write_edgelist")(*args, **kwargs)

    def write_gexf(self, *args, **kwargs):
        return self.__getattr__("write_gexf")(*args, **kwargs)

    def write_gml(self, *args, **kwargs):
        return self.__getattr__("write_gml")(*args, **kwargs)

    def write_graph6(self, *args, **kwargs):
        return self.__getattr__("write_graph6")(*args, **kwargs)

    def write_graphml(self, *args, **kwargs):
        return self.__getattr__("write_graphml")(*args, **kwargs)

    def write_graphml_lxml(self, *args, **kwargs):
        return self.__getattr__("write_graphml_lxml")(*args, **kwargs)

    def write_graphml_xml(self, *args, **kwargs):
        return self.__getattr__("write_graphml_xml")(*args, **kwargs)

    def write_latex(self, *args, **kwargs):
        return self.__getattr__("write_latex")(*args, **kwargs)

    def write_multiline_adjlist(self, *args, **kwargs):
        return self.__getattr__("write_multiline_adjlist")(*args, **kwargs)

    def write_network_text(self, *args, **kwargs):
        return self.__getattr__("write_network_text")(*args, **kwargs)

    def write_pajek(self, *args, **kwargs):
        return self.__getattr__("write_pajek")(*args, **kwargs)

    def write_sparse6(self, *args, **kwargs):
        return self.__getattr__("write_sparse6")(*args, **kwargs)

    def write_weighted_edgelist(self, *args, **kwargs):
        return self.__getattr__("write_weighted_edgelist")(*args, **kwargs)


class _LazyNXProxyDynamic:
    """Lazy, cached NX (NetworkX) adapter:
    - On-demand backend conversion (no persistent NX graph).
    - Cache keyed by options until AnnNet._version changes.
    - Selective edge attr exposure (weight/capacity only when needed).
    - Clear warnings when conversion is lossy.
    - Auto label-ID mapping for vertex arguments (kwargs + positionals).
    - _nx_simple to collapse Multi* - simple Graph/DiGraph for algos that need it.
    - _nx_edge_aggs to control parallel-edge aggregation (e.g., {"capacity":"sum"}).
    """

    # -- init ---
    def __init__(self, owner: AnnNet):
        self._G = owner
        self._cache = {}  # key -> {"nxG": nx.Graph, "version": int}
        self.cache_enabled = True

    def __getattr__(self, name):
        print(f"Resolved NX attr: {name}")
        return lambda *a, **kw: 123

    #  public API
    def clear(self):
        """Drop all cached NX graphs."""
        self._cache.clear()

    def peek_vertices(self, k: int = 10):
        """Debug helper: return up to k vertex IDs visible to NX."""
        nxG = self._get_or_make_nx(
            directed=True,
            hyperedge_mode="expand",
            slice=None,
            slices=None,
            needed_attrs=set(),
            simple=False,
            edge_aggs=None,
        )
        out = []
        it = iter(nxG.nodes())
        for _ in range(max(0, int(k))):
            try:
                out.append(next(it))
            except StopIteration:
                break
        return out

    # - dynamic dispatch -
    # Public helper: obtain the cached/backend NX graph directly
    # Usage in tests: nxG = G.nx.backend(directed=False, simple=True)
    def backend(
        self,
        *,
        directed: bool = True,
        hyperedge_mode: str = "expand",
        slice=None,
        slices=None,
        needed_attrs=None,
        simple: bool = False,
        edge_aggs: dict | None = None,
    ):
        """Return the underlying NetworkX graph built with the same lazy/cached
        machinery as normal calls.

        Args:
            directed: build DiGraph (True) or Graph (False) view
            hyperedge_mode: "skip" | "expand"
            slice/slices: slice selection if Graph is multisliceed
            needed_attrs: set of edge attribute names to keep (default empty)
            simple: if True, collapse Multi* -> simple (Di)Graph
            edge_aggs: how to aggregate parallel edge attrs when simple=True,
                        e.g. {"capacity": "sum", "weight": "min"} or callables

        """
        if needed_attrs is None:
            needed_attrs = set()
        return self._get_or_make_nx(
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            slice=slice,
            slices=slices,
            needed_attrs=needed_attrs,
            simple=simple,
            edge_aggs=edge_aggs,
        )

    def __getattr__(self, name: str):
        nx_callable = self._resolve_nx_callable(name)

        def wrapper(*args, **kwargs):
            import networkx as _nx

            # Proxy-only knobs (consumed here; not forwarded to NX)
            directed = bool(kwargs.pop("_nx_directed", getattr(self, "default_directed", True)))
            hyperedge_mode = kwargs.pop(
                "_nx_hyperedge", getattr(self, "default_hyperedge_mode", "expand")
            )  # "skip" | "expand"
            slice = kwargs.pop("_nx_slice", None)
            slices = kwargs.pop("_nx_slices", None)
            label_field = kwargs.pop("_nx_label_field", None)  # explicit label column
            guess_labels = kwargs.pop("_nx_guess_labels", True)  # try auto-infer when not provided

            # force simple Graph/DiGraph and aggregation policy for parallel edges
            simple = bool(kwargs.pop("_nx_simple", getattr(self, "default_simple", False)))
            edge_aggs = kwargs.pop(
                "_nx_edge_aggs", None
            )  # e.g. {"weight":"min","capacity":"sum"} or callables

            # Determine required edge attributes (keep graph skinny)
            needed_edge_attrs = self._needed_edge_attrs(nx_callable, kwargs)

            # Do NOT auto-inject G. Only convert/replace if the user passed our Graph.
            args = list(args)
            has_owner_graph = any(a is self._G for a in args) or any(
                v is self._G for v in kwargs.values()
            )

            # Build backend ONLY if we actually need to replace self._G
            nxG = None
            if has_owner_graph:
                nxG = self._get_or_make_nx(
                    directed=directed,
                    hyperedge_mode=hyperedge_mode,
                    slice=slice,
                    slices=slices,
                    needed_attrs=needed_edge_attrs,
                    simple=simple,
                    edge_aggs=edge_aggs,
                )

            # Replace any occurrence of our Graph with the NX backend
            if nxG is not None:
                for i, v in enumerate(args):
                    if v is self._G:
                        args[i] = nxG
                for k, v in list(kwargs.items()):
                    if v is self._G:
                        kwargs[k] = nxG

            # Bind to NX signature so we can coerce vertex args (no defaults!)
            bound = None
            try:
                sig = inspect.signature(nx_callable)
                bound = sig.bind_partial(*args, **kwargs)
            except Exception:
                pass

            # Coerce vertex args (labels/indices -> vertex IDs)
            try:
                # Determine default label field if not given
                if label_field is None and guess_labels:
                    label_field = self._infer_label_field()

                if bound is not None and nxG is not None:
                    self._coerce_vertices_in_bound(bound, nxG, label_field)
                    # Reconstruct WITHOUT applying defaults (avoid flow_func=None, etc.)
                    pargs = bound.args
                    pkwargs = bound.kwargs
                else:
                    # Fallback: best-effort coercion on kwargs only
                    if nxG is not None:
                        self._coerce_vertices_in_kwargs(kwargs, nxG, label_field)
                    pargs, pkwargs = tuple(args), kwargs
            except Exception:
                pargs, pkwargs = tuple(args), kwargs  # best effort; let NX raise if needed

            # Never leak private knobs to NX
            for k in list(pkwargs.keys()):
                if isinstance(k, str) and k.startswith("_nx_"):
                    pkwargs.pop(k, None)

            from networkx.exception import NodeNotFound

            try:
                raw = nx_callable(*pargs, **pkwargs)
                return self._map_output_vertices(raw)
            except NodeNotFound as e:
                # Add actionable tip that actually tells how to fix it now.
                sample = self.peek_vertices(5)
                tip = (
                    f"{e}. vertices must be graph's vertex IDs.\n"
                    f"- If you passed labels, specify _nx_label_field=<vertex label column> "
                    f"or rely on auto-guess (columns like 'name'/'label'/'title').\n"
                    f"- Example: G.nx.shortest_path_length(G, source='a', target='z', weight='weight', _nx_label_field='name')\n"
                    f"- A few vertex IDs NX sees: {sample}"
                )
                raise _nx.NodeNotFound(tip) from e

        return wrapper

    # -- internals ---
    def _resolve_nx_callable(self, name: str):
        import networkx as _nx

        candidates = [
            _nx,
            getattr(_nx, "algorithms", None),
            getattr(_nx.algorithms, "community", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx.algorithms, "approximation", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx.algorithms, "centrality", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx.algorithms, "shortest_paths", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx.algorithms, "flow", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx.algorithms, "components", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx.algorithms, "traversal", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx.algorithms, "bipartite", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx.algorithms, "link_analysis", None) if hasattr(_nx, "algorithms") else None,
            getattr(_nx, "classes", None),
            getattr(_nx.classes, "function", None) if hasattr(_nx, "classes") else None,
        ]
        for mod in (m for m in candidates if m is not None):
            attr = getattr(mod, name, None)
            if callable(attr):
                return attr
        raise AttributeError(f"networkx has no callable '{name}'")

    def _needed_edge_attrs(self, target, kwargs) -> set:
        needed = set()
        # weight
        w_name = kwargs.get("weight", "weight")
        try:
            sig = inspect.signature(target)
            if "weight" in sig.parameters and w_name is not None:
                needed.add(str(w_name))
        except Exception:
            if "weight" in kwargs and w_name is not None:
                needed.add(str(w_name))
        # capacity (flows)
        c_name = kwargs.get("capacity", "capacity")
        try:
            sig = inspect.signature(target)
            if "capacity" in sig.parameters and c_name is not None:
                needed.add(str(c_name))
        except Exception:
            if "capacity" in kwargs and c_name is not None:
                needed.add(str(c_name))
        return needed

    def _convert_to_nx(
        self,
        *,
        directed: bool,
        hyperedge_mode: str,
        slice,
        slices,
        needed_attrs: set,
        simple: bool,
        edge_aggs: dict | None,
    ):
        from ...adapters import networkx_adapter as _gg_nx  # annnet.adapters.networkx_adapter

        nxG, manifest = _gg_nx.to_nx(
            self._G,
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            slice=slice,
            slices=slices,
            public_only=True,
        )
        # Keep only needed edge attrs
        if needed_attrs:
            for _, _, _, d in nxG.edges(keys=True, data=True):
                for k in list(d.keys()):
                    if k not in needed_attrs:
                        d.pop(k, None)
        else:
            # allow collapsing to use weight/capacity defaults
            if simple:
                # keep weight/capacity if present
                for _, _, _, d in nxG.edges(keys=True, data=True):
                    for k in list(d.keys()):
                        if k not in ("weight", "capacity"):
                            d.pop(k, None)
            else:
                # normal slimming
                for _, _, _, d in nxG.edges(keys=True, data=True):
                    d.clear()

        # Collapse Multi* - simple Graph/DiGraph if requested
        if simple and nxG.is_multigraph():
            nxG = self._collapse_multiedges(
                nxG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
            )

        self._warn_on_loss(
            hyperedge_mode=hyperedge_mode, slice=slice, slices=slices, manifest=manifest
        )
        return nxG

    def _get_or_make_nx(
        self,
        *,
        directed: bool,
        hyperedge_mode: str,
        slice,
        slices,
        needed_attrs: set,
        simple: bool,
        edge_aggs: dict | None,
    ):
        key = (
            bool(directed),
            str(hyperedge_mode),
            tuple(sorted(slices)) if slices else None,
            str(slice) if slice is not None else None,
            tuple(sorted(needed_attrs)) if needed_attrs else (),
            bool(simple),
            tuple(sorted(edge_aggs.items())) if isinstance(edge_aggs, dict) else None,
        )
        version = getattr(self._G, "_version", None)
        entry = self._cache.get(key)
        if (
            (not self.cache_enabled)
            or (entry is None)
            or (version is not None and entry.get("version") != version)
        ):
            nxG = self._convert_to_nx(
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                needed_attrs=needed_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )
            if self.cache_enabled:
                self._cache[key] = {"nxG": nxG, "version": version}
            return nxG
        return entry["nxG"]

    def _warn_on_loss(self, *, hyperedge_mode, slice, slices, manifest):
        import warnings

        has_hyper = False
        try:
            ek = getattr(self._G, "edge_kind", {})  # dict[eid] -> "hyper"/"binary"
            if hasattr(ek, "values"):
                has_hyper = any(str(v).lower() == "hyper" for v in ek.values())
        except Exception:
            pass
        msgs = []
        if has_hyper and hyperedge_mode != "expand":
            msgs.append("hyperedges dropped (hyperedge_mode='skip')")
        try:
            slices_dict = getattr(self._G, "_slices", None)
            if (
                isinstance(slices_dict, dict)
                and len(slices_dict) > 1
                and (slice is None and not slices)
            ):
                msgs.append("multiple slices flattened into single NX graph")
        except Exception:
            pass
        if manifest is None:
            msgs.append("no manifest provided; round-trip fidelity not guaranteed")
        if msgs:
            warnings.warn(
                "AnnNet-NX conversion is lossy: " + "; ".join(msgs) + ".",
                category=RuntimeWarning,
                stacklevel=3,
            )

    # -- label/ID mapping helpers
    def _infer_label_field(self) -> str | None:
        """Heuristic label column if user didn't specify:
        1) AnnNet.default_label_field if present
        2) first present in ["name","label","title","slug","external_id","string_id"]
        """
        try:
            if hasattr(self._G, "default_label_field") and self._G.default_label_field:
                return self._G.default_label_field
            va = getattr(self._G, "vertex_attributes", None)
            cols = list(va.columns) if va is not None and hasattr(va, "columns") else []
            for c in ("name", "label", "title", "slug", "external_id", "string_id"):
                if c in cols:
                    return c
        except Exception:
            pass
        return None

    def _vertex_id_col(self) -> str:
        """Best-effort to determine the vertex ID column name in vertex_attributes."""
        try:
            va = self._G.vertex_attributes
            cols = list(va.columns)
            for k in ("vertex_id", "id", "vid"):
                if k in cols:
                    return k
        except Exception:
            pass
        return "vertex_id"

    def _lookup_vertex_id_by_label(self, label_field: str, val):
        """Return vertex_id where vertex_attributes[label_field] == val, else None."""
        try:
            va = self._G.vertex_attributes
            if va is None or not hasattr(va, "columns") or label_field not in va.columns:
                return None
            id_col = self._vertex_id_col()
            # Prefer polars path
            try:
                # type: ignore

                matches = va.filter(pl.col(label_field) == val)
                if matches.height == 0:
                    return None
                try:
                    return matches.select(id_col).to_series().to_list()[0]
                except Exception:
                    return matches.select(id_col).item(0, 0)
            except Exception:
                # Fallback: convert to dicts (slower; fine for ad-hoc lookups)
                for row in va.to_dicts():
                    if row.get(label_field) == val:
                        return row.get(id_col)
        except Exception:
            return None
        return None

    def _coerce_vertex_id(self, x, nxG, label_field: str | None):
        # 1) If x is an internal row index, convert to entity_id FIRST
        try:
            if isinstance(x, int) and x in getattr(self._G, "idx_to_entity", {}):
                cand = self._G.idx_to_entity[x]
                if getattr(self._G, "entity_types", {}).get(cand) == "vertex":
                    x = cand
        except Exception:
            pass

        # 2) If already a valid NX node, return it
        if x in nxG:
            return x

        # 3) Label-based lookup
        if label_field:
            try:
                cand = self._lookup_vertex_id_by_label(label_field, x)
                if cand is not None:
                    return cand
            except Exception:
                pass

        # 4) Let NX raise NodeNotFound later
        return x

    def _coerce_vertex_or_iter(self, obj, nxG, label_field: str | None):
        if isinstance(obj, (list, tuple, set)):
            coerced = [self._coerce_vertex_id(v, nxG, label_field) for v in obj]
            return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
        return self._coerce_vertex_id(obj, nxG, label_field)

    def _coerce_vertices_in_kwargs(self, kwargs: dict, nxG, label_field: str | None):
        vertex_keys = {
            "source",
            "target",
            "u",
            "v",
            "vertex",
            "vertices",
            "nbunch",
            "center",
            "path",
        }
        for key in list(kwargs.keys()):
            if key in vertex_keys:
                kwargs[key] = self._coerce_vertex_or_iter(kwargs[key], nxG, label_field)

    def _coerce_vertices_in_bound(self, bound, nxG, label_field: str | None):
        """Coerce vertices in a BoundArguments object using common vertex parameter names."""
        vertex_keys = {
            "source",
            "target",
            "u",
            "v",
            "vertex",
            "vertices",
            "nbunch",
            "center",
            "path",
        }
        for key in list(bound.arguments.keys()):
            if key in vertex_keys:
                bound.arguments[key] = self._coerce_vertex_or_iter(
                    bound.arguments[key], nxG, label_field
                )

    def _map_output_vertices(self, obj):
        """Recursively map NX output structures’ vertex keys and values to internal row indices."""
        G = self._G
        id2row = G.entity_to_idx  # entity_id -> row index

        # --- Node ID mapping helper ---
        def map_id(x):
            if isinstance(x, str) and x in id2row:
                return id2row[x]
            if isinstance(x, int) and x in getattr(G, "idx_to_entity", {}):
                # already mapped earlier but keep for consistency
                return x
            return x

        # --- Dict: map keys and recursively map values ---
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                mapped_k = map_id(k)
                out[mapped_k] = self._map_output_vertices(v)
            return out

        # --- List ---
        if isinstance(obj, list):
            return [self._map_output_vertices(x) for x in obj]

        # --- Tuple ---
        if isinstance(obj, tuple):
            return tuple(self._map_output_vertices(x) for x in obj)

        # --- Set ---
        if isinstance(obj, set):
            return {self._map_output_vertices(x) for x in obj}

        # --- Base case: try mapping ID ---
        return map_id(obj)

    # -- Multi* collapse helpers -

    def _collapse_multiedges(
        self, nxG, *, directed: bool, aggregations: dict | None, needed_attrs: set
    ):
        """Collapse parallel edges into a single edge with aggregated attributes.
        Defaults: weight -> min (good for shortest paths), capacity -> sum (good for max-flow).
        """
        import networkx as _nx

        H = _nx.DiGraph() if directed else _nx.Graph()
        H.add_nodes_from(nxG.nodes(data=True))

        aggregations = aggregations or {}

        def _agg_for(key):
            agg = aggregations.get(key)
            if callable(agg):
                return agg
            if agg == "sum":
                return sum
            if agg == "min":
                return min
            if agg == "max":
                return max
            # sensible defaults:
            if key == "capacity":
                return sum
            if key == "weight":
                return min
            # fallback: first value
            return lambda vals: next(iter(vals))

        # Bucket parallel edges
        bucket = {}  # (u,v) or sorted(u,v) -> {attr: [values]}
        for u, v, _, d in nxG.edges(keys=True, data=True):
            key = (u, v) if directed else tuple(sorted((u, v)))
            entry = bucket.setdefault(key, {})
            for k, val in d.items():
                if needed_attrs and k not in needed_attrs:
                    continue
                entry.setdefault(k, []).append(val)

        # Aggregate per (u,v)
        for (u, v), attrs in bucket.items():
            out = {k: _agg_for(k)(vals) for k, vals in attrs.items()}
            H.add_edge(u, v, **out)

        return H


class _LazyNXProxy(_LazyNXProxyDynamic, _LazyNXProxyAutogen):
    """Final NX proxy with both autocomplete and dynamic fallback."""

    pass
