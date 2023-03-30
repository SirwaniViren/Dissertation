"""
Author: Guillermo Romero Moreno
Date: 10/2/2022

This file contains bespoke graph classes for the construction of different patient and morbidity networks.
"""
import os
import time
from gi.repository import Gtk, Gdk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import graph_tool.all as gt

from networks import pyintergraph, projection, backboning
from networks.nx2gt import nx2gt

from mulmorb_wranglr.constants import CLEANED_DATA_FILE2

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TEMP_FOLDER = SCRIPT_DIR + "/temp/"


def plot_hist_from_bins_and_counts(counts, bins, ax):
    assert len(bins) == len(counts) + 1
    # recover
    centroids = (bins[1:] + bins[:-1]) / 2
    counts_, bins_, _ = ax.hist(centroids, bins=len(counts), weights=counts, range=(min(bins), max(bins)))


class HealthNetwork(gt.Graph):
    def __init__(self, data, cut_points_dict=None, from_gt_graph=None):
        self.data = data
        if from_gt_graph is not None:
            super().__init__(from_gt_graph)
        else:
            super().__init__(directed=False)
            self.discretise_features(cut_points_dict)

    def discretise_features(self, cut_points_dict=None):
        self.feature_sets = {feat_type: self.data.discretise_features(features, cut_points_dict=cut_points_dict) for
                             feat_type, features in self.data.cols.items()}

    def get_all_partitions(self, feature_type):
        return {key: val for partition_dict in self.feature_sets[feature_type].values() for key, val in
                partition_dict.items()}

    def get_partition(self, partition_name):
        for feature_set in self.feature_sets.values():
            for partition_dict in feature_set.values():
                if partition_name in partition_dict:
                    return partition_dict[partition_name]
        raise Exception(f"Partition with name {partition_name} not found.")


class IndividualToFeatureBipartite(HealthNetwork):
    def __init__(self, data, cut_points_dict=None):  # , morb_cols=None, demo_cols=None):
        """
        This class generates a bipartite graph from a dataset where one partite are individuals and the other partite are features.
        
        :param: data: pandas DataFrame. Rows are individuals, columns are features.
        """
        t_ini = time.time()
        super().__init__(data, cut_points_dict=cut_points_dict)

        n_patients = len(data)
        print("N of patients:", n_patients)

        self.add_vertex(n_patients)
        self.vp.patient = self.new_vp("bool", val=True)
        self.vp.name = self.new_vp("string")
        for node_type in self.feature_sets:
            self.vp[node_type] = self.new_vp("bool")

            edge_list = None
            for feature_name, partition_dict in self.feature_sets[node_type].items():
                self.vp[feature_name] = self.new_vp("bool")

                for part_name, part_mask in partition_dict.items():
                    patients_in_partition = np.arange(n_patients)[part_mask]
                    if len(patients_in_partition) > 0:
                        new_v = self.add_vertex()
                        self.vp.name[new_v] = part_name
                        self.vp[node_type][new_v] = True
                        self.vp[feature_name][new_v] = True

                        sub_list = np.vstack([patients_in_partition, np.ones_like(patients_in_partition) * int(new_v)])
                        edge_list = np.hstack([edge_list, sub_list]) if edge_list is not None else sub_list

            self.add_edge_list(edge_list.T)
            print(f"{node_type} features: {self.vp[node_type].a.sum()}")

        print(f"N of vertices: {self.num_vertices()}, N of edges:", self.num_edges())
        assert gt.is_bipartite(self)
        # TODO: Also include sanity checks for each feature: each patient should only have a single link to one partition of the same feature

        print(f"Bipartite network construction finished (Elapsed time:{time.time() - t_ini:.2f}s)")

        self.nx_g = None

    def get_morbidity_patient_subgraph(self):
        return gt.GraphView(self, vfilt=self.vp.morb.a.astype(bool) | self.vp.patient.a.astype(bool))

    def plot_degree_distribution(self, node_type):
        """
        Plot a histogram with nodes' degree distribution. Both in and out edges are considered.
        """
        assert node_type in self.feature_sets or node_type == "patient"

        vertices = gt.find_vertex(self, self.vp[node_type], True)
        degrees = self.get_all_degrees(vertices)

        fig, ax = plt.subplots(figsize=(18, 6))
        if node_type in ("morb", "demo"):
            names = [self.vp.name[el] for el in vertices]
            ax.bar(names, degrees)
            ax.set(xlabel=node_type, ylabel="# of patients")
            plt.xticks(rotation=90)

        elif node_type == "patient":
            ax.hist(degrees, bins=16, range=(-0.5, 15.5))
            ax.set(xlabel="# of features", ylabel="# of patients")
            ax.set_yscale("log")

        fig.show()

    def plot_distance_histogram(self, samples=100):
        """
        Plots a histogram of distances between nodes in the network.
        :param: samples (default=100): number of samples from which to compute the distance wrt to all nodes in the network. Computational time is |V| x samples.
        """
        t_ini = time.time()
        fig, ax = plt.subplots(figsize=(18, 4))

        dhist = gt.distance_histogram(self, samples=samples)
        ax.bar(dhist[1][:-1], dhist[0])
        ax.set(xlabel="distance", ylabel="n of node pairs")
        print(f"Plotting distance histogram with {samples} samples. Elapsed time:{time.time() - t_ini:.2f}s")

    def project(self, method="hybrid", directed=False, l=0.5):
        if self.nx_g is None:
            print("Transforming the network to `networkx`...")
            t_ini = time.time()
            self.nx_g = pyintergraph.gt2nx(self)
            print("Elapsed time:{:.2f}s".format(time.time() - t_ini))

        nodes = nx.algorithms.bipartite.basic.sets(self.nx_g)
        rows = sorted(list(nodes[0]))
        cols = sorted(list(nodes[1]))
        print("Size of the projected graph:", len(projected_nodes := rows if len(rows) < len(cols) else cols))

        print(f"Projecting the graph with method '{method}'...")
        t_ini = time.time()
        kwargs = {"directed": directed} if method in ("ycn", "probs", "heats", "hybrid") else {}
        kwargs.update({"l": l}) if method == "hybrid" else None
        projected_bipartite_nx = projection.__dict__[method](self.nx_g, projected_nodes, **kwargs)

        attributes = {n_id: node_attributes for i, (n_id, node_attributes) in enumerate(self.nx_g.nodes.data(True)) if
                      i >= len(self.data)}
        nx.set_node_attributes(projected_bipartite_nx, attributes)
        print("Elapsed time:{:.2f}s".format(time.time() - t_ini))

        projected_bipartite = CooccurrenceNetwork(self.data, from_gt_graph=nx2gt(projected_bipartite_nx),
                                                  link_type=method, feature_sets=self.feature_sets)

        # sl_labels = projected_bipartite.new_ep("bool")
        # gt.label_self_loops(projected_bipartite, mark_only=True, eprop=sl_labels)
        # if any(sl_labels):
        #    print("Projection method includes self-loops. These will be removed.")
        #    #gt.remove_self_loops(projected_bipartite)
        #    gt.remove_labeled_edges(projected_bipartite, sl_labels)
        return projected_bipartite


class CooccurrenceNetwork(HealthNetwork):
    def __init__(self, data, link_type="count", cut_points_dict=None, from_gt_graph=None,
                 feature_sets=None):  # , morb_cols=None, demo_cols=None):
        """
        This class generates a co-occurrence network from a dataset where nodes are features and links are the number of
         patients sharing such feature.
        
        :param: data: pandas DataFrame. Rows are individuals, columns are features.
        :param: link_type (str): "count", "RR" (Relative Risk)
        :param: morb_cols (None): Column names of data to be selected as morbidity features for constructing the graph.
        If None, extract them from DataFrame object.
        :param: demo_cols (None): Column names of data to be selected as demographic features for constructing the
        graph. If None, extract the from the DataFrame object.
        """
        self.link_type = link_type
        self.feature_sets = feature_sets

        t_ini = time.time()
        super().__init__(data, cut_points_dict=cut_points_dict, from_gt_graph=from_gt_graph)

        if from_gt_graph is None:
            self.ep.weight = self.new_ep("float")
            self.vp.name = self.new_vp("string")
            self._add_features(link_type)

            print(f"Network built (Elapsed time:{time.time() - t_ini:.2f}s)")
            print(f"N of vertices: {self.num_vertices()}, N of edges: {self.num_edges()}")

    def _add_features(self, link_type="count"):
        """
        :param: link_type (str): "count", "RR" (Relative Risk), "cosine" (Salton Cosine Index), "phi" (Pearson's
        """
        for feature_type in self.feature_sets:
            self.vp[feature_type] = self.new_vp("bool")

            for partition_name, partition_mask in self.get_all_partitions(feature_type).items():
                v = self.add_vertex()
                self.vp.name[v] = partition_name
                self.vp[feature_type][v] = True

                if (num_nodes := self.num_vertices()) > 1:
                    for i in self.get_vertices():
                        if i == num_nodes - 1:
                            break

                        partition_mask2 = self.get_partition(self.vp.name[i])
                        N = len(partition_mask)
                        P1, P2 = partition_mask.sum() / N, partition_mask2.sum() / N
                        if (P12 := (partition_mask & partition_mask2).sum() / N) > 0:
                            new_e = self.add_edge(v, i)

                            if link_type == "count":
                                self.ep.weight[new_e] = P12 * N
                            elif link_type == "RR":
                                self.ep.weight[new_e] = P12 / P1 / P2
                            elif link_type == "cosine":
                                self.ep.weight[new_e] = P12 / np.sqrt(P1) / np.sqrt(P2)
                            elif link_type == "phi":
                                self.ep.weight[new_e] = (P12 - P1 * P2) / np.sqrt(P1) / np.sqrt(P2) / np.sqrt(
                                    1 - P1) / np.sqrt(1 - P2)
                            elif link_type == "ABC":
                                self.ep.weight[new_e] = (P12 - P1 * P2) * (1 - P1) * (1 - P2) / (P1 - P12) / (
                                            P2 - P12) / (1 - P1 - P2 + P12)
                            else:
                                raise Exception(f"Link type '{link_type}' not understood.")

    def get_adjacency(self, sparse=True):
        A = gt.adjacency(self, weight=self.ep.weight)
        return A if sparse else A.toarray()

    def plot(self, size="edge_linear", layout=None, interactive=False):
        """
        Network visualisation where node size is proportional to its weighted degree (both in and out).
        """
        v_style = {"text": self.vp.name,
                   "text_color": "k",
                   "text_position": -7,  # Negative values place text inside, except for -1
                   "fill_color": self.vp.morb,
                   "pen_width": 1,
                   }

        assert type(size) == str or not size, f"Size parameter '{size}' not understood."
        if type(size) == str:
            MAX_SIZE, TEXT_FACTOR = 70, 0.2

            def compute_sizes(size_scale):
                if size_scale in ("edge_linear", "edge_log"):
                    sizes = np.array([max(self.ep.weight[e] for e in v.all_edges()) if len(
                        self.get_all_edges(v)) > 0 else 0 for v in self.vertices()])

                    if size_scale == "edge_log":
                        sizes[np.isclose(sizes, 0)] = sizes[sizes > 0].min() / 100
                        sizes = np.log(sizes)
                        sizes += sizes.min()

                elif size_scale == "degree_linear":
                    sizes = self.get_all_degrees(self.get_vertices(), eweight=self.ep.weight)
                else:
                    raise Exception(f"Size parameter '{size_scale}' not understood.")

                sizes = (sizes / sizes.max()) * MAX_SIZE
                return sizes

            ss, font_sizes = self.new_vp("float"), self.new_vp("float")
            default_sizes = compute_sizes(size)

            def reset_sizes():
                ss.a = default_sizes
                font_sizes.a = default_sizes * TEXT_FACTOR

            reset_sizes()
            v_style.update({"size": ss, "font_size": 10})  # font_sizes})

        draw_kwargs = {"edge_pen_width": self.new_ep("float", vals=self.ep.weight.a / self.ep.weight.a.max()),
                       "vprops": v_style, }

        if layout == "arf":
            draw_kwargs["pos"] = gt.arf_layout(self)
        elif layout == "weighted_sfdp":
            draw_kwargs["pos"] = gt.sfdp_layout(self, eweight=self.ep.weight)  # , C=-0.1)
        elif layout is not None:
            raise Exception(f"Graph drawing layout '{layout}' not understood.")

        if not interactive:
            gt.graph_draw(self, output_size=(1000, 1000), **draw_kwargs)
        else:
            self.win = gt.GraphWindow(self, geometry=(500, 400), **draw_kwargs)

            self.old_src = None

            def update_bfs(widget, event):
                self.old_src
                # retrieve selected node
                src = widget.picked
                if src is None:
                    return True
                if isinstance(src, gt.PropertyMap):
                    src = [v for v in self.vertices() if src[v]]
                    if len(src) == 0:
                        return True
                    src = src[0]
                if src == self.old_src:
                    return True
                self.old_src = src

                reset_sizes()
                if src:
                    def assign_size(vertex, value):
                        ss[vertex] = value
                        font_sizes[vertex] = value * TEXT_FACTOR

                    assign_size(src, MAX_SIZE)
                    max_weight = max(self.ep.weight[e] for e in src.out_edges())
                    # TODO: reduce non-neighbours to min_weight
                    for e in src.out_edges():
                        if e is not None:
                            assign_size(e.target(), self.ep.weight[e] / max_weight * MAX_SIZE)

                widget.regenerate_surface()
                widget.queue_draw()

            # Bind the function above as a montion notify handler
            self.win.graph.connect("motion_notify_event", update_bfs)

            # We will give the user the ability to stop the program by closing the window.
            self.win.connect("delete_event", Gtk.main_quit)

            # Actually show the window, and start the main loop.
            self.win.show_all()
            Gtk.main()

    def plot_edge_histogram(self, bins=10, log=False):
        bins = ((emin := self.ep.weight.a.min()), (self.ep.weight.a.max() - emin) / bins)
        ehist = gt.edge_hist(self, eprop=self.ep.weight, bins=bins)

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_hist_from_bins_and_counts(*ehist, ax)
        ax.set(xlabel="weight", ylabel="n of edges")
        ax.set_yscale("log") if log else None

    def filter_edges(self, method="noise_corrected", delta=1):
        """
        :param: method (str, default="noise_corrected").
        :param: delta (int, default=3). Parameter for 'noise_corrected' method.
        """

        print(f"Filtering the network via the '{method}' method.")
        self.set_edge_filter(None)  # inverted=True)
        if method is None:
            return

        if method == "below_avg":
            self.set_edge_filter(self.new_ep("bool", vals=self.ep.weight.a / self.ep.weight.a.mean() > 1))
            return

        # TODO: this is very dirty. I should re-implement the code to directly work with graph data without having to
        #  pass through the pandas Dataframe
        table = pd.DataFrame(
            [{"src": e.source(), "trg": e.target(), "weight": self.ep.weight[e]} for e in self.edges()])
        table_path = TEMP_FOLDER + "table.csv"
        table.to_csv(table_path, sep="\t")
        table, nnodes, nnedges = backboning.read(table_path, "weight", drop_zeroes=False)

        if method == "noise_corrected":
            nc_table = backboning.noise_corrected(table)
            nc_backbone = backboning.thresholding(nc_table, threshold=delta)
        elif method in (
                "doubly_stochastic", "disparity_filter", "high_salience_skeleton", "naive", "maximum_spanning_tree"):
            nc_backbone = backboning.__dict__[method](table)
        else:
            raise Exception(f"Filtering method '{method}' not understood.")

        e_bool_pmap = self.new_ep("bool")
        for e in self.edges():
            row = nc_backbone.loc[(nc_backbone["src"] == e.source()) & (nc_backbone["trg"] == e.target())]
            assert len(row) < 2, f"{e.source()}, {e.target()} \n {self.ep.weight[e]}, {row}"

            if len(row) == 1:
                e_bool_pmap[e] = 1
            # row = nc_table.loc[(nc_table["src"] == e.source()) & (nc_table["trg"] == e.target())]
            # assert len(row) == 1, f"{e.source()}, {e.target()}, {row}"
            # if row.score.values[0] > delta * row.sdev_cij.values[0]:
            #    e_bool_pmap[e] = 1

        print(f"Filtering finished. Keeping {e_bool_pmap.a.sum()} edges out of {self.num_edges()}")
        self.set_edge_filter(e_bool_pmap)  # inverted=True)


if __name__ == "__main__":
    from PCCIU.data import PCCIU_DataFrame

    df = PCCIU_DataFrame()  # nrows=1e5)
    g = IndividualToFeatureBipartite(df)
