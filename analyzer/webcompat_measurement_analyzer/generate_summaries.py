# vim: set tw=99 ts=4 sts=4 sw=4 et:

# Copyright (C) 2022-23 Michael Smith <michael@spinda.net> (https://spinda.net)

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import functools
import json
import multiprocessing
import os

import networkx
from networkx import MultiDiGraph

from ._lib.extract import is_uniquely_blocked_by_alt
from ._lib.paths import get_sharded_path, snapshots_dir_path, subgraphs_dir_path, summaries_dir_path

_concurrency = 8
_chunk_size = 128

_category_variants = {
    'broken': ('child', 'parent'),
    'unbroken': ('parent', 'child')
}

_blacklisted_node_keys = {'id', 'node id', 'script id', 'timestamp'}
_blacklisted_edge_keys = {'event listener id', 'id', 'request id', 'response hash', 'timestamp'}

def _summarize_graph(graph):
    source_node_ids = []
    for node_id, node_data in graph.nodes(data=True):
        if node_data['node type'] == 'resource':
            if is_uniquely_blocked_by_alt(graph, node_id):
                source_node_ids.append(node_id)

    summary = []
    summarized_node_ids = set()

    # for node_id, node_data in graph.nodes(data=True):
    for node_id_a, node_id_b, _ in networkx.algorithms.traversal.edgebfs.edge_bfs(
        graph, source=source_node_ids
    ):
        for node_id in [node_id_a, node_id_b]:
            if node_id in summarized_node_ids:
                continue
            summarized_node_ids.add(node_id)

            node_data = graph.nodes[node_id]
            for key, value in node_data.items():
                if key in _blacklisted_node_keys:
                    continue
                summary.append(str(key))
                summary.append(str(value))

            for out_node_id, in_node_id, edge_data in graph.out_edges(
                data=True, nbunch=node_id
            ):
                for key, value in edge_data.items():
                    if key in _blacklisted_edge_keys:
                        continue
                    summary.append(str(key))
                    summary.append(str(value))

    # summarized_nodes = set()

    # def summarize_node(node_id):
        # if node_id in summarized_nodes:
            # return
        # node_data = graph.nodes[node_id]
        # for key, value in node_data.items():
            # summary.append(str(key))
            # summary.append(str(value))

    # def summarize_edge(out_node_id, in_node_id, edge_key):
        # edge_data = graph[out_node_id][in_node_id][edge_key]
        # for key, value in edge_data.items():
            # summary.append(str(key))
            # summary.append(str(value))

    # for out_node_id, in_node_id, edge_key in networkx.algorithms.traversal.edgebfs.edge_bfs(
        # graph
    # ):
        # summarize_node(out_node_id)
        # summarize_edge(out_node_id, in_node_id, edge_key)
        # summarize_node(in_node_id)

    # summary.sort()
    # log(summary)

    return summary

def _generate(pad_length, subgraph_count, category, commit, url_count):
    def log(*args, **kwargs):
        print(f'[{subgraph_count: >{pad_length}}]', *args, **kwargs)

    log(f'** Generating summary for {category} subgraph from {commit} URL #{url_count} (#{subgraph_count})...')

    category_summaries_dir_path = os.path.join(summaries_dir_path, category)
    commit_summaries_dir_path = get_sharded_path(category_summaries_dir_path, commit)
    summary_file_path = os.path.join(commit_summaries_dir_path, f'{url_count}.json')
    if os.path.exists(summary_file_path):
        log('*** Skipped (summary file already exists)')
        return

    category_subgraphs_dir_path = os.path.join(subgraphs_dir_path, category)
    commit_subgraphs_dir_path = get_sharded_path(category_subgraphs_dir_path, commit)
    subgraph_file_path = os.path.join(commit_subgraphs_dir_path, f'{url_count}.graphml')

    log('*** Reading subgraph...')

    with open(subgraph_file_path, 'r', encoding='utf-8') as subgraph_file:
        subgraph_src = subgraph_file.read()
    subgraph = MultiDiGraph(networkx.graphml.parse_graphml(subgraph_src))

    log('*** Walking over subgraph...')

    subgraph_summary = _summarize_graph(subgraph)

    # category_snapshots_dir_path = os.path.join(snapshots_dir_path, category)
    # commit_snapshots_dir_path = get_sharded_path(category_snapshots_dir_path, commit)
    # url_snapshots_dir_path = os.path.join(commit_snapshots_dir_path, f'{url_count}')

    # before_summary = None
    # after_summary = None

    # for variant in _category_variants[category]:
        # log(f'*** Processing {variant} page graph...')

        # page_graph_file_path = os.path.join(url_snapshots_dir_path, variant, 'page_graph.graphml')

        # if not os.path.exists(page_graph_file_path):
            # log(f'** Skipped ({variant} page graph file does not exist)')
            # return

        # log(f'**** Reading {variant} page graph...')

        # with open(page_graph_file_path, 'r', encoding='utf-8') as page_graph_file:
            # page_graph_src = page_graph_file.read()

        # try:
            # page_graph = networkx.graphml.parse_graphml(page_graph_src)
        # except Exception:
            # log(f'** Skipped (failed to parse {variant} page graph)')
            # return

        # log(f'**** Walking over {variant} page graph...')

        # page_graph_summary = _summarize_graph(page_graph)

        # if before_summary is None:
            # before_summary = page_graph_summary
        # else:
            # after_summary = page_graph_summary

    # for item in after_summary:
        # if item in before_summary:
            # before_summary.remove(item)

    # summary = before_summary

    summary = subgraph_summary

    log('*** Saving result...')

    os.makedirs(commit_summaries_dir_path, exist_ok=True)

    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2)

def run():
    print('* Indexing subgraphs...')

    subgraphs = []

    with os.scandir(subgraphs_dir_path) as category_subgraphs_dirs:
        for category_subgraphs_dir in category_subgraphs_dirs:
            with os.scandir(category_subgraphs_dir) as shard_subgraphs_dirs:
                for shard_subgraphs_dir in shard_subgraphs_dirs:
                    with os.scandir(shard_subgraphs_dir) as commit_subgraphs_dirs:
                        for commit_subgraphs_dir in commit_subgraphs_dirs:
                            with os.scandir(commit_subgraphs_dir) as url_subgraph_files:
                                for url_subgraph_file in url_subgraph_files:
                                    url_count = int(os.path.splitext(url_subgraph_file.name)[0])
                                    subgraphs.append((len(subgraphs) + 1, category_subgraphs_dir.name, commit_subgraphs_dir.name, url_count))

    print(f'* Generating summaries for {len(subgraphs)} subgraph(s)...')

    pad_length = len(f'{len(subgraphs)}')

    with multiprocessing.Pool(_concurrency) as pool:
        pool.starmap(functools.partial(_generate, pad_length), subgraphs, chunksize=_chunk_size)

if __name__ == '__main__':
    run()
