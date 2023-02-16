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
import random

import networkx
from networkx import DiGraph

from ._lib.extract import is_uniquely_blocked_by_alt
from ._lib.paths import (
    differents_file_path,
    get_sharded_path,
    metrics_dir_path,
    snapshots_dir_path,
    subgraphs_dir_path,
)

_concurrency = 8
_chunk_size = 128

_category_variants = {
    'broken': ('child', 'parent'),
    'unbroken': ('parent', 'child')
}

_html_node_types = {'DOM root', 'frame owner', 'HTML element', 'text node'}

_media_tag_names = {'audio', 'img', 'picture', 'track', 'video'}

_storage_buckets = {'cookie jar', 'local storage', 'session storage'}

_storage_categories = {
    'read storage call': 'read',
    'storage set': 'write',
    'delete storage': 'write',
    'clear storage': 'write',
}

_web_api_categories = {
    'NavigatorLanguage.language': 'navigator',
    'NavigatorLanguage.languages': 'navigator',
    'NavigatorPlugins.plugins': 'navigator',
    'Navigator.doNotTrack': 'navigator',
    'NavigatorID.platform': 'navigator',
    'Screen.height': 'screen',
    'Screen.width': 'screen',
    'Screen.colorDepth': 'screen',
    'Screen.pixelDepth': 'screen',
    'Screen.availLeft': 'screen',
    'Screen.availTop': 'screen',
    'Screen.availHeight': 'screen',
    'Screen.availWidth': 'screen',
    'CanvasRenderingContext2D.measureText': 'canvas',
    'HTMLCanvasElement.getContext': 'canvas',
    'HTMLCanvasElement.toDataURL': 'canvas',
    'HTMLCanvasElement.toBlob': 'canvas',
    'WebGLRenderingContext.getParameter': 'WebGL',
    'WebGLRenderingContext.getExtension': 'WebGL',
    'WebGLRenderingContext.getShaderPrecisionFormat': 'WebGL',
    'WebGL2RenderingContext.getParameter': 'WebGL',
}

_console_message_sources = {
    'xml',
    'javascript',
    'network',
    'console-api',
    'storage',
    'appcache',
    'rendering',
    'security',
    'other',
    'deprecation',
    'worker'
}

_console_message_levels = {
    'log',
    'warning',
    'error',
    'debug',
    'info'
}

_request_types = {
    'AJAX',
    'Image',
    'ScriptClassic',
    'ScriptModule',
    'CSS',
    'Video',
    'Audio',
    'SVG',
    'Font',
    'Document',
    'Unknown'
}

def _compute_over_graph(graph):
    graph_metrics = {
        # 'resource block delta': 0,
        # 'katz centrality': 0,
        'number of nodes': graph.number_of_nodes(),
        'number of edges': graph.number_of_edges(),
        'number of event registrations': 0,
        'number of scripts': 0,
        'total number of DOM nodes created': 0,
        'net number of DOM nodes created': 0,
        'number of DOM nodes created and inserted by the same script': 0,
        'number of DOM nodes touched': 0,
        'number of text nodes created': 0,
        'number of media elements created': 0,
        'number of style rule matches': 0,
        'number of failed DOM queries': 0,
        'total number of network requests started': 0,
        'total number of network requests completed': 0,
        'total number of network requests with errors': 0,
        'total number of network requests blocked': 0,
        'total number of network requests uniquely blocked': 0,
        'total number of bytes transferred': 0,
        'total number of storage accesses': 0,
        'total number of web API calls': 0,
        'total number of console messages': 0
    }

    for storage_bucket in sorted(_storage_buckets):
        graph_metrics[f'total number of {storage_bucket} accesses'] = 0

        for storage_category in sorted(set(_storage_categories.values())):
            graph_metrics[f'number of {storage_bucket} {storage_category}s'] = 0

    for storage_category in sorted(set(_storage_categories.values())):
        graph_metrics[f'number of storage {storage_category}s'] = 0

    for web_api_category in sorted(set(_web_api_categories.values())):
        graph_metrics[f'number of {web_api_category} web API calls'] = 0

    for console_message_source in sorted(_console_message_sources):
        graph_metrics[f'number of {console_message_source} console messages'] = 0

    for console_message_level in sorted(_console_message_levels):
        graph_metrics[f'number of {console_message_level}-level console messages'] = 0

    for request_type in sorted(_request_types):
        graph_metrics[f'number of {request_type} network requests started'] = 0
        graph_metrics[f'number of {request_type} network requests completed'] = 0
        graph_metrics[f'number of {request_type} network requests with errors'] = 0
        graph_metrics[f'number of {request_type} network requests blocked'] = 0
        graph_metrics[f'number of {request_type} network requests uniquely blocked'] = 0
        graph_metrics[f'number of {request_type} bytes transferred'] = 0

    # graph_katz_centrality = networkx.algorithms.centrality.katz_centrality(
        # DiGraph(graph), 0.01
    # )

    # katz_centrality_sample_count = 0

    for node_id, node_data in graph.nodes(data=True):
        if node_data['node type'] == 'resource':
            is_uniquely_blocked = is_uniquely_blocked_by_alt(graph, node_id)

            for out_node_id, in_node_id, edge_data in graph.in_edges(
                nbunch=node_id, data=True
            ):
                if edge_data['edge type'] == 'request start':
                    graph_metrics['total number of network requests blocked'] += 1
                    graph_metrics[
                        f'number of {edge_data["request type"]} network requests blocked'
                    ] += 1

                    if is_uniquely_blocked:
                        graph_metrics['total number of network requests uniquely blocked'] += 1
                        graph_metrics[
                            f'number of {edge_data["request type"]} network requests uniquely blocked'
                        ] += 1

            # if is_uniquely_blocked_by_alt(graph, node_id):
                # graph_metrics['resource block delta'] += 1

                # representative_node_ids = []
                # if len(request_starting_node_ids) > 0:
                    # for request_starting_node_id in request_starting_node_ids:
                        # representative_node_id = request_starting_node_id

                        # for out_node_id, in_node_id, edge_data in graph.out_edges(
                            # nbunch=request_starting_node_id, data=True
                        # ):
                            # if edge_data['edge type'] == 'execute':
                                # representative_node_id = in_node_id
                                # break

                        # representative_node_ids.append(representative_node_id)
                # else:
                    # representative_node_ids = [node_id]

                # for representative_node_id in representative_node_ids:
                    # graph_metrics['katz centrality'] += graph_katz_centrality[representative_node_id]
                # katz_centrality_sample_count += len(representative_node_ids)
        elif node_data['node type'] == 'script':
            graph_metrics['number of scripts'] += 1

            dom_nodes_created = set()
            dom_nodes_inserted = set()

            for out_node_id, in_node_id, edge_data in graph.out_edges(
                data=True, nbunch=node_id
            ):
                if edge_data['edge type'] == 'create node':
                    dom_nodes_created.add(in_node_id)
                elif edge_data['edge type'] == 'insert node':
                    dom_nodes_inserted.add(in_node_id)

            graph_metrics['number of DOM nodes created and inserted by the same script'] += len(
                dom_nodes_created & dom_nodes_inserted
            )

    for out_node_id, in_node_id, edge_data in graph.edges(data=True):
        in_node_data = graph.nodes[in_node_id]

        if in_node_data['node type'] in _html_node_types:
            graph_metrics['number of DOM nodes touched'] += 1

        if edge_data['edge type'] == 'add event listener':
            graph_metrics['number of event registrations'] += 1
        elif edge_data['edge type'] == 'create node':
            graph_metrics['total number of DOM nodes created'] += 1

            if not in_node_data['is deleted']:
                graph_metrics['net number of DOM nodes created'] += 1

            if in_node_data['node type'] == 'HTML element':
                if 'tag name' in in_node_data and in_node_data['tag name'] in _media_tag_names:
                    graph_metrics['number of media elements created'] += 1
            elif in_node_data['node type'] == 'text node':
                graph_metrics['number of text nodes created'] += 1
        elif edge_data['edge type'] == 'request complete':
            bytes_transferred = int(edge_data['value'])

            graph_metrics['total number of network requests completed'] += 1
            graph_metrics['total number of bytes transferred'] += bytes_transferred

            for _, _, sibling_edge_data in graph.edges(nbunch=in_node_id, data=True):
                if sibling_edge_data['edge type'] == 'request start' and sibling_edge_data['request id'] == edge_data['request id']:
                    graph_metrics[f'number of {sibling_edge_data["request type"]} network requests completed'] += 1
                    graph_metrics[f'number of {sibling_edge_data["request type"]} bytes transferred'] += bytes_transferred
                    break
        elif edge_data['edge type'] == 'request error':
            bytes_transferred = int(edge_data['value'])

            graph_metrics['total number of network requests with errors'] += 1
            graph_metrics['total number of bytes transferred'] += bytes_transferred

            for _, _, sibling_edge_data in graph.edges(nbunch=in_node_id, data=True):
                if sibling_edge_data['edge type'] == 'request start' and sibling_edge_data['request id'] == edge_data['request id']:
                    graph_metrics[f'number of {sibling_edge_data["request type"]} network requests with errors'] += 1
                    graph_metrics[f'number of {sibling_edge_data["request type"]} bytes transferred'] += bytes_transferred
                    break
        elif edge_data['edge type'] == 'request start':
            graph_metrics['total number of network requests started'] += 1
            graph_metrics[f'number of {edge_data["request type"]} network requests started'] += 1
        elif edge_data['edge type'] == 'js call':
            if in_node_data['node type'] == 'web API':
                web_api_method = in_node_data['method']
                if web_api_method in _web_api_categories:
                    graph_metrics['total number of web API calls'] += 1

                    web_api_category = _web_api_categories[web_api_method]
                    graph_metrics[f'number of {web_api_category} web API calls'] += 1
        elif edge_data['edge type'] == 'style rule match':
            graph_metrics['number of style rule matches'] += 1
        elif edge_data['edge type'] == 'console message':
            graph_metrics['total number of console messages'] += 1
            graph_metrics[f'number of {edge_data["source"]} console messages'] += 1
            graph_metrics[f'number of {edge_data["level"]}-level console messages'] += 1
        elif edge_data['edge type'] in _storage_categories:
            graph_metrics['total number of storage accesses'] += 1

            storage_category = _storage_categories[edge_data['edge type']]
            graph_metrics[f'number of storage {storage_category}s'] += 1

            storage_bucket = in_node_data['node type']
            if storage_bucket in _storage_buckets:
                graph_metrics[f'total number of {storage_bucket} accesses'] += 1
                graph_metrics[f'number of {storage_bucket} {storage_category}s'] += 1
        elif edge_data['edge type'].startswith('empty ') and edge_data['edge type'].endswith(' result'):
            graph_metrics['number of failed DOM queries'] += 1

    # if katz_centrality_sample_count > 0:
        # graph_metrics['katz centrality'] /= katz_centrality_sample_count

    return graph_metrics

def _generate(pad_length, differents, subgraph_count, category, commit, url_count):
    def log(*args, **kwargs):
        print(f'[{subgraph_count: >{pad_length}}]', *args, **kwargs)

    log(f'** Generating metrics for {category} subgraph from {commit} URL #{url_count} (#{subgraph_count})...')

    metrics = {
        'is visually different': 1 if commit in differents else 0
    }

    category_metrics_dir_path = os.path.join(metrics_dir_path, category)
    commit_metrics_dir_path = get_sharded_path(category_metrics_dir_path, commit)
    metrics_file_path = os.path.join(commit_metrics_dir_path, f'{url_count}.json')
    if os.path.exists(metrics_file_path):
        log('*** Skipped (metrics file already exists)')
        return

    log('*** Processing subgraph...')

    log('**** Reading subgraph...')

    category_subgraphs_dir_path = os.path.join(subgraphs_dir_path, category)
    commit_subgraphs_dir_path = get_sharded_path(category_subgraphs_dir_path, commit)
    subgraph_file_path = os.path.join(commit_subgraphs_dir_path, f'{url_count}.graphml')

    with open(subgraph_file_path, 'r', encoding='utf-8') as subgraph_file:
        subgraph_src = subgraph_file.read()
    subgraph = networkx.graphml.parse_graphml(subgraph_src)

    log('**** Computing over subgraph...')

    subgraph_metrics = _compute_over_graph(subgraph)

    log('*** Processing page graphs...')

    category_snapshots_dir_path = os.path.join(snapshots_dir_path, category)
    commit_snapshots_dir_path = get_sharded_path(category_snapshots_dir_path, commit)
    url_snapshots_dir_path = os.path.join(commit_snapshots_dir_path, f'{url_count}')

    variant_metrics = []
    is_before_page_graph = True

    for variant in _category_variants[category]:
        log(f'**** Processing {variant} page graph...')

        page_graph_file_path = os.path.join(url_snapshots_dir_path, variant, 'page_graph.graphml')

        if not os.path.exists(page_graph_file_path):
            log(f'*** Skipped ({variant} page graph file does not exist)')
            return

        log(f'***** Reading {variant} page graph...')

        with open(page_graph_file_path, 'r', encoding='utf-8') as page_graph_file:
            page_graph_src = page_graph_file.read()

        try:
            page_graph = networkx.graphml.parse_graphml(page_graph_src)
        except Exception:
            log(f'*** Skipped (failed to parse {variant} page graph)')
            return

        log(f'***** Computing over {variant} page graph...')

        variant_metrics.append(_compute_over_graph(page_graph))

        if is_before_page_graph:
            is_before_page_graph = False
            metrics['cut size'] = networkx.algorithms.cuts.cut_size(page_graph, page_graph.nodes(), subgraph.nodes())

    log('*** Computing delta metrics...')

    delta_metrics = {}

    for key, value in variant_metrics[0].items():
        delta_metrics[key] = value - variant_metrics[1][key]

    log('*** Saving result...')

    for key, value in delta_metrics.items():
        metrics[f'delta {key}'] = value

    for key, value in subgraph_metrics.items():
        metrics[f'subgraph {key}'] = value

    os.makedirs(commit_metrics_dir_path, exist_ok=True)

    with open(metrics_file_path, 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

def run():
    print('* Reading differents list...')

    with open(differents_file_path, 'r', encoding='utf-8') as differents_file:
        differents = set(
            map(lambda line: line.split('/')[3],
                filter(lambda line: line,
                       map(lambda line: line.strip(),
                           differents_file.readlines()))))

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

    print(f'* Generating metrics for {len(subgraphs)} subgraph(s)...')

    pad_length = len(f'{len(subgraphs)}')

    with multiprocessing.Pool(_concurrency) as pool:
        pool.starmap(functools.partial(_generate, pad_length, differents), subgraphs,
            chunksize=_chunk_size)

if __name__ == '__main__':
    run()
