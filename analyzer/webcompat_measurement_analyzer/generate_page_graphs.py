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

import collections
import functools
import json
import lxml
import multiprocessing
import networkx
from networkx import MultiDiGraph
import os
import sys
import traceback

from ._lib.paths import get_sharded_path, page_graphs_dir_path, snapshots_dir_path

_concurrency = 8
_chunk_size = 128

_Category = collections.namedtuple('_Categories', ['id', 'variant'])
_categories = [_Category('broken', 'child'), _Category('unbroken', 'parent')]

def _generate(pad_length, snapshot_count, category, commit, url_count, variant):
    def log(*args, **kwargs):
        print(f'[{snapshot_count: >{pad_length}}]', *args, **kwargs)

    log(f'** Post-processing {category} page graph from {commit} URL #{url_count} {variant} snapshot (#{snapshot_count})...')

    category_page_graphs_dir_path = os.path.join(page_graphs_dir_path, category)
    commit_page_graphs_dir_path = get_sharded_path(category_page_graphs_dir_path, commit)
    url_page_graphs_dir_path = os.path.join(commit_page_graphs_dir_path, f'{url_count}')
    out_page_graph_file_path = os.path.join(url_page_graphs_dir_path, f'{variant}.graphml')

    if os.path.exists(out_page_graph_file_path):
        log('*** Skipped (page graph file already exists)')
        return

    category_snapshots_dir_path = os.path.join(snapshots_dir_path, category)
    commit_snapshots_dir_path = get_sharded_path(category_snapshots_dir_path, commit)
    snapshot_dir_path = os.path.join(commit_snapshots_dir_path, f'{url_count}', variant)

    in_page_graph_file_path = os.path.join(snapshot_dir_path, 'page_graph.graphml')

    if not os.path.exists(in_page_graph_file_path):
        log('*** Skipped (snapshot does not have a page graph)')
        return

    log('*** Reading page graph from snapshot...')

    with open(in_page_graph_file_path, 'r', encoding='utf-8') as in_page_graph_file:
        in_page_graph_src = in_page_graph_file.read()

    try:
        page_graph = MultiDiGraph(networkx.graphml.parse_graphml(in_page_graph_src))
    except Exception:
        log(traceback.format_exc(), file=sys.stderr)
        log('*** Skipped (page graph parse failure)')
        return

    console_log_file_path = os.path.join(snapshot_dir_path, 'console.json')

    if os.path.exists(console_log_file_path):
        log('*** Reading console log from snapshot...')

        with open(console_log_file_path, 'r', encoding='utf-8') as console_log_file:
            console_log_src = console_log_file.read()

        try:
            console_log = json.loads(console_log_src)
        except Exception:
            log(traceback.format_exc(), file=sys.stderr)
            log('*** Substituting empty console log (parse failure)')
            console_log = []
    else:
        log('*** Substituting empty console log (snapshot does not have a console log)')
        console_log = []

    log('*** Enriching page graph with console log data...')

    next_graph_item_id = 1 + page_graph.number_of_nodes() + page_graph.number_of_edges()

    console_node_id = next_graph_item_id
    next_graph_item_id += 1
    page_graph.add_nodes_from([(
        f'n{console_node_id}',
        {'id': console_node_id, 'node type': 'console'}
    )])

    for console_message in console_log:
        script_id = console_message['scriptId']
        if script_id is None:
            if 'stackTrace' in console_message:
                stack_trace = console_message['stackTrace']
                if 'callFrames' in stack_trace:
                    call_frames = stack_trace['callFrames']
                    if len(call_frames) > 0:
                        call_frame = call_frames[0]
                        if 'scriptId' in call_frame:
                            script_id = call_frame['scriptId']
        if script_id is not None:
            script_id = int(script_id)
            script_node_id = None
            for node_id, node_data in page_graph.nodes(data=True):
                if node_data['node type'] == 'script' and node_data['script id'] == script_id:
                    script_node_id = node_id
                    break
            if script_node_id is not None:
                console_message_edge_id = next_graph_item_id
                next_graph_item_id += 1
                page_graph.add_edges_from([(
                    script_node_id,
                    f'n{console_node_id}',
                    f'e{console_message_edge_id}',
                    {
                        'id': console_message_edge_id,
                        'edge type': 'console message',
                        'source': console_message['source'],
                        'level': console_message['level'],
                        'message text': console_message['messageText'],
                        'type': console_message['type']
                    }
                )])

    log('*** Generating GraphML...')

    out_page_graph_src = '\n'.join(networkx.graphml.generate_graphml(page_graph))

    log('*** Pretty-printing GraphML...')

    try:
        out_page_graph_src = lxml.etree.tostring(
            lxml.etree.fromstring(out_page_graph_src), encoding='unicode', pretty_print=True
        )
    except Exception:
        log(traceback.format_exc(), file=sys.stderr)
        log('**** Pretty-printing failed')

    log('*** Saving result...')

    os.makedirs(url_page_graphs_dir_path, exist_ok=True)

    with open(out_page_graph_file_path, 'w', encoding='utf-8') as out_page_graph_file:
        out_page_graph_file.write(out_page_graph_src)

def run():
    print(f'* Indexing snapshots in {len(_categories)} categor(y/ies)...')

    snapshots = []

    for category_index, category in enumerate(_categories):
        print(f'** Indexing snapshots in category {category.id} (#{category_index + 1})...')

        category_snapshots_dir_path = os.path.join(snapshots_dir_path, category.id)
        with os.scandir(category_snapshots_dir_path) as shard_snapshots_dirs:
            for shard_snapshots_dir in shard_snapshots_dirs:
                with os.scandir(shard_snapshots_dir) as commit_snapshots_dirs:
                    for commit_snapshots_dir in commit_snapshots_dirs:
                        with os.scandir(commit_snapshots_dir) as url_snapshots_dirs:
                            for url_snapshots_dir in url_snapshots_dirs:
                                snapshots.append((len(snapshots) + 1, category.id, commit_snapshots_dir.name, int(url_snapshots_dir.name), category.variant))

    print(f'* Post-processing page graphs from {len(snapshots)} snapshot(s)...')

    pad_length = len(f'{len(snapshots)}')

    with multiprocessing.Pool(_concurrency) as pool:
        pool.starmap(functools.partial(_generate, pad_length), snapshots, chunksize=_chunk_size)

if __name__ == '__main__':
    run()
