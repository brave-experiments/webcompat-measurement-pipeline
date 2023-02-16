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
import gzip
import json
import lxml.etree
import os
import re
import sys
import traceback

import multiprocessing
import networkx

from ._lib.extract import extract_interesting_subgraph
from ._lib.paths import get_sharded_path, page_graphs_dir_path, snapshots_dir_path, subgraphs_dir_path

_concurrency = 8
_chunk_size = 128

_Category = collections.namedtuple('_Categories', ['id', 'variant'])
_categories = [_Category('broken', 'child'), _Category('unbroken', 'parent')]

def _generate(pad_length, snapshot_count, category, commit, url_count, variant):
    def log(*args, **kwargs):
        print(f'[{snapshot_count: >{pad_length}}]', *args, **kwargs)

    log(f'** Extracting {category} subgraph from {commit} URL #{url_count} {variant} snapshot (#{snapshot_count})...')

    category_subgraphs_dir_path = os.path.join(subgraphs_dir_path, category)
    commit_subgraphs_dir_path = get_sharded_path(category_subgraphs_dir_path, commit)
    subgraph_file_path = os.path.join(commit_subgraphs_dir_path, f'{url_count}.graphml')

    if os.path.exists(subgraph_file_path):
        log('*** Skipped (subgraph file already exists)')
        return

    category_page_graphs_dir_path = os.path.join(page_graphs_dir_path, category)
    commit_page_graphs_dir_path = get_sharded_path(category_page_graphs_dir_path, commit)
    url_page_graphs_dir_path = os.path.join(commit_page_graphs_dir_path, f'{url_count}')
    page_graph_file_path = os.path.join(url_page_graphs_dir_path, f'{variant}.graphml')

    if not os.path.exists(page_graph_file_path):
        log('*** Skipped (no page graph)')
        return

    log('*** Reading page graph...')

    with open(page_graph_file_path, 'r', encoding='utf-8') as page_graph_file:
        page_graph_src = page_graph_file.read()

    try:
        page_graph = networkx.graphml.parse_graphml(page_graph_src)
    except Exception:
        log(traceback.format_exc(), file=sys.stderr)
        log('*** Skipped (parse failure)')
        return

    category_snapshots_dir_path = os.path.join(snapshots_dir_path, category)
    commit_snapshots_dir_path = get_sharded_path(category_snapshots_dir_path, commit)
    snapshot_dir_path = os.path.join(commit_snapshots_dir_path, f'{url_count}', variant)
    dom_dump_file_path = os.path.join(snapshot_dir_path, 'dump.json.gz')

    if not os.path.exists(dom_dump_file_path):
        log('*** No DOM dump available')

        dom_dump = None
    else:
        log('*** Reading DOM dump...')

        try:
            with gzip.open(dom_dump_file_path, 'rt', encoding='utf-8') as dom_dump_file:
                dom_dump = json.load(dom_dump_file)
        except Exception:
            log(traceback.format_exc(), file=sys.stderr)
            log('**** DOM dump read failed')
            dom_dump = None

    log('*** Pulling out the interesting part of the page graph...')

    subgraph = extract_interesting_subgraph(page_graph, dom_dump)
    if subgraph.number_of_nodes() < 1:
        log('*** Skipped (empty subgraph)')
        return

    log('*** Generating GraphML...')

    subgraph_graphml = '\n'.join(networkx.graphml.generate_graphml(subgraph))

    log('*** Pretty-printing GraphML...')

    subgraph_pretty_graphml = lxml.etree.tostring(
        lxml.etree.fromstring(subgraph_graphml), encoding='unicode', pretty_print=True
    )

    log('*** Saving result...')

    os.makedirs(commit_subgraphs_dir_path, exist_ok=True)

    with open(subgraph_file_path, 'w', encoding='utf-8') as subgraph_file:
        subgraph_file.write(subgraph_pretty_graphml)

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

    print(f'* Extracting subgraphs from {len(snapshots)} snapshot(s)...')

    pad_length = len(f'{len(snapshots)}')

    with multiprocessing.Pool(_concurrency) as pool:
        pool.starmap(functools.partial(_generate, pad_length), snapshots, chunksize=_chunk_size)

if __name__ == '__main__':
    run()
