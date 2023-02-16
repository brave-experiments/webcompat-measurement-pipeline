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
import itertools

import networkx
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_interesting_subgraph(page_graph, dom_dump):
    resource_nodes = _find_resource_nodes(page_graph)
    affected_resource_nodes = [
        (resource_node_id, resource_node_data)
        for resource_node_id, resource_node_data in resource_nodes
        if is_uniquely_blocked_by_alt(page_graph, resource_node_id)
    ]

    interesting_node_ids = list(
        _enrich_resource_nodes(page_graph, dom_dump, affected_resource_nodes)
    )

    interesting_edges = [
        interesting_edge
        for interesting_node_id in interesting_node_ids
        for interesting_edge in _find_non_structural_edges(page_graph, interesting_node_id)
    ]

    interesting_edges.extend(
        _crawl_up_edges_from_nodes(page_graph, _generate_node_id_set_from_edges(interesting_edges))
    )

    # interesting_edges.extend(
    #    _find_structure_edges_between_nodes(
    #        page_graph, _generate_node_id_set_from_edges(interesting_edges)
    #    )
    # )

    interesting_node_ids = _generate_node_id_set_from_edges(interesting_edges)
    interesting_edge_ids = _generate_edge_id_set_from_edges(interesting_edges)

    return networkx.graphviews.subgraph_view(
        page_graph,
        lambda node_id: node_id in interesting_node_ids,
        lambda out_node_id, in_node_id, edge_id: edge_id in interesting_edge_ids,
    )

def _find_resource_nodes(page_graph):
    for node_id, node_data in page_graph.nodes(data=True):
        if node_data['node type'] == 'resource':
            yield (node_id, node_data)

def is_uniquely_blocked_by_alt(page_graph, resource_node_id):
    resource_node_in_edges = page_graph.in_edges(nbunch=resource_node_id, data=True)

    has_alt_resource_block_edge = False

    for (out_node_id, in_node_id, edge_data) in resource_node_in_edges:
        if edge_data['edge type'] == 'resource block':
            if edge_data['is alt']:
                has_alt_resource_block_edge = True
            else:
                return False

    return has_alt_resource_block_edge

# _Stylesheet = collections.namedtuple('_Stylesheet', ['element_node_ids', 'src', 'line_indices'])

_excluded_request_starting_node_types = {'parser', 'DOM root'}
_dom_node_types = {'DOM root', 'frame owner', 'HTML element', 'text node'}
def _enrich_resource_nodes(page_graph, dom_dump, resource_nodes):
    # stylesheets = []

    for resource_node_id, resource_node_data in resource_nodes:
        yield resource_node_id

        # stylesheet_element_node_ids = set()

        for request_starting_node_id, request_start_edge_data in _find_request_start_edges(
            page_graph, resource_node_id
        ):
            request_starting_node_data = page_graph.nodes[request_starting_node_id]

            if (
                request_starting_node_data['node type']
                not in _excluded_request_starting_node_types
            ):
                yield request_starting_node_id

                for extra_node_id in _follow_request_starting_node_edges(
                    page_graph, request_starting_node_id
                ):
                    yield extra_node_id

                # if dom_dump is not None and request_start_edge_data['request type'] == 'CSS':
                    # stylesheet_element_node_ids.add(request_starting_node_id)
                    # stylesheet_elements[request_starting_node_data['node id']] = request_starting_node_id

        # if len(stylesheet_element_node_ids) > 0:
            # try:
                # stylesheet_src = requests.get(resource_node_data['url'], verify=False).text
            # except:
                # pass
            # else:
                # stylesheet_line_indices = [
                    # 0,
                    # *(i + 1 for i, c in enumerate(stylesheet_src) if c == '\n'),
                # ]
                # stylesheets.append(
                    # _Stylesheet(
                        # stylesheet_element_node_ids, stylesheet_src, stylesheet_line_indices
                    # )
                # )

    if dom_dump is not None:
        dom_dump_root = dom_dump['root']
        dom_dump_stylesheet_links = dom_dump['styleSheetLinks']

        # stylesheet_id_to_matched_selectors = {}
        dom_node_id_to_matched_stylesheet_ids = {}

        dom_node_queue = collections.deque([dom_dump_root])

        while True:
            try:
                dom_node = dom_node_queue.popleft()
            except IndexError:
                break

            dom_node_id = dom_node['backendNodeId']

            if dom_node_id in dom_node_id_to_matched_stylesheet_ids:
                dom_node_matched_stylesheet_ids = dom_node_id_to_matched_stylesheet_ids[
                    dom_node_id
                ]
            else:
                dom_node_matched_stylesheet_ids = set()
                dom_node_id_to_matched_stylesheet_ids[
                    dom_node_id
                ] = dom_node_matched_stylesheet_ids

            matches = itertools.chain()
            if 'matchedCSSRules' in dom_node:
                matches = itertools.chain(matches, dom_node['matchedCSSRules'])
            if 'pseudoElementStyleMatches' in dom_node:
                matches = itertools.chain(
                    matches,
                    itertools.chain.from_iterable(
                        item['matches'] for item in dom_node['pseudoElementStyleMatches']
                    ),
                )

            for match in matches:
                if match['rule']['origin'] != 'regular':
                    continue

                stylesheet_id = match['rule']['styleSheetId']

                # if stylesheet_id in stylesheet_id_to_matched_selectors:
                    # stylesheet_matched_selectors = stylesheet_id_to_matched_selectors[
                        # stylesheet_id
                    # ]
                # else:
                    # stylesheet_matched_selectors = []
                    # stylesheet_id_to_matched_selectors[
                        # stylesheet_id
                    # ] = stylesheet_matched_selectors

                # stylesheet_matched_selectors.extend(match['rule']['selectorList']['selectors'])

                dom_node_matched_stylesheet_ids.add(stylesheet_id)

            if 'children' in dom_node:
                dom_node_queue.extend(dom_node['children'])

        # stylesheet_id_to_element_node_ids = {}

        # for stylesheet in stylesheets:
            # for (
                # stylesheet_id,
                # stylesheet_matched_selectors,
            # ) in stylesheet_id_to_matched_selectors.items():
                # if stylesheet_id in stylesheet_id_to_element_node_ids:
                    # continue

                # is_matching_stylesheet = False

                # for matched_selector in stylesheet_matched_selectors:
                    # if 'text' not in matched_selector or 'range' not in matched_selector:
                        # continue

                    # text = matched_selector['text']
                    # if len(text) < 1:
                        # continue

                    # start_line = matched_selector['range']['startLine']
                    # start_column = matched_selector['range']['startColumn']

                    # if start_line >= len(stylesheet.line_indices):
                        # is_matching_stylesheet = False
                        # break

                    # start_index = stylesheet.line_indices[start_line] + start_column
                    # end_index = start_index + len(text)

                    # if stylesheet.src[start_index:end_index] != text:
                        # is_matching_stylesheet = False
                        # break

                    # is_matching_stylesheet = True

                # if is_matching_stylesheet:
                    # stylesheet_id_to_element_node_ids[stylesheet_id] = stylesheet.element_node_ids
                    # break

        dom_node_id_to_node_id = {}

        for node_id, node_data in page_graph.nodes(data=True):
            if node_data['node type'] not in _dom_node_types or 'node id' not in node_data:
                continue

            dom_node_id_to_node_id[node_data['node id']] = node_id

        stylesheet_id_to_dom_node_id = {}

        for stylesheet_link in dom_dump_stylesheet_links:
            stylesheet_id_to_dom_node_id[stylesheet_link['styleSheetId']] = stylesheet_link['nodeId']

        next_edge_id = 1 + page_graph.number_of_nodes() + page_graph.number_of_edges()

        for node_id, node_data in page_graph.nodes(data=True):
            if node_data['node type'] not in _dom_node_types or 'node id' not in node_data:
                continue

            dom_node_id = node_data['node id']

            if dom_node_id not in dom_node_id_to_matched_stylesheet_ids:
                continue
            dom_node_matched_stylesheet_ids = dom_node_id_to_matched_stylesheet_ids[dom_node_id]

            for stylesheet_id in dom_node_matched_stylesheet_ids:
                # if stylesheet_id not in stylesheet_id_to_element_node_ids:
                if stylesheet_id not in stylesheet_id_to_dom_node_id:
                    continue
                # stylesheet_element_node_ids = stylesheet_id_to_element_node_ids[stylesheet_id]
                stylesheet_dom_node_id = stylesheet_id_to_dom_node_id[stylesheet_id]

                if stylesheet_dom_node_id not in dom_node_id_to_node_id:
                    continue
                stylesheet_node_id = dom_node_id_to_node_id[stylesheet_dom_node_id]

                # for stylesheet_element_node_id in stylesheet_element_node_ids:
                edge_id = next_edge_id
                next_edge_id += 1

                new_edge = (
                    # stylesheet_element_node_id,
                    stylesheet_node_id,
                    node_id,
                    f'e{edge_id}',
                    {'id': edge_id, 'edge type': 'style rule match'},
                )
                page_graph.add_edges_from([new_edge])

def _find_request_start_edges(page_graph, node_id):
    in_edges = page_graph.in_edges(nbunch=node_id, data=True)
    for out_node_id, in_node_id, edge_data in in_edges:
        if edge_data['edge type'] == 'request start':
            yield (out_node_id, edge_data)

_followable_request_starting_node_edges = {'execute'}
def _follow_request_starting_node_edges(page_graph, node_id):
    for out_node_id, in_node_id, edge_data in page_graph.out_edges(nbunch=node_id, data=True):
        if edge_data['edge type'] in _followable_request_starting_node_edges:
            yield in_node_id

    for out_node_id, in_node_id, edge_data in page_graph.in_edges(nbunch=node_id, data=True):
        if edge_data['edge type'] in _followable_request_starting_node_edges:
            yield out_node_id

def _find_non_structural_edges(page_graph, node_id):
    edges = itertools.chain(
        page_graph.out_edges(nbunch=node_id, data=True),
        page_graph.in_edges(nbunch=node_id, data=True),
    )
    for out_node_id, in_node_id, edge_data in edges:
        if edge_data['edge type'] == 'structure':
            continue
        yield (out_node_id, in_node_id, edge_data)

_crawlable_edges = {'filter', 'shield', 'storage bucket'}  # , 'structure'}
def _crawl_up_edges_from_nodes(page_graph, source_node_ids):
    node_id_queue = collections.deque(source_node_ids)
    queued_node_ids = set(node_id_queue)

    while True:
        try:
            node_id = node_id_queue.popleft()
        except IndexError:
            break

        for out_node_id, in_node_id, edge_data in page_graph.in_edges(nbunch=node_id, data=True):
            if edge_data['edge type'] not in _crawlable_edges:
                continue

            yield (out_node_id, in_node_id, edge_data)

            if out_node_id not in queued_node_ids:
                node_id_queue.append(out_node_id)
                queued_node_ids.add(out_node_id)

def _find_structure_edges_between_nodes(page_id, node_id_set):
    for node_id in node_id_set:
        for out_node_id, in_node_id, edge_data in page_graph.out_edges(nbunch=node_id, data=True):
            if edge_data['edge type'] == 'structure' and in_node_id in node_id_set:
                yield (out_node_id, in_node_id, edge_data)

def _generate_node_id_set_from_edges(edges):
    return set(
        node_id
        for out_node_id, in_node_id, edge_data in edges
        for node_id in [out_node_id, in_node_id]
    )

def _generate_edge_id_set_from_edges(edges):
    return set(f'e{edge_data["id"]}' for out_node_id, in_node_id, edge_data in edges)
