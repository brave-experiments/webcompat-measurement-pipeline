import os
import pandas as pd
import numpy as np
import json
import networkx as nx
from tqdm import tqdm
from src.config import *
from src.features import *
from src.util import *


def __get_graph_ids(sub_directory):
    for file_name in os.listdir(os.path.join(BASE_PATH, PG_DATA_PATH, sub_directory)):
        if not file_name.startswith('.'):
            yield file_name


def __get_graph_path(graph_id_with_label, graph_type):
    assert(isinstance(graph_type, GraphType))

    graph_id, label = split_graph_id_and_label(graph_id_with_label)
    if graph_type == GraphType.DELTA:
        path = DELTA_PATH
    elif graph_type == GraphType.PRE_INTERVENTION:
        path = PRE_INTERVENTION_PATH
    elif graph_type == GraphType.POST_INTERVENTION:
        path = POST_INTERVENTION_PATH
    else:
        return None

    return os.path.join(BASE_PATH, PG_DATA_PATH, __get_subdirectory(label), graph_id, path)


def __get_manifest_path(graph_id_with_label):
    graph_id, label = split_graph_id_and_label(graph_id_with_label)

    return os.path.join(BASE_PATH, PG_DATA_PATH, __get_subdirectory(label), graph_id, MANIFEST_PATH)


def __get_subdirectory(label):
    return BROKEN_DIR if bool(int(label)) else UNBROKEN_DIR


def __get_graph_ids_with_labels():
    negative_class = [id + NEGATIVE_CLASS_SUFFIX for id  in __get_graph_ids(UNBROKEN_DIR)]
    positive_class = [id + POSITIVE_CLASS_SUFFIX for id  in __get_graph_ids(BROKEN_DIR)]
            
    return negative_class + positive_class


def __load_graph(graph_id_with_label, graph_type, verbose=False):
    assert(isinstance(graph_type, GraphType))

    path = __get_graph_path(graph_id_with_label, graph_type)
    
    try:
        G = nx.read_graphml(path)
    except Exception as e:
        if verbose:
            print(e)
        return None
        
    return G


def __pad_missing_cols(df, col_set):
    missing_cols = list(col_set - set(df.columns.to_list()))
    return df.reindex(df.columns.tolist() + missing_cols, axis=1)


def __to_df(G):
    edge_prefix = 'edge_'
    source_prefix = 'source_'
    target_prefix = 'target_'

    edges_df = nx.to_pandas_edgelist(G)
    edge_cols = edges_df.columns
    edges_df.columns = [edge_prefix + col.lower().replace(' ', '_') for col in edge_cols]

    nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    source_df = nodes_df
    target_df = nodes_df.copy()

    source_cols = source_df.columns
    source_df.columns = [source_prefix + col.lower().replace(' ', '_') for col in source_cols]

    target_cols = target_df.columns
    target_df.columns = [target_prefix + col.lower().replace(' ', '_') for col in target_cols]

    graph_df = edges_df.join(source_df, on='edge_source', how='left') \
                       .join(target_df, on='edge_target', how='left')

    padded_df = __pad_missing_cols(graph_df, set(GRAPH_DF_COLUMNS.keys()))

    str_cols = [col for col, dtype in GRAPH_DF_COLUMNS.items() if dtype == str]
    padded_df[str_cols] = padded_df[str_cols].fillna('')

    return padded_df


def generate_graph_dfs_and_metadata():
    graphs = []

    for graph_id_with_label in tqdm(__get_graph_ids_with_labels()):
        G_pre = __load_graph(graph_id_with_label, GraphType.PRE_INTERVENTION)
        if G_pre:
            G_pre_df = __to_df(G_pre)
            G_pre_df.to_csv(os.path.join(BASE_PATH, GRAPH_DF_DIR, graph_id_with_label + '_pre.csv'), index=False)
    
        G_post = __load_graph(graph_id_with_label, GraphType.POST_INTERVENTION)
        if G_post:
            G_post_df = __to_df(G_post)
            G_post_df.to_csv(os.path.join(BASE_PATH, GRAPH_DF_DIR, graph_id_with_label + '_post.csv'), index=False)

        G_delta = __load_graph(graph_id_with_label, GraphType.DELTA)
        if G_delta:
            G_delta_df = __to_df(G_delta)
            G_delta_df.to_csv(os.path.join(BASE_PATH, GRAPH_DF_DIR, graph_id_with_label + '_delta.csv'), index=False)
        
        graph_info = {
            'graph_id_with_label': graph_id_with_label,
            'G_pre_nodes': G_pre.number_of_nodes() if G_pre else np.nan,
            'G_pre_edges': G_pre.number_of_edges() if G_pre else np.nan,
            'G_post_nodes': G_post.number_of_nodes() if G_post else np.nan,
            'G_post_edges': G_post.number_of_edges() if G_post else np.nan,
            'G_delta_nodes': G_delta.number_of_nodes() if G_delta else np.nan,
            'G_delta_edges': G_delta.number_of_edges() if G_delta else np.nan,
        }

        graphs.append(graph_info)

    meta_df = pd.DataFrame(graphs)
    meta_df[['id', 'label']] = meta_df['graph_id_with_label'].str.split('-', expand=True)
    meta_df = meta_df.set_index('graph_id_with_label').sort_index()
    meta_df.to_csv(os.path.join(BASE_PATH, GRAPH_METADATA_PATH))


def get_graph_metadata(filter_nas=False, iqr=False, uq=False, uc=1):
    meta_df = pd.read_csv(os.path.join(BASE_PATH, GRAPH_METADATA_PATH), index_col='graph_id_with_label', converters={'G_pre_cols': pd.eval})

    if filter_nas:
        meta_df = meta_df[meta_df[['G_pre_nodes', 'G_delta_nodes']].isna().sum(axis=1) == 0]

    if not (iqr or uq):
        return meta_df

    if iqr:
        assert(0 < iqr < 1)
        eps = (1-iqr)/2
        lower_quantile, upper_quantil = 0+eps, 1-eps

        lb_nodes = meta_df['G_pre_nodes'].quantile(lower_quantile)
        ub_nodes = meta_df['G_pre_nodes'].quantile(upper_quantil)
        lb_edges = meta_df['G_pre_edges'].quantile(lower_quantile)
        ub_edges = meta_df['G_pre_edges'].quantile(upper_quantil)

        predicate = """
            G_pre_nodes >= %i and \
            G_pre_nodes <=%i and \
            G_pre_edges >= %i and \
            G_pre_edges <=%i
        """ % (lb_edges, ub_edges, lb_nodes, ub_nodes)
        return meta_df.query(predicate)

    if uq:
        assert(0 < uq < 1)
        meta_df['edge_node_ratio'] = meta_df['G_pre_edges'] / meta_df['G_pre_nodes']
        ub_edge_node_ratio = meta_df['edge_node_ratio'].quantile(uq)

        assert(0 < uc <= 1)
        ub_node_count = meta_df['G_pre_nodes'].quantile(uc)
        ub_edge_count = meta_df['G_pre_edges'].quantile(uc)

        predicate = """
            edge_node_ratio <= %i and \
            G_pre_nodes <= %i and \
            G_pre_edges <= %i
        """ % (ub_edge_node_ratio, ub_node_count, ub_edge_count)
        return meta_df.query(predicate)


def load_df(graph_id_with_label, graph_type, verbose=False):
    assert(isinstance(graph_type, GraphType))

    path = os.path.join(BASE_PATH, GRAPH_DF_DIR, graph_id_with_label)
    if graph_type == GraphType.DELTA:
        path += '_delta.csv'
    elif graph_type == GraphType.PRE_INTERVENTION:
        path += '_pre.csv'
    elif graph_type == GraphType.POST_INTERVENTION:
        path += '_post.csv'
    else:
        return None

    try:
        df = pd.read_csv(path, dtype=GRAPH_DF_COLUMNS, keep_default_na=False)
    except Exception as e:
        if verbose:
            print(e)
        return None
        
    return df


def load_manifest(graph_id_with_label):
    with open(__get_manifest_path(graph_id_with_label)) as f:
        json_data = f.read()
        data = json.loads(json_data)

        return data


def create_training_data(meta_df, from_file=False):
    if from_file:
        return pd.read_csv(TRAINING_DATA_PATH, index_col=ID_NAME)

    features = []
    for graph_id in tqdm(meta_df.index.values):
        df_delta = load_df(graph_id, GraphType.DELTA)
        df_pre = load_df(graph_id, GraphType.PRE_INTERVENTION)
        manifest = load_manifest(graph_id)
        extracted_features = extract_features(graph_id, df_pre, df_delta, manifest)
        features.append(extracted_features)

    df = pd.DataFrame(features).fillna(0).set_index(ID_NAME)
    df = pd.DataFrame(features).set_index(ID_NAME)
    df.to_csv(TRAINING_DATA_PATH,  index=True)
    return df
