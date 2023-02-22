import pandas as pd
import numpy as np
from src.config import *
from src.util import split_graph_id_and_label


def get_no_of_edges_of_type_x(df):
    key_prefix = 'no_of_edges_of_type_'
    feature_group = df['edge_edge_type'].value_counts().to_dict()

    return {key_prefix + k.lower().replace(' ', '_'): v for (k, v) in feature_group.items()}


def get_no_of_source_nodes_of_type_x(df):
    key_prefix = 'no_of_source_nodes_of_type_'
    feature_group = df['source_node_type'].value_counts().to_dict()

    return {key_prefix + k.lower().replace(' ', '_'): v for (k, v) in feature_group.items()}


def get_no_of_target_nodes_of_type_x(df):
    key_prefix = 'no_of_target_nodes_of_type_'
    feature_group = df['target_node_type'].value_counts().to_dict()

    return {key_prefix + k.lower().replace(' ', '_'): v for (k, v) in feature_group.items()}


def get_no_of_unique_edge_types(df):
    key = 'no_of_unique_edge_types'
    value = df['edge_edge_type'].nunique()

    return {key: value}


def get_no_of_unique_source_node_types(df):
    key = 'no_of_unique_source_node_types'
    value = df['source_node_type'].nunique()

    return {key: value}


def get_no_of_unique_target_node_type(df):
    key = 'no_of_unique_target_node_type'
    value = df['target_node_type'].nunique()

    return {key: value}


def get_total_no_of_nodes(df):
    key = 'total_no_of_nodes'
    value = len(np.unique(df[['edge_source', 'edge_target']]))

    return {key: value}


def get_total_no_of_edges(df):
    key = 'total_no_of_edges'
    value = df.shape[0]

    return {key: value}


def get_no_of_text_nodes_created(df):
    key = 'no_of_text_nodes_created'
    predicate = """
        `edge_edge_type` == 'create node' and \
        `target_node_type` == 'text node'
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_html_elements_created(df):
    key = 'no_of_html_elements_created'
    predicate = """
        `edge_edge_type` == 'create node' and \
        `target_node_type` == 'HTML element' and \
        `target_tag_name` in ['video', 'img', 'picture', 'audio', 'track']
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_dom_nodes_created_by_scripts(df):
    key = 'no_of_dom_nodes_created_by_scripts'
    predicate = """
        `source_node_type` == 'script' and \
        `edge_edge_type` in ['insert node', 'create node'] and \
        `target_node_type` in ['HTML element', 'text node', 'frame owner', 'DOM root']
    """
    value = np.sum(df.query(predicate).groupby(['edge_target']).count()['edge_source'] == 2)

    return {key: value}


def get_no_of_canvas_web_api_calls(df):  # TODO: remove ispointinpath
    key = 'no_of_canvas_web_api_calls'
    predicate = """
        `target_node_type`.str.contains('web') and \
        `target_method`.str.contains('canvas') and \
        'CanvasRenderingContext2D.isPointInPath' not in `target_method`
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_navigator_web_api_calls(df):  # TODO: make simpler
    key = 'no_of_navigator_web_api_calls'
    predicate = """
        `source_node_type` == 'script' and \
        `target_node_type` == 'web API' and \
        `edge_edge_type` == 'js call' and \
        `target_method`.str.contains('Navigator') and \
        `target_method` not in ['Navigator.cookieEnabled', 'NavigatorID.userAgent', 'NavigatorID.platform']
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_screen_web_api_calls(df):
    key = 'no_of_screen_web_api_calls'
    predicate = """
        `target_node_type`.str.contains('web') and \
        `target_method`.str.contains('Screen')
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_webgl_web_api_calls(df):
    key = 'no_of_webgl_web_api_calls'
    predicate = """
        `target_node_type`.str.contains('web') and \
        `target_method`.str.contains('WebGL')
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_cookie_jar_read_storage_calls(df):
    key = 'no_of_cookie_jar_read_storage_calls'
    predicate = """
        `edge_edge_type` == 'read storage call' and \
        `target_node_type` == 'cookie jar'
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_cookie_jar_set_storage_calls(df):
    key = 'no_of_cookie_jar_set_storage_calls'
    predicate = """
        `edge_edge_type` == 'storage set' and \
        `target_node_type` == 'cookie jar'
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_session_storage_read_storage_calls(df):
    key = 'no_of_session_storage_read_storage_calls'
    predicate = """
        `edge_edge_type` == 'read storage call' and \
        `target_node_type` == 'session storage'
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_session_storage_delete_storage_calls(df):
    key = 'no_of_session_storage_delete_storage_calls'
    predicate = """
        `edge_edge_type` == 'delete storage' and \
        `target_node_type` == 'session storage'
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_local_storage_read_storage_calls(df):
    key = 'no_of_local_storage_read_storage_calls'
    predicate = """
        `edge_edge_type` == 'read storage call' and \
        `target_node_type` == 'local storage'
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_no_of_local_storage_delete_storage_calls(df):
    key = 'no_of_local_storage_delete_storage_calls'
    predicate = """
        `edge_edge_type` == 'delete storage' and \
        `target_node_type` == 'local storage'
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def get_sum_of_bytes_transferred_by_requests(df):
    key = 'sum_of_bytes_transferred_by_requests'
    predicate = """
        `edge_edge_type`.str.contains('request')
    """
    value = np.sum(df.query(predicate)['edge_value'].replace('', 0).astype(int))

    return {key: value}


def get_no_of_resources_blocked(df):
    key = 'no_of_resources_blocked'
    predicate = """
        `edge_edge_type`.str.contains('resource block')
    """
    value = df.query(predicate)['edge_target'].nunique()

    return {key: value}


def get_no_of_dom_related_target_nodes(df):
    key = 'no_of_dom_related_target_nodes'    
    predicate = """
        `target_node_type` in ['HTML element', 'text node', 'frame owner', 'DOM root']
    """
    value = df.query(predicate).shape[0]

    return {key: value}


def id(graph_id):
    key = ID_NAME

    return {key: graph_id}


def label(graph_id):
    key = LABEL_NAME
    _, label = split_graph_id_and_label(graph_id)

    return {key: int(label)}


def __extract_auto_and_expert_features_from_graph_df_for_scope(df, manifest, local_scope=False):
    if local_scope:
        mask = np.column_stack([df[col].str.contains(manifest['validation']['url'], na=False) for col in df])
        df = df.loc[mask.any(axis=1)]

    return {
        # Dimension: auto/expert
        **{'auto_' + k: v for k, v in get_no_of_edges_of_type_x(df).items()},
        **{'auto_' + k: v for k, v in get_no_of_source_nodes_of_type_x(df).items()},
        **{'auto_' + k: v for k, v in get_no_of_target_nodes_of_type_x(df).items()},

        **{'auto_' + k: v for k, v in get_no_of_unique_edge_types(df).items()},
        **{'auto_' + k: v for k, v in get_no_of_unique_source_node_types(df).items()},
        **{'auto_' + k: v for k, v in get_no_of_unique_target_node_type(df).items()},

        **{'auto_' + k: v for k, v in get_total_no_of_nodes(df).items()},
        **{'auto_' + k: v for k, v in get_total_no_of_edges(df).items()},

        **{'expert_' + k: v for k, v in get_no_of_text_nodes_created(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_html_elements_created(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_dom_nodes_created_by_scripts(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_canvas_web_api_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_navigator_web_api_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_screen_web_api_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_webgl_web_api_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_cookie_jar_read_storage_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_cookie_jar_set_storage_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_session_storage_read_storage_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_session_storage_delete_storage_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_local_storage_read_storage_calls(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_local_storage_delete_storage_calls(df).items()},
        **{'expert_' + k: v for k, v in get_sum_of_bytes_transferred_by_requests(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_resources_blocked(df).items()},
        **{'expert_' + k: v for k, v in get_no_of_dom_related_target_nodes(df).items()},
    }


def extract_features(graph_id, df_pre, df_delta, manifest):
    global_pre_features = __extract_auto_and_expert_features_from_graph_df_for_scope(df_pre, manifest, local_scope=False)
    local_pre_features = __extract_auto_and_expert_features_from_graph_df_for_scope(df_pre, manifest, local_scope=True)

    global_delta_features = __extract_auto_and_expert_features_from_graph_df_for_scope(df_delta, manifest, local_scope=False)
    local_delta_features = __extract_auto_and_expert_features_from_graph_df_for_scope(df_delta, manifest, local_scope=True)

    global_relative_features = (pd.DataFrame([global_delta_features]) / pd.DataFrame([global_pre_features])).fillna(0).iloc[0].to_dict()
    local_relative_features = (pd.DataFrame([local_delta_features]) / pd.DataFrame([local_pre_features])).fillna(0).iloc[0].to_dict()

    return {
        **id(graph_id),
        **label(graph_id),

        **{'global_pre_absolute_' + k: v for k, v in global_pre_features.items()},
        **{'local_pre_absolute_' + k: v for k, v in local_pre_features.items()},
        
        **{'global_delta_absolute_' + k: v for k, v in global_delta_features.items()},
        **{'local_delta_absolute_' + k: v for k, v in local_delta_features.items()},
        
        **{'global_relative_' + k: v for k, v in global_relative_features.items()},
        **{'local_relative_' + k: v for k, v in local_relative_features.items()},
    }
