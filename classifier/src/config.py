from enum import Enum
import os

BASE_PATH = '/Volumes/brave-build-drive/downloads'
PG_DATA_PATH = 'web-compat-combined-dataset'
BROKEN_DIR = 'broken'
UNBROKEN_DIR = 'unbroken'
GRAPH_DF_DIR = 'graph_dfs'
TRAINING_DIR = 'training'
TRAINING_DATA_PATH = os.path.join(BASE_PATH, TRAINING_DIR, 'data.csv')
PRE_INTERVENTION_PATH = 'without/page_graph.graphml'
POST_INTERVENTION_PATH = 'with/page_graph.graphml'
DELTA_PATH = 'delta.graphml'
MANIFEST_PATH = 'with/manifest.json'  # "with" and "without" manifest files are the same (spot-checked 10 files or so)
GRAPH_METADATA_PATH = 'graph-metadata.csv'
NEGATIVE_CLASS_SUFFIX = '-0'
POSITIVE_CLASS_SUFFIX = '-1'
LABEL_NAME = 'did_break'
ID_NAME = 'graph_id'

class GraphType(Enum):
    PRE_INTERVENTION = 1
    POST_INTERVENTION = 2
    DELTA = 3

XGB_DEFAULT_PARAMS = {
    'use_label_encoder': False,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}

XGB_DEFAULT_GRID_PARAMS = {
    # NOT USED, but left for reference as per https://xgboost.readthedocs.io/en/stable/parameter.html
    'model__base_score': (0.5),  # The initial prediction score of all instances, global bias (0, 1)
    'model__colsample_bylevel': (1),  # Family of parameters for subsampling of columns (0, 1]
    'model__colsample_bynode': (1),
    'model__colsample_bytree': (1),
    'model__gamma': (0),  # Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. [0, +inf]
    'model__learning_rate': (0.3),
    'model__max_delta_step': (0),  # Maximum delta step we allow each leaf output to be [0, +inf]
    'model__max_depth': (6),  # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit [0, +inf]
    'model__min_child_weight': (1),  # Minimum sum of instance weight (hessian) needed in a child [0, +inf]
    'model__n_estimators': (100),
    'model__num_parallel_tree': (1),  # Number of parallel trees constructed during each iteration
    'model__reg_alpha': (0),  # L1 regularization term on weights. Increasing this value will make model more conservative
    'model__reg_lambda': (1),  # L2 regularization term on weights. Increasing this value will make model more conservative
    'model__scale_pos_weight': (1),  # Control the balance of positive and negative weights, useful for unbalanced classes.
    'model__subsample': (1),  # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees (0, 1]
}

GRAPH_DF_COLUMNS = {
    'edge_source': str,
    'edge_target': str,
    'edge_id': object,
    'edge_event_listener_id': object,
    'edge_timestamp': object,
    'edge_request_id': object,
    'edge_script_id': object,
    'edge_resource_type': str,
    'edge_key': str,
    'edge_value': str,
    'edge_request_type': str,
    'edge_attr_name': str,
    'edge_edge_type': str,
    'edge_args': str,
    'edge_is_alt': object,
    'edge_parent': object,
    'edge_is_style': object,
    'edge_before': object,
    'edge_status': str,
    'source_node_type': str,
    'source_id': object,
    'source_timestamp': object,
    'source_node_id': object,
    'source_is_deleted': object,
    'source_tag_name': str,
    'source_url': str,
    'source_script_id': object,
    'source_script_type': str,
    'source_text': str,
    'source_method': str,
    'source_rule': str,
    'source_is_alt': str,
    'target_node_type': str,
    'target_id': object,
    'target_timestamp': object,
    'target_node_id': object,
    'target_is_deleted': object,
    'target_tag_name': str,
    'target_url': str,
    'target_script_id': object,
    'target_script_type': str,
    'target_text': str,
    'target_method': str,
    'target_rule': str,
    'target_is_alt': str,
    'edge_response_hash': str,
}
