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

import os

def get_shard_dir_name(name):
    return name[0]

def get_sharded_path(parent_dir_path, name):
    return os.path.join(parent_dir_path, get_shard_dir_name(name), name)

lib_dir_path = os.path.dirname(os.path.realpath(__file__))
package_dir_path = os.path.dirname(lib_dir_path)
analyzer_dir_path = os.path.dirname(package_dir_path)
root_dir_path = os.path.dirname(analyzer_dir_path)

data_dir_path = os.path.join(root_dir_path, 'data')

easylist_dir_path = os.path.join(data_dir_path, 'easylist')
entries_dir_path = os.path.join(data_dir_path, 'entries')
snapshots_dir_path = os.path.join(data_dir_path, 'snapshots')
page_graphs_dir_path = os.path.join(data_dir_path, 'page_graphs')
subgraphs_dir_path = os.path.join(data_dir_path, 'subgraphs')
metrics_dir_path = os.path.join(data_dir_path, 'metrics')
summaries_dir_path = os.path.join(data_dir_path, 'summaries')
splits_dir_path = os.path.join(data_dir_path, 'splits')

differents_file_path = os.path.join(data_dir_path, 'differents.txt')
labels_file_path = os.path.join(data_dir_path, 'labels.json')
report_file_path = os.path.join(data_dir_path, 'report.json')

models_dir_path = os.path.join(data_dir_path, 'models')
classifier_models_dir_path = os.path.join(models_dir_path, 'classifier')
vectorizer_models_dir_path = os.path.join(models_dir_path, 'vectorizer')
