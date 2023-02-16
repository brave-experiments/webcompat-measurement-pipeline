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
import json
import os
import subprocess

from ._lib.paths import (
    easylist_dir_path,
    entries_dir_path,
    get_sharded_path,
    metrics_dir_path,
    labels_file_path,
    report_file_path,
)

def _load_entry(category, commit):
    category_entries_dir_path = os.path.join(entries_dir_path, category)
    entry_file_path = get_sharded_path(category_entries_dir_path, f'{commit}.json')
    with open(entry_file_path, 'r', encoding='utf-8') as entry_file:
        return json.load(entry_file)

def _load_metrics(category, commit, url_index):
    category_metrics_dir_path = os.path.join(metrics_dir_path, category)
    commit_metrics_dir_path = get_sharded_path(category_metrics_dir_path, commit)
    metrics_file_path = os.path.join(commit_metrics_dir_path, f'{url_index}.json')
    with open(metrics_file_path, 'r', encoding='utf-8') as metrics_file:
        return json.load(metrics_file)

def run():
    print('* Generating report...')

    with open(report_file_path, 'w', encoding='utf-8') as report_file:
        with open(labels_file_path, 'r', encoding='utf-8') as labels_file:
            for index, line in enumerate(labels_file):
                print(f'** Processing row #{index + 1}...')

                row = json.loads(line)

                category = row['category']
                commit = row['commit']
                url_index = row['urlIndex']

                entry = _load_entry(category, commit)
                row['url'] = entry['urls'][url_index - 1]
                row['commit'] = {
                    'id': commit,
                    'parent': entry['parent'],
                    'date': entry['date'],
                    'message': entry['message']
                }

                metrics = _load_metrics(category, commit, url_index)
                # del metrics['delta is script block']
                # del metrics['subgraph is script block']
                row['metrics'] = metrics

                git_process = subprocess.run(['git', 'diff', f'{commit}~', commit],
                    cwd=easylist_dir_path, capture_output=True, encoding='utf-8')
                row['commit']['diff'] = git_process.stdout

                json.dump(row, report_file)
                report_file.write('\n')

if __name__ == '__main__':
    run()
