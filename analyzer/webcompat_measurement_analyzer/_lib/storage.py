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

import boto3
import gzip
import json
import networkx
import traceback

class Storage:
    def __init__(self, config):
        self._s3 = boto3.client(
            's3',
            aws_access_key_id=config['awsAccessKeyId'],
            aws_secret_access_key=config['awsSecretAccessKey'],
        )

        self._bucket = config['bucket']
        self._root = config['root']

    def index(self, category, variant):
        params = {'Bucket': self._bucket, 'Prefix': f'{self._root}/snapshots/{category}/'}

        suffix = f'/{variant}/page_graph.graphml'

        while True:
            response = self._s3.list_objects_v2(**params)

            for blob in response['Contents']:
                if blob['Key'].endswith(suffix):
                    yield blob['Key'][: blob['Key'].rindex('/')]

            if 'NextContinuationToken' in response:
                params['ContinuationToken'] = response['NextContinuationToken']
            else:
                break

    def get_page_graph(self, key):
        page_graph_src = (
            self._s3.get_object(Bucket=self._bucket, Key=f'{key}/page_graph.graphml')['Body']
            .read()
            .decode('utf-8')
        )
        try:
            return networkx.graphml.parse_graphml(page_graph_src)
        except:
            return None

    def get_dom_dump(self, key):
        try:
            dom_dump_src = gzip.decompress(
                self._s3.get_object(Bucket=self._bucket, Key=f'{key}/dump.json.gz')['Body'].read()
            ).decode('utf-8')
            return json.loads(dom_dump_src)
        except:
            traceback.print_exc()
            return None
