// vim: set tw=99 ts=2 sts=2 sw=2 et:

// Copyright (C) 2022-23 Michael Smith <michael@spinda.net> (https://spinda.net)

// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

'use strict';

const os = require('os');
const path = require('path');

const glob = require('glob-promise');
const { PriorityQueue } = require('multiprocessing');
const shuffle = require('shuffle-array');

const { filterListsDirPath } = require('./lib/paths');

(async () => {
  console.log('* Indexing commits from generated filter lists...');

  const filterListDirPaths = await glob(path.join(filterListsDirPath, '*', '*'));
  shuffle(filterListDirPaths);

  console.log(`* Generating serialized engines for ${filterListDirPaths.length} commit(s)...`);

  const inputs = filterListDirPaths.map((filterListDirPath, index) => [filterListDirPath, index, filterListDirPaths.length]);

  const queue = new PriorityQueue(os.cpus().length);
  try {
    const generateSerializedEngineModulePath = path.join(__dirname, 'lib', 'generateSerializedEngine');
    await Promise.all(filterListDirPaths.map((filterListDirPath, commitIndex) => queue.push([filterListDirPath, commitIndex + 1, filterListDirPaths.length], filterListDirPaths.length - commitIndex, generateSerializedEngineModulePath)));
  } finally {
    queue.pool.close();
  }
})();
