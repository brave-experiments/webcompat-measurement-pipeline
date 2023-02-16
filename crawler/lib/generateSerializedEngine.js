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

const path = require('path');

const { Engine } = require('adblock-rs');
const fs = require('fs-extra');

const { initShardedPath, serializedEnginesDirPath } = require('./paths');

module.exports = async ([filterListDirPath, commitCount, totalCommits]) => {
  const padLength = `${totalCommits}`.length;
  const paddedCommitCount = `${commitCount}`.padStart(padLength);
  const log = (...args) => console.log(`[${paddedCommitCount}]`, ...args);

  const commit = path.basename(filterListDirPath);

  log(`** Generating serialized engine for commit ${commit} (#${commitCount})...`);

  const serializedEngineFilePath = await initShardedPath(serializedEnginesDirPath, `${commit}.dat`);
  if (await fs.pathExists(serializedEngineFilePath)) {
    log('*** Skipped (serialized engine file already exists)');
    return;
  }

  log('*** Indexing filter lists...');

  const filterListFileNames = await fs.readdir(filterListDirPath);

  log(`*** Reading ${filterListFileNames.length} filter list(s)...`);

  let filterListCount = 0;
  const rules = [];
  for (const filterListFileName of filterListFileNames) {
    ++filterListCount;

    log(`**** Reading filter list ${filterListFileName} (#${filterListCount})...`);

    const filterListFilePath = path.join(filterListDirPath, filterListFileName);

    Array.prototype.push.apply(rules,
      (await fs.readFile(filterListFilePath, { encoding: 'utf-8' }))
        .split('\n'));
  }

  log(`*** Initializing engine with ${rules.length} rule(s)...`);

  const engine = new Engine(rules, true);

  log('*** Serializing engine...');

  const serializedEngine = Buffer.from(engine.serialize());

  log('*** Saving serialized engine...');

  await fs.writeFile(serializedEngineFilePath, serializedEngine);
};
