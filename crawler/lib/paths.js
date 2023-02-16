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

const fs = require('fs-extra');
const path = require('path');

exports.getShardDirName = name => name[0];

exports.getShardedPath = (parentDirPath, name) => path.join(parentDirPath, exports.getShardDirName(name), name);

exports.initShardedPath = async (parentDirPath, name) => {
  const shardDirName = exports.getShardDirName(name);
  const shardDirPath = path.join(parentDirPath, shardDirName);
  const shardedPath = path.join(shardDirPath, name);
  await fs.mkdir(shardDirPath, { recursive: true });
  return shardedPath;
};

exports.crawlerDirPath = path.dirname(__dirname);
exports.rootDirPath = path.dirname(exports.crawlerDirPath);

exports.easyListTemplatesDirPath = path.join(exports.crawlerDirPath, 'easylist-templates');

exports.binDirPath = path.join(exports.rootDirPath, 'bin');
exports.dataDirPath = path.join(exports.rootDirPath, 'data');

exports.browserExeFilePath = path.join(exports.binDirPath, 'brave-browser-page-graph', 'brave');

exports.easyListRepoDirPath = path.join(exports.dataDirPath, 'easylist');
exports.entriesDirPath = path.join(exports.dataDirPath, 'entries');
exports.filterListsDirPath = path.join(exports.dataDirPath, 'filter-lists');
exports.serializedEnginesDirPath = path.join(exports.dataDirPath, 'serialized-engines');
exports.coveredCommitsFilePath = path.join(exports.dataDirPath, 'covered-commits.txt');
exports.snapshotsDirPath = path.join(exports.dataDirPath, 'snapshots');
