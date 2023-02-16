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

const fs = require('fs-extra');
const linkify = require('linkifyjs');
const git = require('simple-git/promise');

const { easyListRepoDirPath, entriesDirPath, initShardedPath } = require('./lib/paths');

const categories = {
  A: 'unbroken',
  P: 'broken'
};

(async () => {
  console.log('* Loading commit history...');

  const repo = git(easyListRepoDirPath);
  const commits = (await repo.log(['origin/master'])).all;

  console.log(`* Parsing ${commits.length} commits...`);

  let count = 0;
  for (const commit of commits) {
    ++count;

    if (commit.message.toLowerCase().indexOf('(anti-adblock)') >= 0 ||
        commit.message.toLowerCase().indexOf('(anti adblock)') >= 0) {
      continue;
    }

    const commitPrefix = commit.message[0];
    if (!categories.hasOwnProperty(commitPrefix)) {
      continue;
    }
    const category = categories[commitPrefix];

    console.log(`** Parsing ${category} commit ${commit.hash} (#${count})...`);

    const categoryEntriesDirPath = path.join(entriesDirPath, category);
    const entryFilePath = await initShardedPath(categoryEntriesDirPath, `${commit.hash}.json`);
    if (await fs.pathExists(entryFilePath)) {
      console.log(`*** Skipped (entry file already exists)`);
      continue;
    }

    console.log(`*** Commit message: ${commit.message.split('\n')[0]}`);
    console.log(`*** Commit date: ${commit.date}`);

    console.log('*** Extracting URLs...');

    let urls = linkify.find(commit.message)
      .filter(link => link.type === 'url')
      .map(link => link.href)
      .filter(url =>
        url.indexOf('lanik.us') < 0 &&
        url.indexOf('github.com') < 0 &&
        url.indexOf('community.brave.com') < 0 &&
        url.indexOf('sourceforge.net/apps/trac/sourceforge') < 0);

    // Filter out duplicate URLs.
    urls = urls.filter((url, index) => urls.indexOf(url) === index);

    if (urls.length < 1) {
      console.log('*** Skipped (no URL(s) extracted)');
      continue;
    }

    for (const url of urls) {
      console.log(`**** Extracted URL: ${url}`);
    }

    console.log('*** Determining parent commit hash...');
    const parent = await repo.revparse([`${commit.hash}^`])
    console.log(`**** Parent commit hash: ${parent}`);

    const entry = {
      child: commit.hash,
      parent: parent,
      date: commit.date,
      message: commit.message,
      urls: urls
    };

    console.log('*** Saving data...');
    await fs.writeFile(entryFilePath, JSON.stringify(entry));
  }
})();
