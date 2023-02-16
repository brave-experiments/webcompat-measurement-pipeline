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

const childProcess = require('child_process');
const fs = require('fs-extra');
const path = require('path');

const git = require('simple-git/promise');
const handlebars = require('handlebars');
const tmp = require('tmp-promise');

const loadEasyListTemplates = require('./loadEasyListTemplates');
const log = require('./log');
const { easyListRepoDirPath, filterListsDirPath, initShardedPath } = require('./paths');

const excludeCookieListsThreshold = new Date('2018-06-01 11:15:06 +0200');
const useSeparateFanboyDirsThreshold = new Date('2013-03-17 15:39:25 +0100');
const tooOldThreshold = new Date('2013-02-14 16:04:38 +0100');

module.exports = async ([commitCount, commit]) => {
  log(`** Processing commit ${commit} (#${commitCount})...`);

  const outputDirPath = await initShardedPath(filterListsDirPath, commit);
  if (await fs.pathExists(outputDirPath)) {
    log('*** Skipped (output directory already exists)');
    return;
  }

  log('*** Opening repo...');

  const repo = git(easyListRepoDirPath);

  log('*** Retrieving commit date...');

  const commitDate = new Date(await repo.show(['-s', '--format=%ci', commit]));

  log(`**** Date: ${commitDate}`);

  if (commitDate <= tooOldThreshold) {
    log('*** Skipped (commit too old)');
    return;
  }

  const templateData = {
    includeCookieLists: true,
    fanboyAnnoyanceDir: 'fanboy-addon',
    fanboySocialDir: 'fanboy-addon',
    fanboyCookieDir: 'fanboy-addon'
  };

  if (commitDate <= excludeCookieListsThreshold) {
    templateData.includeCookieLists = false;
  }

  if (commitDate <= useSeparateFanboyDirsThreshold) {
    templateData.fanboyAnnoyanceDir = 'fanboy-annoyance';
    templateData.fanboySocialDir = 'fanboy-social';
    templateData.fanboyCookieDir = 'fanboy-cookie';
  }

  return;

  log('*** Initializing temporary working directory...');

  await tmp.withDir(async workingDir => {
    log('*** Checking out commit...');

    await repo.env('GIT_WORK_TREE', workingDir.path).checkout(['-f', commit]);

    log('*** Populating templates...');

    const templates = await loadEasyListTemplates();

    const templateFilePaths = [];
    for (const [templateName, template] of templates) {
      const templateFilePath = path.join(workingDir.path, `${templateName}.template`);
      templateFilePaths.push(templateFilePath);

      const templateSrc = template(templateData);

      try {
        await fs.writeFile(templateFilePath, templateSrc, { flags: 'wx' });
      } catch (error) {
        if (error.code === 'EEXIST') {
          // File already exists; skip copy.
        } else {
          throw error;
        }
      }
    }

    log('*** Initializing output directory...');

    await fs.mkdir(outputDirPath, { recursive: true });

    log(`*** Generating ${templateFilePaths.length} filter list(s)...`);

    let filterListCount = 0;
    for (const templateFilePath of templateFilePaths) {
      ++filterListCount;

      const outputFileName = `${path.basename(templateFilePath, '.template')}.txt`;

      log(`**** Generating ${outputFileName} (#${filterListCount})...`);

      const renderProcess = childProcess.spawn('poetry', [
        'run', 'flrender', '-i', `easylist=${workingDir.path}`, templateFilePath,
        path.join(outputDirPath, outputFileName)
      ], {
        cwd: __dirname,
        stdio: 'inherit'
      });

      await new Promise((resolve, reject) => {
        renderProcess.on('error', error => {
          reject(error);
        });
        renderProcess.on('exit', (code, signal) => {
          if (code === 0) {
            resolve();
          } else if (code != null) {
            reject(new Error(`Unexpected exit code ${code}`));
          } else {
            reject(new Error(`Process exited due to signal ${signal}`));
          }
        });
      });
    }
  }, { unsafeCleanup: true });
};
