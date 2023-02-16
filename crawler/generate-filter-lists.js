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
const os = require('os');
const path = require('path');

const async = require('async');
const fs = require('fs-extra');
const glob = require('glob-promise');
const handlebars = require('handlebars');
const rmfr = require('rmfr');
const shuffle = require('shuffle-array');
const git = require('simple-git/promise');
const tmp = require('tmp-promise');

const { easyListRepoDirPath, easyListTemplatesDirPath, entriesDirPath, filterListsDirPath, initShardedPath } = require('./lib/paths');

const variants = ['parent', 'child'];

const useSeparateFanboyDirsThreshold = new Date('2013-03-17 15:39:25 +0100');
const tooOldThreshold = new Date('2013-02-14 16:04:38 +0100');

const fanboyCookieOptions = {
  includeFanboyCookieGeneralBlock: 'fanboy_cookie_general_block.txt',
  includeFanboyCookieGeneralHide: 'fanboy_cookie_general_hide.txt',
  includeFanboyCookieThirdParty: 'fanboy_cookie_thirdparty.txt',
  includeFanboyCookieSpecificBlock: 'fanboy_cookie_specific_block.txt',
  includeFanboyCookieSpecificHide: 'fanboy_cookie_specific_hide.txt',
  includeFanboyCookieInternationalSpecificHide: 'fanboy_cookie_international_specific_hide.txt',
  includeFanboyCookieInternationalSpecificBlock: 'fanboy_cookie_international_specific_block.txt',
  includeFanboyCookieWhitelistGeneralHide: 'fanboy_cookie_whitelist_general_hide.txt',
  includeFanboyCookieWhitelist: 'fanboy_cookie_whitelist.txt'
};

const concurrency = os.cpus().length;

(async () => {
  console.log('* Loading templates...');

  const templates = new Map();

  const templateFileNames = await fs.readdir(easyListTemplatesDirPath);

  for (const templateFileName of templateFileNames) {
    const templateFilePath = path.join(easyListTemplatesDirPath, templateFileName);
    const templateName = path.basename(templateFileName, '.handlebars');
    templates.set(templateName,
      handlebars.compile(await fs.readFile(templateFilePath, { encoding: 'utf-8' })));
  }

  console.log('* Retrieving commit metadata from repository...');

  const historyEntries = new Map();
  for (const historyEntry of (await git(easyListRepoDirPath).log(['origin/master'])).all) {
    historyEntries.set(historyEntry.hash, historyEntry);
  }

  console.log(`** Retrieved metadata for ${historyEntries.size} commit(s)`);

  console.log('* Indexing entries...');

  const entryFilePaths = await glob(path.join(entriesDirPath, '*', '*', '*.json'));

  console.log(`* Collecting commits from ${entryFilePaths.length} entr(y/ies)...`);

  let commits = (await async.map(entryFilePaths, async entryFilePath => {
    const entry = JSON.parse(await fs.readFile(entryFilePath));
    return variants.map(variant => entry[variant]);
  })).flat();
  commits = commits.filter((commit, index) => commits.indexOf(commit) === index);
  shuffle(commits);

  console.log(`* Generating filter lists for ${commits.length} unique commit(s)...`);

  const padLength = `${commits.length}`.length;

  await async.eachOfLimit(commits, concurrency, async (commit, commitIndex) => {
    const commitCount = commitIndex + 1;
    const paddedCommitCount = `${commitCount}`.padStart(padLength);
    const log = (...args) => console.log(`[${paddedCommitCount}]`, ...args);
    const logError = (...args) => console.error(`[${paddedCommitCount}]`, ...args);

    let commitDate = null;
    try {
      commitDate = new Date(historyEntries.get(commit).date);
    } catch (err) {
      throw err;
    }

    log(`** Processing commit ${commit} from ${commitDate} (#${commitIndex + 1})...`);

    if (commitDate <= tooOldThreshold) {
      log('*** Skipped (commit too old)');
      return;
    }

    const outputDirPath = await initShardedPath(filterListsDirPath, commit);
    if (await fs.pathExists(outputDirPath)) {
      log('*** Skipped (output directory already exists)');
      return;
    }

    log('*** Initializing temporary working directory...');

    await tmp.withDir(async workingDir => {
      log('*** Checking out commit...');

      await git().clone(easyListRepoDirPath, workingDir.path, ['--shared', '--no-checkout']);
      await git(workingDir.path).checkout(['-f', commit]);

      log('*** Populating templates...');

      const templateData = {
        fanboyAnnoyanceDir: 'fanboy-addon',
        fanboySocialDir: 'fanboy-addon',
        fanboyCookieDir: 'fanboy-addon'
      };

      if (commitDate <= useSeparateFanboyDirsThreshold) {
        templateData.fanboyAnnoyanceDir = 'fanboy-annoyance';
        templateData.fanboySocialDir = 'fanboy-social';
        templateData.fanboyCookieDir = 'fanboy-cookie';
      }

      for (const [optionKey, optionFileName] of Object.entries(fanboyCookieOptions)) {
        templateData[optionKey] = await fs.pathExists(path.join(workingDir.path, templateData.fanboyCookieDir, optionFileName));
      }

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

        try {
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
        } catch (error) {
          logError(error.stack);
          await rmfr(outputDirPath);
          break;
        }
      }
    }, { unsafeCleanup: true });
  });
})();
