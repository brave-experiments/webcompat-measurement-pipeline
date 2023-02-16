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

const proxyThroughWaybackMachine = false;

const os = require('os');
const path = require('path');
const process = require('process');

const AnyProxy = proxyThroughWaybackMachine ? require('anyproxy') : null;
const async = require('async');
const sleep = require('await-sleep');
const axios = require('axios');
const Timeout = require('await-timeout');
const fs = require('fs-extra');
const glob = require('glob-promise');
const { gzip } = require('node-gzip');
const moment = require('moment');
const puppeteer = require('puppeteer-core');
const tmp = require('tmp-promise');

const variants = [
  {
    'id': 'parent',
    'alt': 'child'
  },
  {
    'id': 'child',
    'alt': 'parent'
  }
];

const concurrency = Math.ceil(os.cpus().length / 6);

const browserSettleDelay = 1000 * 15;

const retryMax = 5;
const retryDelay = 1000 * 5;

const pageGraphTimeout = 1000 * 60;
const dumpTimeout = 1000 * 60 * 5;
const screenshotTimeout = 1000 * 60;

const viewportWidth = 1920;
const viewportHeight = 1080;

const { browserExeFilePath, coveredCommitsFilePath, entriesDirPath, getShardedPath, serializedEnginesDirPath, snapshotsDirPath } = require('./lib/paths');

const adBlockerUpdaterExtensionId = 'cffkpbalmllkdoenhmdmpbkajipdjfam';
const adBlockerUpdaterExtensionVersionRegex = /^[0-9]+\.[0-9]+\.[0-9]+$/;

const dstSerializedEngineFileName = 'rs-ABPFilterParserData.dat';
const dstAltSerializedEngineFileName = 'rs-ABPFilterParserDataAlt.dat';

const proxyPorts = proxyThroughWaybackMachine ? new Set() : null;

const launchBrowser = (profileDirPath, extraArgs = []) => {
  return puppeteer.launch({
    executablePath: browserExeFilePath,
    userDataDir: profileDirPath,
    args: [
      /*
      '--enable-logging=stderr',
      '--v=0',
      */
      '--disable-brave-update',
      '--start-fullscreen',
      ...extraArgs
    ],
    ignoreDefaultArgs: [
      '--enable-features=NetworkService,NetworkServiceInProcess'
      // ^ breaks Brave Shields
    ],
    /*
    dumpio: true,
    */
    headless: false
  });
};

(async () => {
  // Disable faulty heuristic we trip due to our concurrency.
  process.setMaxListeners(0);

  /*
  console.log('* Reading covered commit hashes...');

  const coveredCommits = new Set((await fs.readFile(coveredCommitsFilePath, { encoding: 'utf-8' })).split('\n').filter(line => line));
  */

  console.log('* Initializing clean Brave profile...');

  await tmp.withDir(async baseProfileDir => {
    const browser = await launchBrowser(baseProfileDir.path);
    await sleep(browserSettleDelay);
    await browser.close();

    console.log('* Finding ad blocker updater extension version...');

    const adBlockerUpdaterExtensionDirListing =
      await fs.readdir(path.join(baseProfileDir.path, adBlockerUpdaterExtensionId));
    const adBlockerUpdaterExtensionVersion =
      adBlockerUpdaterExtensionDirListing.find(x => adBlockerUpdaterExtensionVersionRegex.test(x));
    if (!adBlockerUpdaterExtensionVersion) {
      throw new Error('Failed to identify ad blocker updater extension version');
    }

    console.log(`** Version: ${adBlockerUpdaterExtensionVersion}`);

    console.log('* Indexing categories...');

    const categories = await fs.readdir(entriesDirPath);

    console.log(`* Taking snapshots for ${categories.length} categor(y/ies)...`);

    let categoryCount = 0;
    for (const category of categories) {
      ++categoryCount;

      console.log(`** Taking snapshots for category ${category} (#${categoryCount})...`);

      console.log('*** Indexing entries for category...');

      const categoryEntriesDirPath = path.join(entriesDirPath, category);
      const entryFilePaths = await glob(path.join(categoryEntriesDirPath, '*', '*.json'));

      const categorySnapshotsDirPath = path.join(snapshotsDirPath, category);

      console.log(`*** Taking snapshots for ${entryFilePaths.length} ${category} entr(y/ies)...`);

      const padLength = `${entryFilePaths.length}`.length;

      await async.eachOfLimit(entryFilePaths, concurrency, async (entryFilePath, entryIndex) => {
        const entryCount = entryIndex + 1;
        const paddedEntryCount = `${entryCount}`.padStart(padLength);
        const log = (...args) => console.log(`[${paddedEntryCount}]`, ...args);
        const logError = (...args) => console.error(`[${paddedEntryCount}]`, ...args);

        log(`**** Taking snapshots for ${category} entry #${entryCount}...`);

        log('***** Loading data...');

        const entry = JSON.parse(await fs.readFile(entryFilePath, { encoding: 'utf-8' }));
        log(`****** Parent: ${entry.parent}`);
        log(`****** Child: ${entry.child}`);
        log(`****** Date: ${entry.date}`);
        log(`****** Message: ${entry.message}`);
        log(`****** Extracted URLs: ${entry.urls.join(' ')}`);

        /*
        if (coveredCommits.has(entry.child)) {
          log('***** Skipped (commit already covered)');
          return;
        }
        */

        const variantSerializedEngineFilePaths = {};
        for (const variant of variants) {
          const variantCommit = entry[variant.id];
          const variantSerializedEngineFilePath = getShardedPath(serializedEnginesDirPath, `${variantCommit}.dat`);
          if (!await fs.pathExists(variantSerializedEngineFilePath)) {
            log(`***** Skipped (no serialized engine for ${variant.id} ${variantCommit})`);
            return;
          }
          variantSerializedEngineFilePaths[variant.id] = variantSerializedEngineFilePath;
        }

        log(`***** Taking snapshots for ${entry.urls.length} URL(s)...`);

        let urlCount = 0;
        for (const url of entry.urls) {
          ++urlCount;

          log(`****** Taking snapshots for URL #${urlCount}: ${url}...`);

          log(`******* Taking snapshots for ${variants.length} variant(s)...`);

          let variantCount = 0;
          for (const variant of variants) {
            ++variantCount;

            log(`******** Taking snapshot for variant ${variant.id} (#${variantCount})...`);

            const snapshotDirPath = path.join(getShardedPath(categorySnapshotsDirPath, entry.child), `${urlCount}`, variant.id);
            if (await fs.pathExists(snapshotDirPath)) {
              log('********* Skipped (snapshot directory already exists)');
              return;
            }

            log('********* Initializing snapshot directory...');

            await fs.mkdir(snapshotDirPath, { recursive: true });
            const consoleLogFilePath = path.join(snapshotDirPath, 'console.json');
            const pageGraphFilePath = path.join(snapshotDirPath, 'page_graph.graphml');
            const dumpFilePath = path.join(snapshotDirPath, 'dump.json.gz');
            const screenshotFilePath = path.join(snapshotDirPath, 'screenshot.png');
            const manifestFilePath = path.join(snapshotDirPath, 'manifest.json');

            log('********* Initializing browser profile...');
            await tmp.withDir(async profileDir => {
              await fs.copy(baseProfileDir.path, profileDir.path);

              log('********* Populating serialized engines...');

              const adBlockerUpdaterExtensionDirPath =
                path.join(profileDir.path, adBlockerUpdaterExtensionId);
              const adBlockerUpdaterSerializedEnginesDirPath =
                path.join(adBlockerUpdaterExtensionDirPath, adBlockerUpdaterExtensionVersion);

              const srcSerializedEngineFilePath = variantSerializedEngineFilePaths[variant.id];
              const dstSerializedEngineFilePath =
                path.join(adBlockerUpdaterSerializedEnginesDirPath, dstSerializedEngineFileName);
              await fs.copy(srcSerializedEngineFilePath, dstSerializedEngineFilePath);

              const srcAltSerializedEngineFilePath = variantSerializedEngineFilePaths[variant.alt];
              const dstAltSerializedEngineFilePath =
                path.join(adBlockerUpdaterSerializedEnginesDirPath, dstAltSerializedEngineFileName);
              await fs.copy(srcAltSerializedEngineFilePath, dstAltSerializedEngineFilePath);

              let proxyServer = null;
              let proxyPort = null;

              if (proxyThroughWaybackMachine) {
                log('********* Launching proxy server...');

                const entryDate = new Date(entry.date);

                const waybackTimestamp = moment(entryDate).utc().format('YYYYMMDDHHmmss');

                const proxyRule = {
                  summary: 'rewrite request through the Wayback Machine',
                  async beforeSendRequest (request) {
                    try {
                      const response = await axios.head(`https://web.archive.org/web/${encodeURIComponent(waybackTimestamp)}id_/${encodeURIComponent(request.url)}`);
                      const waybackUrl = new URL(response.request.res.responseUrl);
                      return {
                        protocol: 'https',
                        url: waybackUrl.toString(),
                        requestOptions: {
                          ...request.requestOptions,
                          hostname: waybackUrl.hostname,
                          port: 443,
                          path: `${waybackUrl.pathname}${waybackUrl.search}`,
                          headers: {
                            ...request.requestOptions.headers,
                            Host: waybackUrl.host
                          }
                        }
                      };
                    } catch (error) {
                      return request;
                    }
                  },
                  async beforeSendResponse (request, response) {
                    const newHeader = { ...response.response.header };
                    delete newHeader['Content-Security-Policy'];
                    delete newHeader['Link'];

                    return {
                      ...response,
                      response: {
                        ...response.response,
                        header: newHeader
                      }
                    };
                  }
                };

                do {
                  proxyPort = Math.floor(Math.random() * 9999 + 10000);
                } while (proxyPorts.has(proxyPort));
                proxyPorts.add(proxyPort);

                const proxyOptions = {
                  port: proxyPort,
                  rule: proxyRule,
                  forceProxyHttps: true,
                  wsIntercept: false,
                  dangerouslyIgnoreUnauthorized: true,
                  silent: true
                };

                while (!proxyServer) {
                  try {
                    proxyServer = new AnyProxy.ProxyServer(proxyOptions);
                  } catch (error) {
                    if (!error.code || error.code !== 'EEXIST') {
                      throw error;
                    }
                  }
                }

                proxyServer.on('ready', () => {
                  log('********** Proxy server ready');
                });

                proxyServer.on('error', error => {
                  logError('Proxy server error', error.stack);
                });

                proxyServer.start();
              }

              let result = {
                visit: { status: 'done' },
                console: { status: 'done' },
                snapshot: { status: 'done' },
                dump: { status: 'done' },
                screenshot: { status: 'done' }
              };

              try {
                log('********* Launching browser...');

                const extraArgs = proxyThroughWaybackMachine ? [
                  '--ignore-certificate-errors',
                  `--proxy-server=127.0.0.1:${proxyPort}`
                ] : [];

                const browser = await launchBrowser(profileDir.path, extraArgs);

                log('********* Opening new tab...');

                const page = await browser.newPage();

                let pageError = null;
                let pageErrorHandled = false;
                page.on('error', error => {
                  pageError = error;
                });
                page.on('dialog', async dialog => {
                  await dialog.dismiss();
                });

                log('********* Resizing tab...');

                await page.setViewport({
                  width: viewportWidth,
                  height: viewportHeight
                });

                log('********* Opening DevTools tab...');

                const devToolsPage = await browser.newPage();

                log('********* Navigating to DevTools...');

                const browserWsEndpoint = browser.wsEndpoint();
                const pageTargetId = page.target()._targetId;
                const pageWsEndpoint = `${browserWsEndpoint.split('/browser/')[0]}/page/${pageTargetId}`;
                const pageDevToolsUrl = `chrome-devtools://devtools/bundled/devtools_app.html?ws=${pageWsEndpoint.substring(5)}`;
                await devToolsPage.goto(pageDevToolsUrl);

                log('********* Activating DevTools console...');

                await devToolsPage.waitForSelector('.tabbed-pane', { visible: true });
                const consoleTab = await devToolsPage.evaluateHandle('document.querySelector(".tabbed-pane").shadowRoot.querySelector("#tab-console")');
                await consoleTab.click();

                log('********* Flipping to main tab...');

                await page.bringToFront();

                log(`********* Navigating to ${url}...`);

                let navigationAttempts = 0;
                while (true) {
                  try {
                    await page.goto(url);
                    break;
                  } catch (error) {
                    if (error.message.indexOf('net::') >= 0) {
                      if (++navigationAttempts < retryMax) {
                        log(`********** Pausing (network error: ${error.message})`);
                        await sleep(retryDelay);
                        log(`********** Retrying`);
                      } else {
                        log(`********** Skipped (network error: ${error.message})`);
                        await browser.close();
                        throw error;
                      }
                    } else if (error.name === 'TimeoutError') {
                      break;
                    } else {
                      throw error;
                    }
                  }
                }

                log('********* Waiting...');

                await sleep(browserSettleDelay);

                if (pageError) {
                  logError(pageError.stack);

                  result.visit.status = 'error';
                  result.visit.error = {
                    name: pageError.name,
                    message: pageError.message
                  };

                  result.console.status = 'skipped';
                  result.snapshot.status = 'skipped';
                  result.dump.status = 'skipped';
                  result.screenshot.status = 'skipped';

                  pageErrorHandled = true;
                }

                if (!pageError) {
                  log('********* Flipping to DevTools tab...');

                  await devToolsPage.bringToFront();

                  try {
                    log('********* Dumping console messages...');

                    const messages = await devToolsPage.evaluate('window.dumpConsole()');

                    if (Array.isArray(messages)) {
                      log(`********** Saving console messages (${messages.length} messages)...`);

                      await fs.writeFile(consoleLogFilePath, JSON.stringify(messages, null, 2));
                    }
                  } catch (error) {
                    log(error.stack);

                    result.console.status = 'error';
                    result.console.error  = {
                      name: error.name,
                      message: error.message
                    };
                  } finally {
                    log('********* Flipping to main tab...');

                    await page.bringToFront();
                  }
                }

                if (pageError && !pageErrorHandled) {
                  logError(pageError.stack);

                  result.console.status = 'error';
                  result.console.error = {
                    name: pageError.name,
                    message: pageError.message
                  };

                  result.snapshot.status = 'skipped';
                  result.dump.status = 'skipped';
                  result.screenshot.status = 'skipped';

                  pageErrorHandled = true;
                }

                let client = null;
                if (!pageError) {
                  client = await page.target().createCDPSession();
                }

                if (!pageError) {
                  const pageGraphTimer = new Timeout();

                  try {
                    log('********* Grabbing page graph...');

                    const response = await Promise.race([
                      client.send('Page.generatePageGraph'),
                      pageGraphTimer.set(pageGraphTimeout, 'Page graph capture timed out')
                    ]);

                    const pageGraph = response.data;

                    log(`********** Saving page graph (${pageGraph.length} characters)...`);

                    await fs.writeFile(pageGraphFilePath, pageGraph);
                  } catch (error) {
                    logError(error.stack);

                    result.snapshot.status = 'error';
                    result.snapshot.error = {
                      name: error.name,
                      message: error.message
                    };
                  } finally {
                    pageGraphTimer.clear();
                  }
                }

                if (pageError && !pageErrorHandled) {
                  logError(pageError.stack);

                  result.snapshot.status = 'error';
                  result.snapshot.error = {
                    name: pageError.name,
                    message: pageError.message
                  };

                  result.dump.status = 'skipped';
                  result.screenshot.status = 'skipped';

                  pageErrorHandled = true;
                }

                if (!pageError) {
                  const dumpTimer = new Timeout();

                  try {
                    log('********* Dumping DOM...');

                    const response = await Promise.race([
                      (async () => {
                        await client.send('DOM.enable');
                        await client.send('CSS.enable');

                        return await client.send('DOM.getDocument', {
                          depth: -1,
                          pierce: true
                        });
                      })(),
                      dumpTimer.set(dumpTimeout, 'DOM dump timed out')
                    ]);

                    let dump = JSON.stringify(response);

                    if (!pageError) {
                      log(`********** Compressing DOM dump (${dump.length} characters)...`);

                      dump = await gzip(dump);

                      log(`********** Saving DOM dump (${dump.length} bytes)...`);

                      await fs.writeFile(dumpFilePath, dump);
                    }
                  } catch (error) {
                    logError(error.stack);

                    result.dump.status = 'error';
                    result.dump.error = {
                      name: error.name,
                      message: error.message
                    };
                  } finally {
                    dumpTimer.clear();
                  }
                }

                if (pageError && !pageErrorHandled) {
                  logError(pageError.stack);

                  result.dump.status = 'error';
                  result.dump.error = {
                    name: pageError.name,
                    message: pageError.message
                  };

                  result.screenshot.status = 'skipped';

                  pageErrorHandled = true;
                }

                if (!pageError) {
                  const screenshotTimer = new Timeout();

                  try {
                    log('********* Taking screenshot...');

                    const screenshot = await Promise.race([
                      page.screenshot({ fullPage: true }),
                      screenshotTimer.set(screenshotTimeout, 'Screenshot timed out')
                    ]);

                    if (!pageError && screenshot) {
                      log('********** Saving screenshot...');

                      await fs.writeFile(screenshotFilePath, screenshot);
                    }
                  } catch (error) {
                    logError(error.stack);

                    result.screenshot.status = 'error';
                    result.screenshot.error = {
                      name: error.name,
                      message: error.message
                    };
                  } finally {
                    screenshotTimer.clear();
                  }
                }

                if (pageError && !pageErrorHandled) {
                  logError(pageError.stack);

                  result.screenshot.status = 'error';
                  result.screenshot.error = {
                    name: pageError.name,
                    message: pageError.message
                  };

                  pageErrorHandled = true;
                }

                log('********* Closing browser...');
                await browser.close();
              } catch (error) {
                logError(error.stack);

                result.snapshot.status = 'error';
                result.snapshot.error = {
                  name: error.name,
                  message: error.message
                };
              } finally {
                if (proxyThroughWaybackMachine) {
                  proxyServer.close();
                  proxyPorts.delete(proxyPort);
                }
              }

              log('********* Writing manifest file...');

              await fs.writeFile(manifestFilePath, JSON.stringify({
                entry,
                url,
                result
              }));
            }, { unsafeCleanup: true });
          }
        }
      });
    }
  }, { unsafeCleanup: true });
})();
