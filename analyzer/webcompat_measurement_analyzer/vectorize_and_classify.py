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

_concurrency = 8 # multiprocessing.cpu_count()

import os

os.environ['OMP_NUM_THREADS'] = str(_concurrency)
os.environ['MKL_NUM_THREADS'] = str(_concurrency)
os.environ['OPENBLAS_NUM_THREADS'] = str(_concurrency)
os.environ['BLIS_NUM_THREADS'] = str(_concurrency)

import collections
import functools
import itertools
import json
import multiprocessing
import random

# import dateutil.parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import numpy
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer, Normalizer, StandardScaler
from sklearn.utils import parallel_backend

from ._lib.paths import (
    classifier_models_dir_path,
    data_dir_path,
    differents_file_path,
    # entries_dir_path,
    get_sharded_path,
    labels_file_path,
    metrics_dir_path,
    splits_dir_path,
    subgraphs_dir_path,
    summaries_dir_path,
    vectorizer_models_dir_path,
)

_use_doc2vec = False

_chunk_size = 128
_n_splits = 2

# _too_old_threshold = dateutil.parser.parse('2018-01-01 00:00:00 +0000')

_Sample = collections.namedtuple('_Sample', ['category', 'commit', 'url_count', 'summary', 'metrics_vector'])

def _read_sample(category_label, category, commit, url_count):
    # if category == 'unbroken':
        # category_entries_dir_path = os.path.join(entries_dir_path, category)
        # entry_file_path = get_sharded_path(category_entries_dir_path, f'{commit}.json')
        # with open(entry_file_path, 'r', encoding='utf-8') as entry_file:
            # entry = json.load(entry_file)

        # entry_date = dateutil.parser.parse(entry['date'])
        # if entry_date < _too_old_threshold:
            # return None

    category_summaries_dir_path = os.path.join(summaries_dir_path, category)
    commit_summaries_dir_path = get_sharded_path(category_summaries_dir_path, commit)
    summary_file_path = os.path.join(commit_summaries_dir_path, f'{url_count}.json')
    if not os.path.exists(summary_file_path):
        return None

    with open(summary_file_path, 'r', encoding='utf-8') as summary_file:
        summary = json.load(summary_file)

    category_metrics_dir_path = os.path.join(metrics_dir_path, category)
    commit_metrics_dir_path = get_sharded_path(category_metrics_dir_path, commit)
    metrics_file_path = os.path.join(commit_metrics_dir_path, f'{url_count}.json')
    if not os.path.exists(metrics_file_path):
        return None

    with open(metrics_file_path, 'r', encoding='utf-8') as metrics_file:
        metrics = json.load(metrics_file)

    # if metrics['subgraph is script block'] == 0:
    #     return None

    metrics_entries = sorted(
        ((key, value) for key, value in metrics.items()), key=lambda item: item[0]
    )
    metrics_vector = [value for key, value in metrics_entries] # if ' is script block' not in key]
    # print(','.join(key for key, value in metrics_entries if ' is script block' not in key))
    # with open('metrics.csv', 'a', encoding='utf-8') as metrics_csv_file:
        # print(','.join([category, commit, str(url_count), *(str(x) for x in metrics_vector)]), file=metrics_csv_file)

    return (category_label, _Sample(category, commit, url_count, summary, metrics_vector))

_considered_keys = {
        'subgraph total number of bytes transferred',
        'subgraph total number of DOM nodes created',
        'subgraph number of text nodes created',
        'subgraph number of nodes',
        'subgraph number of failed DOM queries',
        'subgraph number of event registrations',
        'subgraph number of edges',
        'subgraph number of ScriptClassic bytes transferred',
        'subgraph number of DOM nodes touched',
        'subgraph number of DOM nodes created and inserted by the same script',
        'subgraph net number of DOM nodes created',
        'delta total number of network requests started',
        'delta total number of network requests completed',
        'delta total number of bytes transferred',
        'delta total number of DOM nodes created',
        'delta number of text nodes created',
        'delta number of scripts',
        'delta number of nodes',
        'delta number of failed DOM queries',
        'delta number of event registrations',
        'delta number of edges',
        'delta number of ScriptClassic bytes transferred',
        'delta number of Image network requests started',
        'delta number of Image network requests completed',
        'delta number of Image bytes transferred',
        'delta number of DOM nodes touched',
        'delta number of DOM nodes created and inserted by the same script',
        'delta net number of DOM nodes created',
}

def _read_legacy_sample(category_label, category, snapshot_id, variant):
    category_summaries_dir_path = os.path.join(data_dir_path, 'summaries.old', category)
    summary_file_path = os.path.join(category_summaries_dir_path, f'page_graph_web_compat_snapshots_{category}_{snapshot_id}_{variant}.json')
    with open(summary_file_path, 'r', encoding='utf-8') as summary_file:
        summary = json.load(summary_file)

    category_metrics_dir_path = os.path.join(data_dir_path, 'metrics.old', category)
    metrics_file_path = os.path.join(category_metrics_dir_path, f'page_graph_web_compat_snapshots_{category}_{snapshot_id}_{variant}.json')
    with open(metrics_file_path, 'r', encoding='utf-8') as metrics_file:
        metrics = json.load(metrics_file)

    metrics_entries = sorted(
        ((key, value) for key, value in metrics.items()), key=lambda item: item[0]
    )
    metrics_vector = [value for key, value in metrics_entries]

    return (category_label, _Sample(category, snapshot_id, -1, summary, metrics_vector))

def _infer_summary_vector_from_sample(model, sample):
    return model.infer_vector(sample.summary)

def _evaluate_predictions(categories, test_labels, predictions):
    print(f'\nAccuracy: {accuracy_score(test_labels, predictions)}\n')
    print(classification_report(test_labels, predictions, target_names=categories))
    print()

def _flatten_vector(vector):
    if hasattr(vector, 'toarray'):
        vector = vector.toarray()[0]
    return list(vector)

def run():
    print('* Indexing categories...')

    categories = os.listdir(subgraphs_dir_path)
    categories.sort()

    # print('* Reading commit lists...')

    # category_commits = {}

    # for category in categories:
        # print(f'** Reading commit list for category {category}...')

        # category_commit_list_file_path = os.path.join(data_dir_path, f'{category}-commits.txt')
        # with open(category_commit_list_file_path, 'r', encoding='utf-8') as category_commit_list_file:
            # category_commits[category] = set(filter(lambda line: line, map(lambda line: line.strip(), category_commit_list_file.readlines())))

    # print('* Reading differents list...')

    # with open(differents_file_path, 'r', encoding='utf-8') as differents_file:
    #     differents = set(
    #         map(lambda line: line.split('/')[3],
    #             filter(lambda line: line,
    #                    map(lambda line: line.strip(),
    #                        differents_file.readlines()))))

    print('* Indexing subgraphs...')

    subgraphs = []

    for category_label, category in enumerate(categories):
        category_subgraphs_dir_path = os.path.join(subgraphs_dir_path, category)
        # category_subgraph_count = 0
        with os.scandir(category_subgraphs_dir_path) as shard_subgraphs_dirs:
            for shard_subgraphs_dir in shard_subgraphs_dirs:
                with os.scandir(shard_subgraphs_dir) as commit_subgraphs_dirs:
                    for commit_subgraphs_dir in commit_subgraphs_dirs:
                        commit = commit_subgraphs_dir.name
                        # if commit not in differents:
                        #     continue
                        # if commit in category_commits[category]:
                            # category_commits[category].remove(commit)
                        # else:
                            # continue
                        # if category_subgraph_count >= 1239:
                            # continue
                        # category_subgraph_count += 1
                        with os.scandir(commit_subgraphs_dir) as url_subgraph_files:
                            for url_subgraph_file in url_subgraph_files:
                                if '.swp' in url_subgraph_file.name:
                                    continue
                                url_count = int(os.path.splitext(url_subgraph_file.name)[0])
                                subgraphs.append((category_label, category, commit, url_count))

    # print(len(category_commits['broken']))
    # print(len(category_commits['unbroken']))

    random.shuffle(subgraphs)

    with multiprocessing.Pool(_concurrency) as pool:
        print(f'* Reading samples for {len(subgraphs)} subgraph(s)...')

        # labels, samples = zip(*filter(lambda row: row is not None, pool.starmap(_read_sample, subgraphs, chunksize=_chunk_size)))
        # labels, samples = zip(*pool.starmap(_read_sample, subgraphs, chunksize=_chunk_size))
        # labels = list(labels)
        # samples = list(samples)

        inputs = filter(lambda row: row is not None, pool.starmap(_read_sample, subgraphs, chunksize=_chunk_size))
        label_samples = {}
        for label, sample in inputs:
            if label in label_samples:
                label_samples[label].append(sample)
            else:
                label_samples[label] = [sample]

        min_label_sample_count = min(len(samples_group) for samples_group in label_samples.values())

        labels = []
        samples = []
        for label, samples_group in label_samples.items():
            min_label_sample_count = len(samples_group)
            labels.extend([label] * min_label_sample_count)
            samples.extend(samples_group[0:min_label_sample_count])

        # labels = []
        # samples = []

        # print('* Rounding out with legacy data...')

        # category_label = 0
        # for category, missing_commits in category_commits.items():
            # with open(os.path.join(data_dir_path, f'{category}-commit-map.json'), 'r', encoding='utf-8') as category_commit_map_file:
                # category_commit_map = json.load(category_commit_map_file)

            # variant = 'child' if category == 'broken' else 'parent'
            # legacy_subgraphs = [(category_label, category, category_commit_map[missing_commit], variant) for missing_commit in missing_commits]
            # legacy_labels, legacy_samples = zip(*pool.starmap(_read_legacy_sample, legacy_subgraphs, chunksize=_chunk_size))
            # labels.extend(list(legacy_labels))
            # samples.extend(list(legacy_samples))
            # category_label += 1

        print(f'** Collected {len(samples)} sample(s)')

        print(f'* Performing {_n_splits}-way split evaluation...')

        print('** Generating and evaluating splits...')

        all_test_labels = []
        all_test_samples = []
        all_predictions = []

        kf = StratifiedKFold(n_splits=_n_splits, shuffle=True)

        for split_index, split_chunk in enumerate(kf.split(samples, labels)):
            split_count = split_index + 1

            print(f'*** Evaluating split #{split_count}...')

            print('**** Extracting split data...')

            train_indices, test_indices = split_chunk
            random.shuffle(train_indices)
            random.shuffle(test_indices)

            train_labels = [labels[train_index] for train_index in train_indices]
            test_labels = [labels[test_index] for test_index in test_indices]

            train_samples = [samples[train_index] for train_index in train_indices]
            test_samples = [samples[test_index] for test_index in test_indices]


            train_label_samples = {}
            for train_label, train_sample in zip(train_labels, train_samples):
                if train_label in train_label_samples:
                    train_label_samples[train_label].append(train_sample)
                else:
                    train_label_samples[train_label] = [train_sample]

            min_train_label_sample_count = min(
                len(train_samples_group)
                    for train_samples_group in train_label_samples.values())

            train_labels = []
            train_samples = []
            for train_label, train_samples_group in train_label_samples.items():
                train_labels.extend([train_label] * min_train_label_sample_count)
                train_samples.extend(train_samples_group[0:min_train_label_sample_count])


            train_vectors = [sample.metrics_vector for sample in train_samples]
            test_vectors = [sample.metrics_vector for sample in test_samples]

            print('**** Scaling metrics vectors...')

            with parallel_backend('loky', n_jobs=_concurrency):
                scaler = StandardScaler()
                scaler.fit(train_vectors)
                train_vectors = scaler.transform(train_vectors)
                test_vectors = scaler.transform(test_vectors)

                # normalizer = Normalizer()
                # normalizer.fit(train_vectors)
                # train_vectors = normalizer.transform(train_vectors)
                # test_vectors = normalizer.transform(test_vectors)

                # discretizer = KBinsDiscretizer(n_bins=3)
                # discretizer.fit(train_vectors)
                # train_vectors = discretizer.transform(train_vectors)
                # test_vectors = discretizer.transform(test_vectors)

                train_vectors = [_flatten_vector(vector) for vector in train_vectors]
                test_vectors = [_flatten_vector(vector) for vector in test_vectors]

            if _use_doc2vec:
                print('**** Optimizing Doc2Vec model...')

                model = Doc2Vec(
                    [
                        TaggedDocument(words=sample.summary, tags=[sample_index])
                        for sample_index, sample in enumerate(train_samples)
                    ],
                    vector_size=128,
                    window=0,
                    min_count=5,
                    dm=0,
                    sample=0.0001,
                    workers=_concurrency,
                    epochs=10,
                    alpha=0.025,
                )

                print('***** Saving model...')

                os.makedirs(vectorizer_models_dir_path, exist_ok=True)
                vectorizer_model_file_path = os.path.join(vectorizer_models_dir_path, f'{split_count}.dat')
                model.save(vectorizer_model_file_path)

                print('**** Inferring summary vectors...')

                summary_vectors = pool.imap(
                    functools.partial(_infer_summary_vector_from_sample, model),
                    itertools.chain(train_samples, test_samples),
                    chunksize=_chunk_size,
                )

                train_vectors = [metrics_vector + _flatten_vector(summary_vector) for metrics_vector, summary_vector in zip(train_vectors, summary_vectors)]
                test_vectors = [metrics_vector + _flatten_vector(summary_vector) for metrics_vector, summary_vector in zip(test_vectors, summary_vectors)]

            print('**** Training classifier...')

            with parallel_backend('loky', n_jobs=_concurrency):
                def create_model():
                    keras_model = Sequential()
                    keras_model.add(Dense(3, kernel_initializer='uniform', activation='relu', input_dim=len(train_vectors[0])))
                    keras_model.add(Dropout(rate=0.1))
                    keras_model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
                    keras_model.add(Dropout(rate=0.1))
                    keras_model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
                    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    return keras_model

                classifier = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=True)

                # classifier = MLPClassifier(max_iter=10000, verbose=True)

                # classifier = MLPClassifier(max_iter=10000, activation='tanh', alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam', verbose=True)
                # classifier = MLPClassifier(max_iter=10000, activation='tanh', alpha=0.0001, hidden_layer_sizes=(50, 100, 50), solver='sgd', verbose=True)
                # classifier = MLPClassifier(solver='lbfgs', max_iter=1000, verbose=True)
                # classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
                # classifier = AdaBoostClassifier()
                # classifier = LogisticRegression(solver='newton-cg', max_iter=1000)

                # parameter_space = {
                #     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                #     'activation': ['tanh', 'relu'],
                #     'solver': ['sgd', 'adam'],
                #     'alpha': [0.0001, 0.05],
                #     'learning_rate': ['constant','adaptive'],
                # }
                # classifier = GridSearchCV(classifier, parameter_space, n_jobs=1, cv=3, verbose=3)
                classifier.fit(numpy.array(train_vectors), numpy.array(train_labels))

                #print('Best parameters found:\n', classifier.best_params_)

                #means = classifier.cv_results_['mean_test_score']
                #stds = classifier.cv_results_['std_test_score']
                #for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
                #        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            # print('***** Saving model...')

            # os.makedirs(classifier_models_dir_path, exist_ok=True)
            # classifier_model_file_path = os.path.join(
                # classifier_models_dir_path, f'{split_count}.dat'
            # )
            # joblib.dump(classifier, classifier_model_file_path)

            print('**** Predicting labels...')

            predictions = classifier.predict(numpy.array(test_vectors))

            all_test_labels.extend(test_labels)
            all_test_samples.extend(test_samples)
            all_predictions.extend(predictions)

            print('**** Evaluating predictions...')

            _evaluate_predictions(categories, test_labels, predictions)

            # for test_sample, test_label, prediction in zip(test_samples, test_labels, predictions):
            #     prediction = int(prediction)
            #     if test_label != prediction:
            #         print(f'label:{categories[test_label]} prediction:{categories[prediction]} category:{test_sample.category} commit:{test_sample.commit} url:{test_sample.url_count}')
            #     else:
            #         print(f'=== label:{categories[test_label]} prediction:{categories[prediction]} category:{test_sample.category} commit:{test_sample.commit} url:{test_sample.url_count}')

    print('** Evaluating all predictions across splits...')

    _evaluate_predictions(categories, all_test_labels, all_predictions)

    print('** Saving labels...')

    with open(labels_file_path, 'w', encoding='utf-8') as labels_file:
        for test_sample, prediction in zip(all_test_samples, all_predictions):
            json.dump({
                'category': test_sample.category,
                'commit': test_sample.commit,
                'urlIndex': test_sample.url_count,
                'prediction': categories[int(prediction)]
            }, labels_file)
            labels_file.write('\n')

if __name__ == '__main__':
    run()
