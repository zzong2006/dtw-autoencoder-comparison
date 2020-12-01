# DWT를 이용한 clustering
import json
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import matplotlib.cm as cm

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn import metrics
from load_dataset import DatasetLoader
from autoencoder import AutoEncoder


class AEClusteringDriver:
    def __init__(self, dataset_loader: DatasetLoader, params):
        self.dl = dataset_loader
        self.infos = dataset_loader.data_info
        self.params = params
        self.silhouette_scores = dict()
        self.adjusted_ri_scores = dict()

    def clustering(self, scores: dict):
        for network_type in ['normal', 'var']:
            for d in [self.silhouette_scores, self.adjusted_ri_scores]:
                if network_type not in d:
                    d[network_type] = dict()

            for latent_units in scores.keys():
                if network_type == 'rnn':
                    ae_model = None
                else:
                    hidden_units = (self.infos['length'] + latent_units) // 2
                    ae_model = AutoEncoder([self.infos['length'], hidden_units, latent_units], network_type=network_type)

                ae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
                                 loss='mae')

                ae_model.fit(x=self.dl.get_numpy_dataset(data_type='train', label=False),
                             y=self.dl.get_numpy_dataset(data_type='train', label=True),
                             epochs=self.params['epochs'],
                             batch_size=self.params['batch_size'],
                             shuffle=True)

                reduced_outputs = ae_model.encoding(tf.convert_to_tensor(dl_sep.get_numpy_dataset(data_type='test',label=False)))
                sil_avg, rand_score = clustering(dataset=reduced_outputs,
                                                 num_of_cluster=self.infos["number_of_classes"],
                                                 ground_truths=np.reshape(
                                                     self.dl.get_numpy_dataset(data_type='test', label=True), [-1]))

                if latent_units not in self.silhouette_scores:
                    self.silhouette_scores[network_type][latent_units] = sil_avg
                    self.adjusted_ri_scores[network_type][latent_units] = rand_score

        return self.silhouette_scores, self.adjusted_ri_scores


class DWTClusteringDriver:
    def __init__(self, dataset_loader: DatasetLoader, wavelet_type):
        self.dl = dataset_loader
        self.wavelet = wavelet_type
        self.infos = dataset_loader.data_info
        self.silhouette_scores = dict()
        self.adjusted_ri_scores = dict()

    def clustering(self, data_type):
        """
            measure clustering performance of compressed timeseries data by DTW
            performance metrics : silhouette scores & adjusted rand index
        :return:
        """
        reduced_outputs = pywt.wavedec(self.dl.get_numpy_dataset(data_type=data_type, label=False),
                                       wavelet=self.wavelet)
        total_level = len(reduced_outputs)

        for k in range(total_level):
            num_samples, length = reduced_outputs[k].shape
            sil_avg, rand_score = clustering(dataset=reduced_outputs[k],
                                             num_of_cluster=self.infos["number_of_classes"],
                                             ground_truths=np.reshape(
                                                 self.dl.get_numpy_dataset(data_type=data_type, label=True), [-1]))
            if length not in self.silhouette_scores:
                self.silhouette_scores[length] = sil_avg
                self.adjusted_ri_scores[length] = rand_score
        return self.silhouette_scores, self.adjusted_ri_scores


def clustering(dataset, num_of_cluster, ground_truths):
    # n_clusters : The number of clusters to form as well as the number of centroids to generate.
    # n_init : Number of time the k-means algorithm will be run with different centroid seeds.
    # The final results will be the best output of n_init consecutive runs in terms of inertia.
    kmeans_model = KMeans(init='k-means++', n_clusters=num_of_cluster, verbose=False)

    labels = kmeans_model.fit_predict(dataset)
    silhouette_avg = metrics.silhouette_score(dataset, labels)
    rand_score = metrics.adjusted_rand_score(
        labels_true=ground_truths,
        labels_pred=labels)

    return silhouette_avg, rand_score


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        json_data = json.load(f)

        for data_dictionary in json_data['input_data']:
            dl_sep = DatasetLoader(data_dictionary, normalize='sep')
            dwt_driver = DWTClusteringDriver(dl_sep, json_data['wavelet_type'])
            sil, adj = dwt_driver.clustering(data_type='test')

            dl_same = DatasetLoader(data_dictionary, normalize='same')
            ae_driver = AEClusteringDriver(dl_same, json_data['AutoEncoder'])
            ae_sil, ae_ajd = ae_driver.clustering(scores=sil)

            # print result
            print('Dataset : {}'.format(data_dictionary['name']))
            print('Silhouette Score Comparison')
            for w in sil.keys():
                print('# Dimension {} #'.format(w))
                print('{} wavelet: {:.5f}'.format(json_data['wavelet_type'], sil[w]))
                for net_type in ae_sil.keys():
                    print('{} autoEncoder: {:.5f}'.format(net_type, ae_sil[net_type][w]), end=" | ")
                print()

            print()

            print('Adjusted Rand Index Comparison')
            for w in adj.keys():
                print('# Dimension {} #'.format(w))
                print('{} wavelet: {:.5f}'.format(json_data['wavelet_type'], adj[w]))
                for net_type in ae_ajd.keys():
                    print('{} autoEncoder: {:.5f}'.format(net_type, ae_ajd[net_type][w]), end=" | ")
                print()

