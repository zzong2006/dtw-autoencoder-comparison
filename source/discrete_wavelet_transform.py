"""
    차원을 압축할때 detail한 부분을 자르지 않고 압축 level을 줄여나가는 방식으로 접근하는 version
"""

import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing


class DWT:
    def __init__(self, wavelet_type='haar'):
        self.windowData = []
        self.waveletType = wavelet_type

    def get_dwt_max_level(self, dataset):
        """
        Return the maximum DWT level this time_series dataset can reach.
        :param dataset: 시계열 데이터가 존재하는 데이터셋 (DataFrame 또는 ndarray)
        :return:
        """
        total, length = dataset.shape
        max_level = pywt.dwt_max_level(data_len=length, filter_len=pywt.Wavelet(self.waveletType).dec_len)

        return max_level

    def do_dim_reduction(self, dataset, dim=128, addition_info=True, lvl=5):
        total, length = dataset.shape

        wv_coef = pywt.wavedec(self.windowData[0], self.waveletType, level=lvl)
        print('The number of approximation coefficients of DWT : {}'.format(
            len(wv_coef[0])
        ))

        # coeffSlices는 original data로 복구할려면 필요하다.
        self.reduced_window = []

        self.recoveredWindow = []

        # detailed coefficients의 min, max, variance를 첨가한다.
        if addition_info:
            for z in range(total):
                wv_coef = pywt.wavedec(self.normalizedData[z], self.waveletType, level=lvl)

                # self.reducedWindow.append(coeffs_haar[0].tolist())
                self.reduced_window.append([])

                for w in range(0, len(wv_coef)):
                    self.reduced_window[z] = np.append(self.reduced_window[z], np.max(wv_coef[w]))
                    self.reduced_window[z] = np.append(self.reduced_window[z], np.min(wv_coef[w]))
                    self.reduced_window[z] = np.append(self.reduced_window[z], np.std(wv_coef[w]))
                    self.reduced_window[z] = np.append(self.reduced_window[z], np.mean(wv_coef[w]))
                self.num_elements = len(self.reduced_window[z])

        else:
            for z in range(total):
                wv_coef = pywt.wavedec(self.normalizedData[z], self.waveletType, level=lvl)

                # for i in range(0, len(coeffs_haar)):
                #     modified_coeffs_haar01[i] = pywt.threshold(coeffs_haar[i], 0,'hard')

                self.reduced_window.append(wv_coef[0])
                self.num_elements = len(wv_coef[0])

        self.reduced_window = np.reshape(self.reduced_window, (-1, self.num_elements))
        # self.reducedWindow = preprocessing.scale(self.reducedWindow)

        print(self.reduced_window[0])
        print('We reduce the dimension of window from ' + str(self.length) + ' to ' + str(self.num_elements))
        print('Make ' + str(self.total) + ' window data')

        # print reduced dimensional type of signal
        # w01 = plt.figure(); plt.plot(self.windowData[4]);
        # w02 = plt.figure(); plt.plot(self.reducedWindow[4]);
        # plt.close(w01); plt.close(w02);

    # 압축된 윈도우를 얼마나 줄였는지 보여준다.
    # v2 는 복구하는 버전을 생략했음 (어차피 window 원본을 사용할 것이기 때문에). 시간이 있으면 만들어 봐도 좋음.
    def show_compress_precision(self, num_window=20, withReducedData=0):
        pass
        # for z in range(num_window):
        #     # elementsForRestore =  copy.copy(self.reducedWindow[z])
        #     # elementsForRestore += np.zeros(self.numRemainedWindow - self.num_elements).tolist()
        #     # from_many = pywt.array_to_coeffs(elementsForRestore, self.coeffSlices[z], 'wavedec')
        #     reconed_coeffs_haar01 = self.recoveredWindow[z]
        #
        #     print('#' + str(z + 1) + '#')
        #     if withReducedData :
        #         print('Reduced data : ' + str(self.reducedWindow[z]))
        #     print('Average of original & Reconstructed data')
        #     print(str(np.mean(self.windowData[z])) + ' & ' + str(np.mean(reconed_coeffs_haar01)))
        #     print('Difference Average : "' + str(np.mean(self.windowData[z] - reconed_coeffs_haar01)) + '"')

    def get_reduced_data(self):
        return self.reduced_window
