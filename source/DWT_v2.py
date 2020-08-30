'''
    dimensional을 압축할때 detail한 부분을 자르지 않고 압축 level을 줄여나가는 방식으로 접근하는 version
'''

import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing

class DWT:
    def __init__(self, num_window = 100, dataFile = 'real_splitted_data.csv', withLabel = False
                 , wvType ='haar', normalized = True):
        self.windowData = []
        self.waveletType = wvType

        df = pd.read_csv(dataFile, header=None)
        print(df.head())
        self.windowSize = df.shape[1]
        self.totalWindow = df.shape[0]

        if withLabel :
            self.true_labels_ = df.iloc[:, 0].values
            x = df.drop(df.columns[0], axis=1)
            self.windowData = x.values
            self.windowSize -= 1
        else :
            self.windowData = df.values



        if num_window is 'all' :
            self.num_window = self.totalWindow
        else :
            self.num_window = num_window

        # Ex. (200, 1000)
        print('Window Data shape : {}'.format(self.windowData.shape))

        self.max_level = pywt.dwt_max_level(data_len=self.windowSize, filter_len=pywt.Wavelet(self.waveletType).dec_len)
        print('Max DWT level we can reach : "{}" '.format(self.max_level))

        if normalized :
            self.normalizedData = preprocessing.scale(self.windowData)
        else :
            self.normalizedData = self.windowData

    def reduce_data(self, object_dimension = 128, addition_info = True, lvl = 5):

        coeffs_haar = pywt.wavedec(self.windowData[0], self.waveletType, level=lvl)
        print('The number of approximation coefficients of DWT : {}'.format(
            len(coeffs_haar[0])
        ))

        # coeffSlices는 original data로 복구할려면 필요하다.
        self.reducedWindow = []
        self.coeffSlices = []
        self.recoveredWindow = []

        # detailed coefficients의 min, max, variance를 첨가한다.
        if addition_info :
            for z in range(self.num_window):
                coeffs_haar = pywt.wavedec(self.normalizedData[z], self.waveletType, level=lvl)

                # self.reducedWindow.append(coeffs_haar[0].tolist())
                self.reducedWindow.append([])

                for w in range(0,len(coeffs_haar)):
                    self.reducedWindow[z] = np.append(self.reducedWindow[z], np.max(coeffs_haar[w]))
                    self.reducedWindow[z] = np.append(self.reducedWindow[z], np.min(coeffs_haar[w]))
                    self.reducedWindow[z] = np.append(self.reducedWindow[z], np.std(coeffs_haar[w]))
                    self.reducedWindow[z] = np.append(self.reducedWindow[z], np.mean(coeffs_haar[w]))
                self.num_elements = len(self.reducedWindow[z])

        else :
            for z in range(self.num_window):
                coeffs_haar = pywt.wavedec(self.normalizedData[z], self.waveletType, level=lvl)

                # for i in range(0, len(coeffs_haar)):
                #     modified_coeffs_haar01[i] = pywt.threshold(coeffs_haar[i], 0,'hard')


                self.reducedWindow.append(coeffs_haar[0])
                self.num_elements = len(coeffs_haar[0])

        self.reducedWindow = np.reshape(self.reducedWindow, (-1, self.num_elements))
        # self.reducedWindow = preprocessing.scale(self.reducedWindow)

        print(self.reducedWindow[0])
        print('We reduce the dimension of window from ' + str(self.windowSize) + ' to ' + str(self.num_elements))
        print('Make '+ str(self.num_window) + ' window data')

        # print reduced dimensional type of signal
        # w01 = plt.figure(); plt.plot(self.windowData[4]);
        # w02 = plt.figure(); plt.plot(self.reducedWindow[4]);
        # plt.close(w01); plt.close(w02);

    # v2 는 복구하는 버전을 생략했음 (어차피 window 원본을 사용할 것이기 때문에). 시간이 있으면 만들어 봐도 좋음.
    def showComprassPrecision(self, num_window = 20, withReducedData = 0):
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

    def getReducedDatas(self):
        return self.reducedWindow