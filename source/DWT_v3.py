'''
   또다른 방법 (+ 데이터를 Normalize 하는 작업 포함한다.)
'''

import pywt
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler

# window 데이터 준비

class DWT:
    def __init__(self, dataFile = 'real_splitted_data.csv',
                 window_size = 1000, wvType ='haar'):
        self.windowData = []
        self.waveletType = wvType
        self.windowSize = window_size

        with open(dataFile, 'r') as reader:
            for i, line in enumerate(reader):
                self.windowData.append(line.split(','))

        # Normalization
        for line in self.windowData:
             line[:] = [float(x) for x in line]

        # you need to reshape you data to have a spatial dimension for Conv1d to make sense:
        self.windowData = np.reshape(self.windowData, (-1, window_size))

        # (200, 1000)
        print('Window Data shape : {}'.format(self.windowData.shape))
        self.totalWindow = self.windowData.shape[0]

        # Load 된 window data 를 Normalize 한다.
        # scaler = MinMaxScaler()
        # self.windowData = scaler.fit_transform(np.reshape(self.windowData, (-1, 1)))
        # self.windowData = np.reshape(self.windowData, (-1, self.windowSize))

        self.max_level = pywt.dwt_max_level(data_len=self.windowSize, filter_len=pywt.Wavelet(self.waveletType).dec_len)
        print('Max DWT level we can reach : {} '.format(self.max_level))

    def reduce_data(self, num_window = 20, object_dimension = 128, lvl = 5):
        if num_window is 'all' :
            num_window = self.totalWindow

        coeffs_haar = pywt.wavedec(self.windowData[0], self.waveletType, level=lvl)
        print('The number of approximation coefficients of DWT : {}'.format(
            len(coeffs_haar[0])
        ))

        # coeffSlices는 original data로 복구할려면 필요하다.
        self.reducedWindow = []
        self.coeffSlices = []
        self.recoveredWindow = []

        for z in range(num_window) :
            coeffs_haar = pywt.wavedec(self.windowData[z], self.waveletType, level= lvl)

            dap = []
            for i in coeffs_haar:
                dap += (z for z in i)

            dap.sort(reverse=True, key=abs)
            threshold = dap[object_dimension]

            modified_coeffs_haar01 = copy.copy(coeffs_haar)

            for i in range(0, len(coeffs_haar)):
                modified_coeffs_haar01[i] = pywt.threshold(coeffs_haar[i], abs(threshold), 'soft')

            dap = []
            for i in modified_coeffs_haar01:
                dap += (z for z in i)

            self.reducedWindow.append(dap)

        self.num_elements = len(dap)
        self.reducedWindow = np.reshape(self.reducedWindow, (-1, self.num_elements))

        print(self.reducedWindow[0])
        print('We reduce the dimension of window from ' + str(self.windowSize) + ' to ' + str(self.num_elements))
        print('Make '+ str(num_window) + ' window data')

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