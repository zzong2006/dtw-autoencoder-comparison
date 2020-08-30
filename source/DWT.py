# approximation 의 coefficient 만 가지고 feature 를 뽑아냄.

import pywt
import numpy as np
import copy
import matplotlib.pyplot as plt
import random

# window 데이터 준비

class DWT:
    def __init__(self, dataFile = 'real_splitted_data.csv', window_size = 1000, wvType ='haar'):
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

        self.max_level = pywt.dwt_max_level(data_len=self.windowSize, filter_len=pywt.Wavelet(self.waveletType).dec_len)
        print('Max DWT level we can reach : {} '.format(self.max_level))

    def reduce_data(self, num_window = 100, object_dimension = 128, lvl = 5):
        if num_window is 'all' :
            num_window = self.totalWindow

        coeffs_haar = pywt.wavedec(self.windowData[0], self.waveletType)
        print('The number of approximation coefficients of DWT : {}'.format(
            len(coeffs_haar[0])
        ))

        # coeffSlices는 original data로 복구할려면 필요하다.
        self.reducedWindow = []
        self.coeffSlices = []
        self.recoveredWindow = []

        for z in range(num_window) :
            coeffs_haar = pywt.wavedec(self.windowData[z], self.waveletType)

            modified_coeffs_haar01 = copy.copy(coeffs_haar)

            max_level = pywt.dwt_max_level(data_len= self.windowSize , filter_len = pywt.Wavelet(self.waveletType).dec_len)

            # for i in range(0, len(coeffs_haar)):
            #     modified_coeffs_haar01[i] = pywt.threshold(coeffs_haar[i], 0,'hard')

            # print('Max DWT level we can reach : {} '.format(max_level))

            single = []
            self.num_elements = 0

            # If you want to change the dimension of data, you should change the value of max_level.
            for i in range(0, int(max_level/2+5)):
                self.num_elements += len(modified_coeffs_haar01[i])
            from_single, coeff_slice  = pywt.coeffs_to_array(modified_coeffs_haar01)

            for i in range(0, self.num_elements):
                single.append(from_single[i])

            self.reducedWindow.append(single)

            self.coeffSlices.append(coeff_slice)
            self.numRemainedWindow = len(from_single)

            # 압축된 window를 다시 복구한다.
            elementsForRestore = copy.copy(single)
            elementsForRestore += np.zeros(self.numRemainedWindow - self.num_elements).tolist()
            from_many = pywt.array_to_coeffs(elementsForRestore, coeff_slice, 'wavedec')
            self.recoveredWindow.append(pywt.waverec(from_many, self.waveletType))

        self.reducedWindow = np.reshape(self.reducedWindow, (-1, self.num_elements))
        print('We reduce the dimension of window from ' + str(self.windowSize) + ' to ' + str(self.num_elements))

    # 압축된 윈도우를 얼마나 줄였는지 보여준다.
    def showComprassPrecision(self, num_window = 20, withReducedData = 0):
        for z in range(num_window):
            # elementsForRestore =  copy.copy(self.reducedWindow[z])
            # elementsForRestore += np.zeros(self.numRemainedWindow - self.num_elements).tolist()
            # from_many = pywt.array_to_coeffs(elementsForRestore, self.coeffSlices[z], 'wavedec')
            reconed_coeffs_haar01 = self.recoveredWindow[z]

            print('#' + str(z + 1) + '#')
            if withReducedData :
                print('Reduced data : ' + str(self.reducedWindow[z]))
            print('Average of original & Reconstructed data')
            print(str(np.mean(self.windowData[z])) + ' & ' + str(np.mean(reconed_coeffs_haar01)))
            print('Difference Average : "' + str(np.mean(self.windowData[z] - reconed_coeffs_haar01)) + '"')

    def getReducedDatas(self):
        return self.reducedWindow