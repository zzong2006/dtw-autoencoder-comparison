import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

numberOfWindow = 500

MAX_NUM = 255  # 단일 데이터가 가질 수 있는 최대값
MIN_NUM = 0  # 단일 데이터가 가질 수 있는 최소값
VAR_SIGMA = 40  # 변위로 사용할 정규분포의 표준편차


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_data_to_file(fileName='data.csv', GEN_DATA_PER_WINDOW=1000, GEN_WINDOW_QUANTITY=1000):
    # 스트림 데이터 생성 시작
    out_file = open(fileName, 'w')

    means = np.random.uniform(MIN_NUM, MAX_NUM, GEN_WINDOW_QUANTITY)
    stream_data = []
    for VAR_MEAN in means:
        stream_data = np.append(stream_data,
                                get_truncated_normal(VAR_MEAN, VAR_SIGMA, MIN_NUM, MAX_NUM).rvs(GEN_DATA_PER_WINDOW))

    # 데이터 GEN_DATA_QUANTITY 개 만큼 가우시안 분포로 생성한뒤, convert these stream data into Integer.
    # stream_data = get_truncated_normal(VAR_MEAN, VAR_SIGMA, MIN_NUM, MAX_NUM).rvs(GEN_DATA_QUANTITY)
    # stream_data = np.random.normal(0, 4, GEN_DATA_QUANTITY)
    stream_data = list(map(float, stream_data))

    numberOfData = GEN_DATA_PER_WINDOW * GEN_WINDOW_QUANTITY

    # 데이터를 1개씩 추가로 생성해가며(변위를 주어가며) 파일에 출력
    for x in range(numberOfData):
        output = str(stream_data[x])
        out_file.write(output + '\n')
    out_file.close()

    fig = plt.subplot()
    '''
    Density: If True, the first element of the return tuple will be the counts normalized to form a probability density,
    i.e., the area (or integral) under the histogram will sum to 1. 
    '''
    fig.hist(stream_data, 'auto', density=True, histtype='stepfilled', facecolor='g', alpha=0.75)
    plt.show()


if __name__ == "__main__":
    generate_data_to_file(fileName='test_data.csv', GEN_WINDOW_QUANTITY=numberOfWindow)
    generate_data_to_file(fileName='data.csv', GEN_WINDOW_QUANTITY=numberOfWindow)
