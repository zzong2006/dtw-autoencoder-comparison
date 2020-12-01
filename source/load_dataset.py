import pandas as pd
import json


class DatasetLoader:
    def __init__(self, file_dict, normalize, print_head=True):
        """

        :param file_dict: json config 데이터
        :param print_head: pandas train data load 후 head() 출력 할 것인지 여부
        :param normalize: 시계열 데이터를 z-score normalize 할 것인지 여부.

        """
        self.data_info = file_dict

        if file_dict['header'] == 'None':
            header = None
        else:
            header = file_dict['header']

        try:
            self.train_x = pd.read_csv(file_dict['train_path'], sep=file_dict['sep'], header=header, engine='python')
            self.test_x = pd.read_csv(file_dict['test_path'], sep=file_dict['sep'], header=header, engine='python')
            if file_dict['class_index'] >= 0:
                self.train_y = self.train_x.iloc[:, [file_dict['class_index']]]
                self.train_x = self.train_x.drop([file_dict['class_index']], axis=1)

                self.test_y = self.test_x.iloc[:, [file_dict['class_index']]]
                self.test_x = self.test_x.drop([file_dict['class_index']], axis=1)
        except Exception as e:
            print(e)
        else:
            if print_head:
                print(self.train_x.head())

        # Normalization
        if normalize == 'sep':  # train과 test의 normalization을 다르게 수행
            self.train_x = (self.train_x - self.train_x.mean()) / self.train_x.std()
            self.test_x = (self.test_x - self.test_x.mean()) / self.test_x.std()
        elif normalize == 'same':  # train과 test의 normalization을 같게 수행 (for autoencoder)
            mean_val = self.train_x.mean()
            std_val = self.train_x.std()
            self.train_x = (self.train_x - mean_val) / std_val
            self.test_x = (self.test_x - mean_val) / std_val
        elif normalize is None:  # do not normalization
            pass
        else:
            raise Exception('Unknown Normalization type')

    def get_numpy_dataset(self, data_type, label):
        if label is False:
            if data_type == 'train':
                return self.train_x.values
            elif data_type == 'test':
                return self.test_x.values
            else:
                raise Exception('Unknown data type')
        else:
            if data_type == 'train':
                return self.train_y.values
            elif data_type == 'test':
                return self.test_y.values
            else:
                raise Exception('Unknown data type')

    def get_train_set(self):
        if self.train_y is not None:
            return self.train_x, self.train_y
        else:
            return self.train_x

    def get_test_set(self):
        return self.test_x, self.test_y


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        json_data = json.load(f)

        for data_dictionary in json_data['input_data']:
            dl = DatasetLoader(data_dictionary, normalize='sep')
