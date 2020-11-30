import unittest
import numpy as np
import tensorflow as tf
import ts_datset


class TSDatasetTest(unittest.TestCase):
    def test_slide_one_step(self):
        seq_length = 5  # 시퀀스 길이
        num_train = 32  # 학습 데이터 개수
        seqs = tf.random.normal(shape=[num_train, seq_length, 1])
        [enc_inp, dec_inp], target = ts_datset.slide_one_step(seqs)

        tf.assert_equal(enc_inp[1, :-1, :], dec_inp[1, 1:, :])
        self.assertEqual(dec_inp.shape, [num_train, seq_length, 1])


if __name__ == '__main__':
    unittest.main()
