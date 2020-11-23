import tensorflow as tf


def slide_one_step(seq):
    """
    시퀀스 입력 데이터를 Seq2Seq 학습을 위한 데이터로 변환해주는 함수. tf.data.Dataset 에서 map 적용한다.
    :param seq: sequence 학습 데이터 [batch_size, time_step, features] 의 크기를 가짐
    :return: [인코더, 디코더 입력], 타겟 시퀀스 데이터
    """

    encoder_inp = seq
    # 디코더 입력의 시작은 0
    paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
    decoder_inp = tf.pad(seq[:, :-1, :], paddings, "CONSTANT")
    target_data = seq
    return [encoder_inp, decoder_inp], target_data  # input, output

def add_decoder_input(seq):

    encoder_inp = seq
