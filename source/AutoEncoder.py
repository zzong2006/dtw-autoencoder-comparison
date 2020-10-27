import tensorflow as tf


class AutoEncoderModule(tf.Module):
    def __init__(self, size_list, name="AE"):
        super().__init__(name=name)
        self.encoder = Encoder(size_list, name + "_encoder")
        self.decoder = Decoder(size_list, name + "_decoder")

    def __call__(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output

    def train(self, model, input_x, target_y, learning_rate, loss_fn):
        with tf.GradientTape() as t:
            loss = loss_fn(target_y, model(input_x))

        dvs = t.gradient(loss, self.trainable_variables)
        for v, d in zip(self.trainable_variables, dvs):
            v.assign_sub(learning_rate * d)


class Encoder(tf.Module):
    def __init__(self, size_list, name=None):
        super().__init__(name=name)
        self.weights = []
        self.bias = []
        for i in range(len(size_list) - 1):
            self.weights.append(tf.Variable(tf.random.normal(shape=[size_list[i], size_list[i + 1]]),
                                            name="encoder_weights_" + str(i)))
            self.bias.append(tf.Variable(tf.zeros([1, 1]), name="encoder_bias_" + str(i)))

    def __call__(self, x):
        for i in range(len(self.weights)):
            if i == 0:
                output = tf.matmul(x, self.weights[i]) + self.bias[i]
            else:
                output = tf.matmul(output, self.weights[i]) + self.bias[i]
            output = tf.nn.relu(output)
        return output


class Decoder(tf.Module):
    def __init__(self, size_list, name=None):
        super().__init__(name=name)
        r_size_list = size_list[::-1]
        self.weights = []
        self.bias = []
        for i in range(len(size_list) - 1):
            self.weights.append(tf.Variable(tf.random.normal(shape=[r_size_list[i], r_size_list[i + 1]]),
                                            name="encoder_weights_" + str(i)))
            self.bias.append(tf.Variable(tf.zeros([1, 1]), name="encoder_bias_" + str(i)))

    def __call__(self, x):
        for i in range(len(self.weights)):
            if i == 0:
                output = tf.matmul(x, self.weights[i]) + self.bias[i]
            else:
                output = tf.matmul(output, self.weights[i]) + self.bias[i]
            if i != len(self.weights) - 1:
                output = tf.nn.relu(output)
            else:
                output = tf.sigmoid(output)
        return output
