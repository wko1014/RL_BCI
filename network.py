import tensorflow as tf, numpy as np

tf.keras.backend.set_floatx("float64")


class ShallowNet(tf.keras.Model):
    tf.keras.backend.set_floatx("float64")
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.n_channels = 20
        self.f_freq = 100

        self.constraint = tf.keras.constraints.MaxNorm(2)

        self.conv1 = tf.keras.layers.Conv2D(40, (1, 13), padding="same", kernel_constraint=self.constraint)
        self.conv2 = tf.keras.layers.Conv2D(40, (self.n_channels, 1), padding="valid", kernel_constraint=self.constraint)

        self.pool = tf.keras.layers.AveragePooling2D((1, 35), (1, 7))

        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(2, kernel_constraint=self.constraint)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.math.square(x)
        # x = tf.math.reduce_mean(x, -2)
        x = self.pool(x)
        x = tf.math.log(x)

        x = self.flatten(x)
        x = self.dropout(x)
        y_hat = self.softmax(self.dense(x))
        return y_hat

class EEGNet(tf.keras.Model):
    tf.keras.backend.set_floatx("float32")
    def __init__(self):
        super(EEGNet, self).__init__()
        self.n_channels = 20
        self.f_freq = 100

        self.elu = tf.keras.layers.ELU()
        self.dropout = tf.keras.layers.Dropout(.5)
        self.avgpool1 = tf.keras.layers.AveragePooling2D((1, 4), (1, 4))
        self.avgpool2 = tf.keras.layers.AveragePooling2D((1, 8), (1, 8))

        self.conv1 = tf.keras.layers.Conv2D(8, (1, int(self.f_freq/2)), padding="same")
        self.conv2 = tf.keras.layers.DepthwiseConv2D((self.n_channels, 1), depth_multiplier=2)
        self.conv3 = tf.keras.layers.SeparableConv2D(16, (1, 16), padding="same")

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(2, kernel_constraint=tf.keras.constraints.MaxNorm(0.25))
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.elu(x)
        # x = tf.math.reduce_mean(x, -2)
        x = self.avgpool2(x)
        x = self.dropout(x)

        x = self.flatten(x)
        y_hat = self.softmax(self.dense(x))
        return y_hat

class DeepNet(tf.keras.Model):
    tf.keras.backend.set_floatx("float32")
    def __init__(self):
        super(DeepNet, self).__init__()
        self.n_channels = 20
        self.f_freq = 100

        self.constraint = tf.keras.constraints.MaxNorm(2)

        self.mp = tf.keras.layers.MaxPool2D((1, 2), (1, 2))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.elu = tf.keras.layers.ELU()

        self.conv1 = tf.keras.layers.Conv2D(25, (1, 5), kernel_constraint=self.constraint)
        self.conv2 = tf.keras.layers.Conv2D(25, (self.n_channels, 1), kernel_constraint=self.constraint)
        self.conv3 = tf.keras.layers.Conv2D(50, (1, 5), kernel_constraint=self.constraint)
        self.conv4 = tf.keras.layers.Conv2D(100, (1, 5), kernel_constraint=self.constraint)
        self.conv5 = tf.keras.layers.Conv2D(200, (1, 5), kernel_constraint=self.constraint)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(2, kernel_constraint=self.constraint)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.mp(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.elu(x)
        x = self.mp(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.elu(x)
        x = self.mp(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.elu(x)
        # x = tf.math.reduce_max(x, -2)
        x = self.mp(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.dropout(x)
        y_hat = self.softmax(self.dense(x))
        return y_hat

class MSNN(tf.keras.Model):
    tf.keras.backend.set_floatx("float32")
    def __init__(self):
        super(MSNN, self).__init__()
        self.n_channels=20
        self.f_freq = 100

        # Regularizer
        self.regularizer = tf.keras.regularizers.L1L2(l1=.001, l2=.01)

        # Activation functions
        self.activation = tf.keras.layers.LeakyReLU()
        self.softmax = tf.keras.layers.Softmax()

        # Spectral convolution
        self.conv0 = tf.keras.layers.Conv2D(4, (1, int(self.f_freq/2)), kernel_regularizer=self.regularizer)
        # Spatio-temporal convolution
        self.conv1t = tf.keras.layers.SeparableConv2D(16, (1, 25), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv1s = tf.keras.layers.Conv2D(16, (self.n_channels, 1), kernel_regularizer=self.regularizer)

        self.conv2t = tf.keras.layers.SeparableConv2D(32, (1, 15), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv2s = tf.keras.layers.Conv2D(32, (self.n_channels, 1), kernel_regularizer=self.regularizer)

        self.conv3t = tf.keras.layers.SeparableConv2D(64, (1, 6), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv3s = tf.keras.layers.Conv2D(64, (self.n_channels, 1), kernel_regularizer=self.regularizer)


        # Flatteninig
        self.flatten = tf.keras.layers.Flatten()

        # Dropout
        self.dropout = tf.keras.layers.Dropout(0.5)

        # Decision making
        self.dense = tf.keras.layers.Dense(2, activation=None, kernel_regularizer=self.regularizer)

    def embedding(self, x, random_mask=False):
        x = self.activation(self.conv0(x))

        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        feature = tf.concat((f1, f2, f3), -1)

        return feature

    def classifier(self, feature):
        # Flattening, dropout, mapping into the decision nodes
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        y_hat = self.softmax(self.dense(feature))
        return y_hat

    def GAP(self, feature):
        return tf.reduce_mean(feature, -2)

    def call(self, x):
        # Extract feature using MSNN encoder
        feature = self.embedding(x)

        # Global Average Pooling (MSNN)
        feature = self.GAP(feature)

        # Decision making
        y_hat = self.classifier(feature)
        return y_hat


class EEGAgent(tf.keras.Model):
    tf.keras.backend.set_floatx("float32")
    def __init__(self):
        super(EEGAgent, self).__init__()
        self.n_channels = 22
        self.f_freq = 250#100#250

        # Regularizer
        self.regularizer = tf.keras.regularizers.L1L2(l1=.001, l2=.01)

        # Activation functions
        self.activation = tf.keras.layers.LeakyReLU()
        self.softmax = tf.keras.layers.Softmax()

        # Spectral convolution
        self.conv0 = tf.keras.layers.Conv2D(4, (1, int(self.f_freq/2)), kernel_regularizer=self.regularizer)
        # Spatio-temporal convolution
        self.conv1t = tf.keras.layers.SeparableConv2D(16, (1, 25), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv1s = tf.keras.layers.Conv2D(16, (self.n_channels, 1), kernel_regularizer=self.regularizer)

        self.conv2t = tf.keras.layers.SeparableConv2D(32, (1, 15), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv2s = tf.keras.layers.Conv2D(32, (self.n_channels, 1), kernel_regularizer=self.regularizer)

        self.conv3t = tf.keras.layers.SeparableConv2D(64, (1, 6), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv3s = tf.keras.layers.Conv2D(64, (self.n_channels, 1), kernel_regularizer=self.regularizer)


        # Flatteninig
        self.flatten = tf.keras.layers.Flatten()

        # Dropout
        self.dropout = tf.keras.layers.Dropout(0.5)

        # Agent
        self.actor = tf.keras.layers.Dense(2, activation=None, kernel_regularizer=self.regularizer)
        self.critic = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=self.regularizer)

        # Decision making
        self.dense = tf.keras.layers.Dense(4, activation=None, kernel_regularizer=self.regularizer)

    def embedding(self, x, random_mask=False):
        x = self.activation(self.conv0(x))

        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        feature = tf.concat((f1, f2, f3), -1)

        if random_mask:
            # To calculate random mask average pooling performance!!!!
            random_mask = tf.round(tf.random.uniform(shape=feature.shape, minval=0, maxval=1, seed=951014)[:, :, :, 0])
            random_mask = tf.expand_dims(random_mask, -1)
            random_mask = tf.tile(random_mask, [1, 1, 1, feature.shape[-1]])
            feature = tf.math.multiply(feature, random_mask)

        return feature

    def classifier(self, feature):
        # Flattening, dropout, mapping into the decision nodes
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        y_hat = self.softmax(self.dense(feature))
        return y_hat

    def GAP(self, feature):
        return tf.reduce_mean(feature, -2)

    def agent(self, feature):
        feature = tf.squeeze(feature)
        return feature

    def call(self, x, reinpool=True):
        # Extract feature using MSNN encoder
        feature = self.embedding(x)

        if reinpool:
            feature = self.agent(feature)

        # Global Average Pooling (MSNN)
        feature = self.GAP(feature)

        # Decision making
        y_hat = self.classifier(feature)
        return y_hat