import tensorflow as tf

@tf.function
def hard_tanh(x):
    return tf.maximum(tf.minimum(x, 1), -1)

class featureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Lambda(hard_tanh)
        
    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.activation(x)
        return x