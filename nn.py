import tensorflow as tf
from tensorflow import keras
import speck
import datetime


def build_model(depth=10):
    inputs = keras.Input(shape=(2*32,)) 

    x = Block1()(inputs)

    for _ in range(depth):
        x = Block2i()(x)

    outputs = Block3()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_model1(depth=10, num_filters=32, kernel_size=3, reg_param=10**-5):
    inputs = keras.Input(shape=(2*32,))
    x = keras.layers.Reshape((-1, 16))(inputs)
    x = keras.layers.Permute((2,1))(x)
    
    # Block 1
    x = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l2(reg_param))(x)
    x = keras.layers.BatchNormalization()(x)

    # Block 2-i
    for _ in range(depth):
        shortcut = x
        x = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_regularizer=keras.regularizers.l2(reg_param))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')

        x = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_regularizer=keras.regularizers.l2(reg_param))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')

        x = keras.layers.Add()([x, shortcut])
    
    # Block 3
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(reg_param))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(reg_param))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(reg_param))(x)
    output = keras.layers.Activation(x)
    model = keras.Model(inputs=inputs, outputs=output)
    return model


    
        
        
        

class SpeckModel(keras.Model):
    def __init__(self, depth=10, reg_param=0.0001, **kwargs):
        super(SpeckModel, self).__init__(**kwargs)

        self.block1 = Block1(reg_param=reg_param)
        self.res_tower = [Block2i(reg_param=reg_param) for _ in range(depth)]
        self.block3 = Block3(reg_param=reg_param)

    def call(self, inputs, training=False):

        x = keras.layers.Reshape((-1, 16))(inputs);
        x = keras.layers.Permute((2,1))(x);
        x = self.block1(inputs)
        for res_block in self.res_tower:
            x = res_block(x)
        x = self.block3(x, is_training=training)

        return x


class Block1(keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=1, reg_param=0.0001, **kwargs):
        super(Block1, self).__init__(**kwargs)

        # Creating layers
        self.conv = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, kernel_regularizer=keras.regularizers.l2(reg_param))
        self.bn = keras.layers.BatchNormalization()
        
        def __call__(self, x, is_training):

            # Building the block
            x = self.conv(x)
            x = self.bn(x, training=is_training)
            x= tf.nn.relu(x)

            return x

class Block2i(keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=3, reg_param=0.0001, **kwargs):
        super(Block2i, self).__init__(**kwargs)

        # Creating layers
        self.conv1 = keras.layers.Conv1D(filters=num_filters,
                                         kernel_size=kernel_size,
                                         padding="same",
                                         kernel_regularizer=keras.regularizers.l2(reg_param)
                                         )
        self.bn1 = keras.layers.BatchNormalization()

        self.conv2 = keras.layers.Conv1D(filters=num_filters,
                                         kernel_size=kernel_size,
                                         kernel_regularizer=keras.regularizers.l2(reg_param),
                                         padding="same")
        self.bn2 = keras.layers.BatchNormalization()

        def __call__(self, x, is_training):
            # Saving x to create shortcut
            shortcut = x
        
            # Building the Residual block
            x = self.conv1(x)
            x = self.bn1(x, training=is_training)
            x = tf.nn.relu(x)

            x = self.conv2(x)
            x = self.bn2(x, training=is_training)
            x = tf.nn.relu(x)

            # returning x with the shortcut added
            return x + shortcut


class Block3(keras.layers.Layer):
    def __init__(self, reg_param=0.0001, **kwargs):
        super(Block3, self).__init__(**kwargs)

        # Creating layers

        self.flatten = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(reg_param))
        self.bn1 = keras.layers.BatchNormalization()

        self.dense2 = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(reg_param))
        self.bn2 = keras.layers.BatchNormalization()

        self.final = keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(reg_param))

    def __call__(self, x, is_training=False):

        # Building the block
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.relu(x)

        x = self.dense2(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.relu(x)

        x = self.final(x)
        x = tf.nn.sigmoid(x)

        return x


        

def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
    return res;

def make_checkpoint(datei):
  res = keras.callbacks.ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);

if __name__ == "__main__":

    x_train, y_train = speck.make_train_data(10**7, 5) 
    x_val, y_val = speck.make_train_data(10**6, 5)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    lr = keras.callbacks.LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    check = make_checkpoint("./fresh_models/"+'best'+str(5)+'depth'+str(10)+'.h5');

    # model = SpeckModel(depth=10, reg_param=10**-5)
    model = build_model1()
    # model = build_model()
    model.compile(
            optimizer=keras.optimizers.Adam(),
            loss = keras.losses.BinaryCrossentropy(),
            metrics=["acc"],
            )
    history = model.fit(
            x_train,
            y_train,
            batch_size=5000,
            epochs=200,
            validation_data=(x_val, y_val),
            callbacks=[lr, check, tensorboard_callback]
            )
# 
# print("Creating jpg...")
# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=False, show_dtype=False,
#     show_layer_names=True, rankdir='TB', expand_nested=True, dpi=150
# )
# print("yeet")
# 



    

