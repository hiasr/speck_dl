import tensorflow as tf
from tensorflow import keras
import speck
import datetime


def build_model_gohr(depth=10, num_filters=32, kernel_size=3, reg_param=10**-5):
    # Shaping input
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
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_regularizer=keras.regularizers.l2(reg_param))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

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
    output = keras.layers.Activation('sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=output)
    return model



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
    model = build_model_gohr()
    # model = build_model()
    model.compile(
            optimizer=keras.optimizers.Adam(),
            loss = keras.losses.BinaryCrossentropy(),
            metrics=["acc"],
            )
#     history = model.fit(
#             x_train,
#             y_train,
#             batch_size=5000,
#             epochs=200,
#             validation_data=(x_val, y_val),
#             callbacks=[lr, check, tensorboard_callback]
#             )
# 
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=False, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=True, dpi=150
    )




    

