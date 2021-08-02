# keras ensemble from cifar data

from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout 
from tensorflow.keras.layers import Activation, Average, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
import glob
import numpy as np
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# create file for weight
weight_dir = 'weights'
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)


CONV_POOL_CNN_WEIGHT_FILE = os.path.join(
    os.getcwd(), 'weights', 'conv_pool_cnn_pretrained_weights.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(
    os.getcwd(), 'weights', 'all_cnn_pretrained_weights.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(
    os.getcwd(), 'weights', 'nin_cnn_pretrained_weights.hdf5')


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = to_categorical(y_train, num_classes=10)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_data()

print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | \
        y_test shape : {}'.format(x_train.shape, y_train.shape, x_test.shape,
                                  y_test.shape))

input_shape = x_train[0, :, :, :].shape
model_input = Input(shape=input_shape)


def conv_pool_cnn(model_input: Tensor) -> training.Model:

    x = Conv2D(96, kernel_size=(3, 3), activation='relu',
               padding='same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='conv_pool_cnn')

    return model


conv_pool_cnn_model = conv_pool_cnn(model_input)
NUM_EPOCHS = 20


def compile_and_train(model: training.Model, 
                      num_epochs: int) -> Tuple[History, str]:

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['acc'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(
        log_dir='logs/', histogram_freq=0, batch_size=32)
    history = model.fit(x=x_train, y=y_train, batch_size=32,
                        epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    weight_files = glob.glob(os.path.join(os.getcwd(), 'weights/*'))
    weight_file = max(weight_files, key=os.path.getctime)  # most recent file
    return history, weight_file

_, conv_pool_cnn_weight_file = compile_and_train(conv_pool_cnn_model, 
                                                  NUM_EPOCHS)

def evaluate_error(model: training.Model) -> np.float64:
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]    
    return error

try:
    conv_pool_cnn_weight_file
except NameError:
    conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)
error_1 = evaluate_error(conv_pool_cnn_model)

print("Error model 1: ", error_1)


## Second model
def all_cnn(model_input: Tensor) -> training.Model:
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
        
    model = Model(model_input, x, name='all_cnn')
    
    return model


all_cnn_model = all_cnn(model_input)
# the following line takes time to train, skip to use pretrained weights
_, all_cnn_weight_file = compile_and_train(all_cnn_model, NUM_EPOCHS)

try:
    all_cnn_weight_file
except NameError:
    all_cnn_model.load_weights(ALL_CNN_WEIGHT_FILE)
error_2 = evaluate_error(all_cnn_model)
print("Error model 2: ", error_2)


def nin_cnn(model_input: Tensor) -> training.Model:
    
    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='nin_cnn')
    
    return model


nin_cnn_model = nin_cnn(model_input)
_, nin_cnn_weight_file = compile_and_train(nin_cnn_model, NUM_EPOCHS)

try:
    nin_cnn_weight_file
except NameError:
    nin_cnn_model.load_weights(NIN_CNN_WEIGHT_FILE)
error_3 = evaluate_error(nin_cnn_model)
print("Error moel 3: ", error_3)


def ensemble(models: List [training.Model], 
             model_input: Tensor) -> training.Model:
    """ ensemble part
    """
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model


try:
    conv_pool_cnn_model.load_weights(conv_pool_cnn_weight_file)
except NameError:
    conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)
try:
    all_cnn_model.load_weights(all_cnn_weight_file)
except NameError:
    all_cnn_model.load_weights(ALL_CNN_WEIGHT_FILE)
try:
    nin_cnn_model.load_weights(nin_cnn_weight_file)
except NameError:
    nin_cnn_model.load_weights(NIN_CNN_WEIGHT_FILE)

# ensemble three models
models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]
ensemble_model = ensemble(models, model_input)
error_123 = evaluate_error(ensemble_model)
print("Error model ensemble (123): ", error_123)