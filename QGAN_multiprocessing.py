# QGAN version with multiprocessing
# This isn't a Jupyter notebook because Jupyter doesn't like multiprocessing for some reason

# disable annoying tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from pennylane import numpy as np
from numpy import pi
from numpy import linalg as la

import pennylane as qml
import cirq
import math as m
import numpy as np
from tqdm import tqdm # library for showing progress bars
from concurrent.futures import ProcessPoolExecutor

from sklearn.preprocessing import MinMaxScaler

res = 4
numQubits = 2 * res + 1
numWeights = 3 * res - 1
dev = qml.device('cirq.simulator', wires=numQubits)
executor = ProcessPoolExecutor(max_workers=(os.cpu_count() - 1))

# initialize weights
disc_weights = tf.Variable(np.pi * np.random.uniform(size=(numWeights,)))
gen_weights = tf.Variable(np.pi * np.random.uniform(size=(numWeights,)))

# https://ruder.io/optimizing-gradient-descent/
vt_gen = tf.Variable(np.zeros((numWeights)))
vt_disc = tf.Variable(np.zeros((numWeights)))

# Utility function to display images
def display_image(X_data, image_num, Y_labels=None):
    plt.figure(figsize=(10,10))
    for i in range(image_num):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_data[i], cmap='Greys')
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        if Y_labels is not None:
            plt.xlabel(Y_labels[i])
    plt.show()

# Load the trained autoencoder
def load_autoencoder():
    model = keras.models.load_model('mnist_enc/mnist_enc/')
    layer = model.get_layer('dense_2')
    bottom_input = keras.layers.Input(model.input_shape[1:], name="Encoder Input")
    bottom_output = bottom_input
#     bottom_output = keras.layers.Input(model.input_shape[1:])
    top_input = keras.layers.Input(layer.output_shape[1:], name="Decoder Input")
    top_output = top_input

    bottom = True
    for layer in model.layers:
        if bottom and layer.name != 'input_1':
            bottom_output = layer(bottom_output)
        else:
            top_output = layer(top_output)
        if layer.name == 'dense_2':
            bottom = False

    enc_model = keras.Model(bottom_input, bottom_output)
    dec_model = keras.Model(top_input, top_output)
    return enc_model, dec_model

# Encodes MNIST data into rotations. Return encoded data and the MinMaxScaler used to
# normalize it (so it can be used to decode the compressed data)
def encode(encoder, data):
    encoded = encoder.predict(data.reshape(-1, 784))
    scaler = MinMaxScaler()
    scaler.fit(encoded)
    scaled_encoded = scaler.transform(encoded)
    qubitized_data = 2*np.arcsin(np.sqrt(scaled_encoded)) # Should play around with
    return qubitized_data, scaler

# Decodes compressed data
def decode(decoder, data, scaler):
    unscaled = scaler.inverse_transform(np.sin(data / 2) ** 2)
    return decoder.predict(unscaled).reshape(-1,28,28)

# Generator and discriminator circuits
def generator(thetas, res):
    for i in range(res):
        qml.RY(thetas[i], wires=i + 1 + res)

    for i in range(res - 1):
        qml.IsingYY(thetas[i + res], wires=[i + 1 + res, i + 2 + res])

    for i in range(res):
        if (i + 2) % res == 0:
            qml.CRY(thetas[i + 2 * res - 1], wires=[i + 1 + res, 2 * res])
        else:
            qml.CRY(thetas[i + 2 * res - 1], wires=[i + 1 + res, ((i + 2) % res) + res])


def discriminator(thetas, res):
    for i in range(res):
        qml.RY(thetas[i], wires=i + 1)

    for i in range(res - 1):
        qml.IsingYY(thetas[i + res], wires=[i + 1, i + 2])

    for i in range(res):
        if (i + 2) % res == 0:
            qml.CRY(thetas[i + 2 * res - 1], wires=[i + 1, res])
        else:
            qml.CRY(thetas[i + 2 * res - 1], wires=[i + 1, (i + 2) % res])

# Full QGAN circuit
@qml.qnode(dev, interface="tf")
def createCircuit(disc_weights, gen_weights, res, real=False, real_data=None):
    qml.Hadamard(wires=0)
    discriminator(disc_weights, res)
    if not real:
        generator(gen_weights, res)

    if real:
        # Encode real data into Quantum Circuit
        for i in range(0, res):
            qml.RY(real_data[i], wires=i + 1 + res)

    for i in range(res):
        qml.CSWAP(wires=[0, i + 1, i + 1 + res])
    qml.Hadamard(wires=0)

    return qml.expval(qml.PauliZ(0))

# Cost functions
def real_disc_cost(expectation):
    if expectation <= 0.005:  # They did in the paper
        expectation = 0.005
    return -tf.math.log(expectation)


def fake_disc_cost(expectation):
    if expectation <= 0.005:
        expectation = 0.005
    return -tf.math.log(1 - expectation)


def gen_cost(expectation):
    if expectation <= 0.005:
        expectation = 0.005
    return -tf.math.log(expectation)  # Maybe incorrect?


def parameter_shift_term(cost, i, params=None):
    if params is None:
        # disc_model (boolean) : which parameters are being shifted, generator's if False.
        # real (boolean) : whether real data is being used
        params = {"disc_model": True, "real": False, "real_data": None}
    if params["disc_model"]:
        shifted = disc_weights.numpy().copy()
        shifted[i] += np.pi / 2
        exp = createCircuit(shifted, gen_weights, res, real=params["real"], real_data=params["real_data"])
    else:
        shifted = gen_weights.numpy().copy()
        shifted[i] += np.pi / 2
        exp = createCircuit(disc_weights, shifted, res, real=False, real_data=None)
    forward = cost(exp)  # forward evaluation

    if params["disc_model"]:
        shifted[i] -= np.pi
        exp = createCircuit(shifted, gen_weights, res, real=params["real"], real_data=params["real_data"])
    else:
        shifted[i] -= np.pi / 2
        exp = createCircuit(disc_weights, shifted, res, real=False, real_data=None)
    backward = cost(exp)  # backward evaluation

    return (0.5 * (forward.numpy() - backward.numpy()))

def calc_gradient(cost, params):
    grad = [0] * numWeights
    for i in range(numWeights):
        grad[i] = parameter_shift_term(cost, i, params=params)
    return np.array(grad)

# Function that produces average measurements from trained generator
@qml.qnode(dev)
def generate_output(weights):
    generator(weights, res)
    return [qml.expval(qml.PauliZ(i+res+1)) for i in range(res)]

@qml.qnode(dev, interface="tf")
def gen_loss():
    generator(gen_weights, res)
    return gen_cost(qml.expval(qml.PauliZ(0)))

# Single training step
def train_step(real_data, hp=None):
    global disc_weights, gen_weights, mt_gen, mt_disc, vt_gen, vt_disc, executor
    if hp is None:
        # R (int) : Disc - Gen train count
        hp = {"R" : len(encoded_data), "I" : len(encoded_data) // 10, "batch_size" : 1, "learning_rate" : 0.01}
    # Discriminator on real data
    print("Discriminator on real")
    futures = []
    for i in tqdm(range(len(real_data)// hp["batch_size"])):
        '''average_grad = np.array([0] * numWeights, dtype="float64")
        for image in real_data[i*hp["batch_size"]:(i+1)*hp["batch_size"]]:
            params = {"disc_model" : True, "real" : True, "real_data" : image}
            gradient = calc_gradient(real_disc_cost, params) # Have to find the actual format for gradient to use tf.apply_gradients
            average_grad = average_grad + gradient
        average_grad /= hp["batch_size"]'''
        params_list = [{"disc_model" : True, "real" : True, "real_data" : image}
                       for image in real_data[i*hp["batch_size"]:(i+1)*hp["batch_size"]]]
        futures = executor.map(calc_gradient, [real_disc_cost for x in range(hp["batch_size"])], params_list)
        average_grad = sum(list(futures))
        average_grad /= hp["batch_size"]

        vt_disc = 0.9 * vt_disc - hp["learning_rate"] * average_grad * (disc_weights - 0.9 * vt_disc)

        # Apply gradient
        # disc_weights = disc_weights - hp["learning_rate"]*average_grad
        disc_weights = disc_weights - vt_disc

    # Discriminator on fake data
    print("Discriminator on fake")
    for _ in tqdm(range(hp["R"] // hp["batch_size"])):
        '''average_grad = np.array([0] * numWeights, dtype="float64")
        params = {"disc_model" : True, "real" : False, "real_data" : None}
        for i in range(hp["batch_size"]):
            gradient = calc_gradient(fake_disc_cost, params)
            average_grad = average_grad + gradient
        average_grad /= hp["batch_size"]'''
        params_list = [{"disc_model": True, "real": False, "real_data": None} for _ in range(hp["R"])]
        futures = executor.map(calc_gradient, [fake_disc_cost for x in range(hp["batch_size"])], params_list)
        average_grad = sum(list(futures))
        average_grad /= hp["batch_size"]


        vt_disc = 0.9 * vt_disc - hp["learning_rate"] * average_grad * (disc_weights - 0.9 * vt_disc)

        # Apply gradient
        # disc_weights = disc_weights - hp["learning_rate"]*average_grad
        disc_weights = disc_weights - vt_disc

    print("Generator")
    #for _ in tqdm(range(hp["I"])):
    for _ in tqdm(range(hp["I"] // hp["batch_size"])):
        # params = {"disc_model" : False, "real" : False, "real_data" : None}
        # gradient = calc_gradient(gen_cost, params)
        params_list = [{"disc_model": True, "real": False, "real_data": None} for _ in range(hp["I"])]
        futures = executor.map(calc_gradient, [gen_cost for x in range(hp["batch_size"])], params_list)
        average_grad = sum(list(futures))
        average_grad /= hp["batch_size"]

        vt_gen = 0.9 * vt_gen - 2.5 * hp["learning_rate"] * average_grad * (gen_weights - 0.9 * vt_gen)

        # They use 10 times less training steps for generator but a 2.5 times higher learning rate.
        # Apply gradient
        # gen_weights = gen_weights - 2.5*hp["learning_rate"]*gradient
        gen_weights = gen_weights - vt_gen

# Train
def train(real_data, epochs=25, hp=None, reset=True, num_shots=30):
    last_loss = 999
    global disc_weights, gen_weights
    if reset:
        disc_weights = tf.Variable(np.pi * np.random.uniform(size=(numWeights,)))
        gen_weights = tf.Variable(np.pi * np.random.uniform(size=(numWeights,)))

    for i in range(epochs):
        print("Epoch {} of {}".format(i + 1, epochs))
        train_step(real_data, hp)
        # Printing out loss/visualizing examples at certain training steps.
        loss = gen_cost(createCircuit(disc_weights, gen_weights, res, real=False))
        print("Cost: {}".format(loss))
        # Drop the learning rate a bit
        hp["learning_rate"] *= 0.9
        if i % 5 == 0:
            data = (generate_output(gen_weights, shots=num_shots) + 1) / 2
            image = decode(dec, data.reshape(1, -1), scaler)
            display_image(image, 1)

if __name__ == "__main__":
    # Prepare and Load Data
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    full_X = np.concatenate((X_train, X_test))
    full_Y = np.concatenate((Y_train, Y_test))
    full_X = full_X / 255
    other_X = full_X[:10000]  # Paper only uses first 10000 images for training
    other_Y = full_Y[:10000]
    # Select only 3s and 8s
    X = []
    Y = []
    for i in range(len(other_X)):
        if other_Y[i] == 8 or other_Y[i] == 3:
            X.append(other_X[i])
            Y.append(other_Y[i])
    X = np.array(X)
    Y = np.array(Y)

    # Display select images
    display_image(X, 25, Y)

    enc, dec = load_autoencoder()
    encoded_data, scaler = encode(enc, X)

    train(encoded_data, hp={"R": len(encoded_data), "I": len(encoded_data) // 10, "batch_size": 32, "learning_rate": 0.01})

    images = []
    outputs = []
    for i in range(25):
        outputs.append((generate_output(gen_weights, shots=20) + 1) / 2)
    images = [decode(dec, x.reshape(1, -1), scaler).reshape((28, 28)) for x in outputs]
    display_image(images, 25)