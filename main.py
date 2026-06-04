# This python script converts the MNIST dataset to the stuff

from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

import struct

for i in range(len(X_train)):
    with open(f"trainingData/{i}-{Y_train[i]}.bin", "wb") as binary_file:
        binary_file.write(struct.pack("<B", Y_train[i]))
        binary_file.write(
            b"\x1c\x1c"
        )  # 28 in hex according to ./imgSpec, the second two bytes are width and height
        for j in range(len(X_train[i])):
            for k in range(len(X_train[i][j])):
                binary_file.write(struct.pack("<B", X_train[i][j][k]))

for i in range(len(X_test)):
    with open(f"testingData/{i}-{Y_test[i]}.bin", "wb") as binary_file:
        binary_file.write(struct.pack("<B", Y_test[i]))
        binary_file.write(
            b"\x1c\x1c"
        )  # 28 in hex according to ./imgSpec, the second two bytes are width and height
        for j in range(len(X_test[i])):
            for k in range(len(X_test[i][j])):
                binary_file.write(struct.pack("<B", X_test[i][j][k]))
