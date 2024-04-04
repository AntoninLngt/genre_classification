from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense
from tensorflow.keras.optimizers import SGD

def build_model():
    model = Sequential(
        [
            InputLayer(input_shape=[11025, 2], name="Input_layer"),
            InputLayer(input_shape=[11025], name="Input_layer"),
            Flatten(name="Flatten"),
            Dense(activation="softmax", name="Dense", trainable=True, units=10),
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model


if __name__=="__main__":

    # test model
    model = build_model()
    print("Following model was built:")
    print(model.summary())