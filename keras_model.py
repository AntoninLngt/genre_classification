from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense
from tensorflow.keras.optimizers import SGD

def build_model():
    model = Sequential(
        [
            Dense(128, input_dim=23, activation='relu'),
            Dense(10,activation="softmax")
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model


if __name__=="__main__":

    # test model
    model = build_model()
    print("Following model was built:")
    print(model.summary())