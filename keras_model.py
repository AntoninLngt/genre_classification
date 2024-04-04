from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, concatenate
from tensorflow.keras.optimizers import SGD

def build_model():
    # Define the first branch for audio input
    model_audio = Sequential([
        InputLayer(input_shape=[11025, 2], name="Input_audio"),
        Flatten(name="Flatten_audio")
    ])
    
    # Define the second branch for other input
    model_other = Sequential([
        InputLayer(input_shape=[11025], name="Input_other"),
        Flatten(name="Flatten_other")
    ])
    
    # Combine both branches
    merged_model = Sequential([
        concatenate([model_audio, model_other]),
        Dense(units=10, activation="softmax", name="Output")
    ])
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    return merged_model


if __name__=="__main__":

    # test model
    model = build_model()
    print("Following model was built:")
    print(model.summary())