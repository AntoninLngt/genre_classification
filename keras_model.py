from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_model(num_filters=32, kernel_size=5, activation='relu'):
    model = Sequential([
        InputLayer(input_shape=[128, 660, 1], name="Input_layer"),  # Modifier la forme d'entrée pour correspondre à la taille attendue
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),  # Première couche Conv2D
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),  # Deuxième couche Conv2D
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),  # Troisième couche Conv2D
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.3),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])

    return model


if __name__ == "__main__":

    # Testez le modèle avec des hyperparamètres personnalisés
    model = build_model(num_filters=64, kernel_size=3, activation='relu')
    print("Le modèle suivant a été construit :")
    print(model.summary())