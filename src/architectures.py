"""
OrbitalFire / ACV - Arquiteturas de CNN treinadas DO ZERO (sem pre-treino).

Duas arquiteturas para classificacao binaria de imagens de satelite
(wildfire x nowildfire), permitindo comparar como mudancas estruturais
(profundidade, BatchNorm, GlobalAveragePooling, Dropout) afetam o resultado.

Compatível com TensorFlow/Keras 2.x.
"""

from tensorflow.keras import layers, models

IMG_SIZE = (128, 128)
INPUT_SHAPE = (*IMG_SIZE, 3)


def data_augmentation():
    """Aumento de dados aplicado igualmente aos dois modelos (controle justo)."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")


def build_cnn_base(input_shape=INPUT_SHAPE):
    """
    Modelo A - CNN-Base (baseline simples).
    3 blocos convolucionais + Flatten + Dense. Sem normalizacao/regularizacao.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="CNN_Base")
    return model


def build_cnn_plus(input_shape=INPUT_SHAPE):
    """
    Modelo B - CNN-Plus (mais profundo e regularizado).
    4 blocos com BatchNorm + GlobalAveragePooling + Dropout.
    Esperado superar o baseline e atingir a referencia de 88%.
    """
    def bloco(x, filtros):
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        return x

    entrada = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(entrada)
    for f in (32, 64, 128, 256):
        x = bloco(x, f)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    saida = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(entrada, saida, name="CNN_Plus")


def compilar(model, lr=1e-3):
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
