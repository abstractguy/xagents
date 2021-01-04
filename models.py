from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model


def dqn_conv(input_shape, num_actions, duel=False, fc_units=512):
    x0 = Input(input_shape)
    x = Conv2D(32, 8, 4, activation='relu')(x0)
    x = Conv2D(64, 4, 2, activation='relu')(x)
    x = Conv2D(64, 3, 1, activation='relu')(x)
    x = Flatten()(x)
    fc1 = Dense(units=fc_units, activation='relu')(x)
    if not duel:
        output = Dense(units=num_actions)(fc1)
    else:
        fc2 = Dense(units=fc_units, activation='relu')(x)
        advantage = Dense(units=num_actions, activation='relu')(fc1)
        value = Dense(units=1, activation='relu')(fc2)
        output = Add()([advantage, value])
    return Model(x0, output)
