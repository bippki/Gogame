'''from dlgo import agent
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input
def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bot = agent.RandomBot()
    while not game.is_over():
        print(chr(27) + "[2J")
        print_board(game.board)
        if game.next_player == gotypes.Player.black:
            human_move = input(' ')
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)

if __name__ == '__main__':
    main()'''
def main():
    import h5py
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    from networks import test_model
    go_board_rows, go_board_cols = 19, 19
    num = go_board_rows * go_board_cols
    encoder = get_encoder_onp(go_board_rows, go_board_cols)
    X = np.load('C:/Users/LubluKotov/PycharmProjects/Gogame/data/features.npy')
    Y = np.load('C:/Users/LubluKotov/PycharmProjects/Gogame/datalabels.npy')
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = test_model.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
        metrics=['accuracy'])
    model.fit(X, Y, batch_size=128, epochs=10, verbose=1)
    model.save("C:/Users/LubluKotov/PycharmProjects/Gogame/better_ser.h5")
    model.save("C:/Users/LubluKotov/PycharmProjects/Gogame/agents/9_2.h5")
    deep_learning_bot.serialize("../agents/better_final.h5")