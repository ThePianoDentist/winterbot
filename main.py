import ast
import json
import os
import time
import traceback
from random import shuffle
from urllib.request import Request, urlopen

import keras
import numpy as np
from keras.layers import Dense, LSTM, Activation, TimeDistributed, Dropout, LeakyReLU
from keras.models import Sequential
from keras.optimizers import adam
import matplotlib.pyplot as plt

with open(os.path.join("/home/jdog/work/python/constants/heroes.json")) as f:
    HEROES = json.load(f)

ix_to_hero = {ix: char for ix, char in enumerate(sorted([int(k) for k in HEROES.keys()]))}
hero_to_ix = {char: ix for ix, char in enumerate(sorted([int(k) for k in HEROES.keys()]))}


def input_ids_to_categorical(heroes):
    input_ = [-1] * (113 * 4)
    for hero in (h for idx, h in enumerate(heroes) if idx in (0, 2, 9, 11, 17)):  # their bans
        input_[hero_ix_to_input_ix(hero_to_ix[hero], False, False)] = 1
    for hero in (h for idx, h in enumerate(heroes) if idx in (1, 3, 8, 10, 16)):  # our bans
        input_[hero_ix_to_input_ix(hero_to_ix[hero], True, False)] = 1
    for hero in (h for idx, h in enumerate(heroes) if idx in (4, 7, 13, 15, 18)):  # their picks
        input_[hero_ix_to_input_ix(hero_to_ix[hero], False, True)] = 1
    for hero in (h for idx, h in enumerate(heroes) if idx in (5, 6, 12, 14)):  # our picks
        input_[hero_ix_to_input_ix(hero_to_ix[hero], True, True)] = 1
    return input_


def hero_ix_to_input_ix(ix, is_us, is_pick):
    if not is_us:
        ix += 113 * 2
    if is_pick:
        ix += 113
    return ix


def request_(req_url, sleep_time=1):
    succeeded = False
    while not succeeded:
        try:
            print("Requesting: %s" % req_url)
            request = Request(req_url)
            request.add_header('User-Agent',
                               'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36')
            response = urlopen(request)
            out = response.read().decode(
                'utf8')  # cos python3 is kind of stupid http://stackoverflow.com/questions/6862770/python-3-let-json-object-accept-bytes-or-let-urlopen-output-strings
            time.sleep(sleep_time)  # obey api rate limits
            succeeded = True
        except:
            sleep_time += 1
            traceback.print_exc()
            continue
    return out


def get_matches():
    # holy crap this league has lots of games 4122
    skip = 0
    leagues_json = []
    while True:
        new = json.loads(request_("https://api.stratz.com/api/v1/league?take=100&skip=%s" % skip))
        if not len(new):
            break
        leagues_json.extend(new)
        skip += 100
    matches = []

    for league in (
            l["id"] for l in leagues_json if l["id"] > 10):
        league_games = json.loads(request_(
            "https://api.stratz.com/api/v1/match?leagueId=%s&include=pickBan,GameVersionId&take=250" % league,
        ))
        left = league_games["total"] - 250
        skip = 250
        matches.extend([l["pickBans"] for l in league_games["results"] if l.get("gameVersionId") >= 75])
        while left >= 0:
            league_games = json.loads(request_(
                "https://api.stratz.com/api/v1/match?leagueId=%s&include=pickBan,GameVersionId&take=250&skip=%s" % (league, skip),
            ))
            matches.extend([l["pickBans"] for l in league_games["results"] if l.get("gameVersionId") >= 75])
            skip += 250
            left -= 250
    return matches


def load_data():
    data = []
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")):
        # with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")) as f:
        #     data = ast.literal_eval(f.read())

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data2.bak2")) as f:
            inputs = ast.literal_eval(f.read())
            print("loadede inputs")

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data3.bak2")) as f:
            outputs = ast.literal_eval(f.read())
            print("loadede outputs")
    else:
        matches = get_matches()
        inputs = []
        outputs = []
        for match in matches:
            input_ = [-1] * (113 * 4)
            if len(match) != 20:
                continue
            last_pick_team = match[-1]["team"]
            for i, pick in enumerate(match):
                is_pick = pick["isPick"]
                hero_id = pick["heroId"]
                data.append(hero_id)
                if pick["team"] == last_pick_team:  # our picks/bans
                    if i == 19:  # lets just focus on testing last pick
                        outputs.append(hero_id)
                        inputs.append(input_)  # copy necessary otherwise future loops will screw up old results!

                    index = hero_to_ix[hero_id]
                    if is_pick:
                        index += 113
                    input_[index] = 1
                else:  # enemy picks/bans
                    index = hero_to_ix[hero_id] + 113 * 2
                    if is_pick:
                        index += 113
                    input_[index] = 1

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt"), "w+") as f:
            f.write(str(data))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data2.txt"), "w+") as f:
            f.write(str(inputs))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data3.txt"), "w+") as f:
            f.write(str(outputs))

    return data, inputs, outputs

# heroes = list(set(data))
# VOCAB_SIZE = len(heroes)  # number of heroes available in captains mode essentially
# SEQ_LENGTH = 20
# num_sequences = int(len(data) / SEQ_LENGTH)


def next_pick(model, inputs, already_picked, pick_max=True, allow_duplicates=True):
    a = model.predict(np.array([inputs]))
    probs = a[0]
    probs = (i for i in reversed(sorted(enumerate(probs), key=lambda x: x[1])))  # https://stackoverflow.com/a/6422754
    if not allow_duplicates:
        probs = (p for p in probs if ix_to_hero[p[0]] not in already_picked)  # is generators actually more efficient here when knowing going to iterate over whole list?

    if pick_max:
        return next(probs)[0]
    else:
        total_prob = sum(p[1] for p in probs)
        # prob not necessary to be REALLY random but meh
        randy = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)  #https://stackoverflow.com/a/33359758/3920439
        randy *= total_prob
        counter = 0.0
        for hero_ix, prob in probs:
            counter += prob
            if counter >= randy:
                return ix_to_hero[hero_ix]


def predict_last_pick(model, *args, full_input=None, pick_max=True, allow_duplicates=True):
    if full_input:
        return next_pick(
            model, full_input, [ix_to_hero[index % 113] for index, value in enumerate(full_input) if value == 1],
            pick_max=pick_max
        )

    our_bans, our_picks, their_bans, their_picks = args
    input_ = [-1] * (113 * 4)
    for b in our_bans:
        input_[hero_ix_to_input_ix(hero_to_ix[b], True, False)] = 1
    for b in their_bans:
        input_[hero_ix_to_input_ix(hero_to_ix[b], False, False)] = 1
    for p in our_picks:
        input_[hero_ix_to_input_ix(hero_to_ix[p], True, True)] = 1
    for p in their_picks:
        input_[hero_ix_to_input_ix(hero_to_ix[p], False, True)] = 1
    return next_pick(model, input_, our_bans + our_picks + their_bans + their_picks, pick_max=pick_max,
                     allow_duplicates=allow_duplicates)


def basic_nn(inputs_, outputs_):
    # validation_split does not shuffle data set i dont think
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    #inputs_, outputs_ = shuffle(inputs_, outputs_)
    # from collections import Counter
    # c = Counter([tuple(i) for i in inputs_])
    # print(len(inputs_))
    # print(len(c))
    inputs_, val_inputs, outputs_, val_outputs = train_test_split(inputs_, outputs_, test_size=0.2)
    # could also just use from sklearn.model_selection import train_test_split
    # check/google relu vs sigmoid
    dim = 113 * 4 * 2
    outputs_ = [hero_to_ix[o] for o in outputs_]
    val_outputs = [hero_to_ix[o] for o in val_outputs]
    one_hot_labels = keras.utils.to_categorical(outputs_, num_classes=113)
    one_hot_labels_val = keras.utils.to_categorical(val_outputs, num_classes=113)
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(113*4,)))
    model.add(Dense(dim, activation='sigmoid'))
    model.add(LeakyReLU())   # normal relu has dead relu problem. sigmoids/tanhs have vanishing gradients
    #model.add(Dropout(0))
    model.add(Dense(dim // 2, activation='sigmoid'))
    model.add(LeakyReLU())
    model.add(Dense(113, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
    model.fit(np.array(inputs_), one_hot_labels, epochs=2, batch_size=32, verbose=2, validation_data=(val_inputs, one_hot_labels_val))
    for i in range(10):
        print(HEROES[str(predict_last_pick(model, full_input=inputs_[~i]))]["localized_name"])
        print(HEROES[str(ix_to_hero[outputs_[~i]])]["localized_name"])
    return model


def generate_draft(model, pick_max=True, allow_duplicates=True):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_hero = [ix_to_hero[ix[-1]]]
    X = np.zeros((1, SEQ_LENGTH, VOCAB_SIZE))
    picks_a = []
    picks_b = []
    bans_a = []
    bans_b = []
    for i in range(SEQ_LENGTH):
        X[0, i, :][ix[-1]] = 1
        if i in [0, 2, 9, 11, 17]:
            bans_a.append(HEROES[str(ix_to_hero[ix[-1]])]["localized_name"])
        elif i in [1, 3, 8, 10, 16]:
            bans_b.append(HEROES[str(ix_to_hero[ix[-1]])]["localized_name"])
        elif i in [4, 7, 13, 15, 18]:
            picks_a.append(HEROES[str(ix_to_hero[ix[-1]])]["localized_name"])
        else:  # 5 6 12 14 19
            picks_b.append(HEROES[str(ix_to_hero[ix[-1]])]["localized_name"])
        predictions = model.predict(X[:, :i + 1, :])[0]
        if allow_duplicates:
            predictions = (p for i, p in enumerate(predictions) if p[0] not in ix)
        if pick_max:
            ix = np.argmax(predictions, 1)
        else:
            total_prob = sum(p[1] for p in predictions)
            # prob not necessary to be REALLY random but meh
            randy = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)  # https://stackoverflow.com/a/33359758/3920439
            randy *= total_prob
            counter = 0.0
            for hero_ix, prob in predictions:
                counter += prob
                if counter >= randy:
                    ix.append(hero_ix)
                    break
        y_hero.append(ix_to_hero[ix[-1]])
    print_list = picks_a + picks_b + bans_a + bans_b
    print("Pick: %s, %s, %s, %s, %s VS %s, %s, %s, %s, %s\n\nBan: %s, %s, %s, %s, %s VS %s, %s, %s, %s, %s" %
          tuple(print_list))
    return y_hero


def phase_accuracy(model, x, phase_start, phase_length):
    samples = len(x)
    correct = [0] * phase_length
    for i in range(samples):
        for j in range(phase_length):
            seq = x[i]
            already_picked = seq[:phase_start + j]
            predict_next = next_pick(model, [ix_to_hero[np.where(p==1)[0][0]] for p in already_picked])
            if hero_to_ix[predict_next] == np.where(seq[phase_start + j]==1)[0][0]:
                correct[j] += 1

    return sum(correct) / (len(correct) * samples)


def last_phase_pick_accuracy(model, x):
    return phase_accuracy(model, x, 18, 2)


def last_phase_ban_accuracy(model, x):
    return phase_accuracy(model, x, 16, 2)


def second_phase_pick_accuracy(model, x):
    return phase_accuracy(model, x, 12, 4)


def next_pick_rnn(model, already_picked):
    X = np.zeros((1, SEQ_LENGTH, VOCAB_SIZE))
    for i, pick in enumerate(already_picked):
        ix = hero_to_ix[pick]
        X[0, i, :][ix] = 1
    return next_pick(model, X[:, :i + 1, :], already_picked)


def plot_learning_curves(hist, filename):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    ax0.set(xlabel="epochs", ylabel="accuracy")
    ax1.set(xlabel="epochs", ylabel="mse")
    ax0.plot(hist['acc'], label="train")
    ax0.plot(hist['val_acc'], label="val")
    ax1.plot(hist['mean_squared_error'], label="train")
    ax1.plot(hist['val_mean_squared_error'], label="val")
    ax0.legend(loc='upper left')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs/%s.png" % filename))
    plt.close()


def rnn(nodes1, nodes2, nodes3, dropout1, dropout2, dropout3, epochs=200, learning_rate=0.001, batch_size=16):
    # 0.001 is default for adam
    X = np.zeros((num_sequences, SEQ_LENGTH, VOCAB_SIZE))
    y = np.zeros((num_sequences, SEQ_LENGTH, VOCAB_SIZE))
    seq_counter = 0
    tmp = list(range(0, num_sequences))
    shuffle(tmp)
    for i in tmp:
        seq_counter += 1
        X_sequence = data[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH]
        X_sequence_ix = [hero_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        for j in range(SEQ_LENGTH):
            input_sequence[j][X_sequence_ix[j]] = 1.
        X[i] = input_sequence

        y_sequence = data[i * SEQ_LENGTH + 1:(i + 1) * SEQ_LENGTH + 1]
        y_sequence_ix = [hero_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))

        for j in range(SEQ_LENGTH):
            target_sequence[j][y_sequence_ix[j]] = 1.
        y[i] = target_sequence

    validation_SEQ_LENGTH = int(num_sequences * 0.2)
    X, Xval = X[:-validation_SEQ_LENGTH], X[-validation_SEQ_LENGTH:]
    y, yval = y[:-validation_SEQ_LENGTH], y[-validation_SEQ_LENGTH:]

    model = Sequential()
    if dropout1:
        model.add(Dropout(dropout1, input_shape=(None, VOCAB_SIZE)))
    model.add(LSTM(nodes1, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    if dropout2:
        model.add(Dropout(dropout2))
    model.add(LSTM(nodes2, return_sequences=True))
    if dropout3:
        model.add(Dropout(dropout3))
    if nodes3:
        model.add(LSTM(nodes3, return_sequences=True))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    optimizer = adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

    hist_dict = {"acc": [], "val_acc": [], "mean_squared_error": [], "val_mean_squared_error": []}  # cant just use results.history because doing epoch 1 in a loop
    for i, e in enumerate(range(epochs)):
        print("Epoch number %s" % (i + 1))
        results = model.fit(X, y, validation_data=(Xval, yval), verbose=2, epochs=1, batch_size=batch_size)
        generate_draft(model)
        print("Last pick accuracy: %s %%" % (last_phase_pick_accuracy(model, Xval) * 100))
        #print("Last ban accuracy: %s %%" % (last_phase_ban_accuracy(model, Xval) * 100))
        #print("2nd phase pick accuracy: %s %%" % (second_phase_pick_accuracy(model, Xval) * 100))
        if i % 10 == 0:
            model.save_weights('weights2.hdf5')
        out = {
            'mse': results.history["mean_squared_error"],
            'val_mse': results.history["val_mean_squared_error"],
            'accuracy': results.history["acc"],
            'val_accuracy': results.history["val_acc"],
            'nodes1': nodes1,
            'nodes2': nodes2,
            'nodes3': nodes3,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'dropout3': dropout3,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'last_phase_pick_accuracy': last_phase_pick_accuracy(model, Xval),
            "last_phase_ban_accuracy": last_phase_ban_accuracy(model, Xval),
            "second_phase_pick_accuracy": second_phase_pick_accuracy(model, Xval)
        }
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results/rnn_%s_%s_%s_%s_%s_%s_%s_%s.json" % (
                                   nodes1, nodes2, nodes3, dropout1, dropout2, dropout3, learning_rate, batch_size
                               )), "w+") as f:
            json.dump(out, f)

        hist_dict['mean_squared_error'].append(results.history["mean_squared_error"][0]),
        hist_dict['val_mean_squared_error'].append(results.history["val_mean_squared_error"][0]),
        hist_dict['acc'].append(results.history["acc"][0]),
        hist_dict['val_acc'].append(results.history["val_acc"][0]),

        plot_learning_curves(hist_dict, "rnn_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                       nodes1, nodes2, nodes3, dropout1, dropout2, dropout3, learning_rate, batch_size
                                   ))
    return model

if __name__ == "__main__":
    from tensorflow.python.client import device_lib
    data, inputs, outputs = load_data()
    model = basic_nn(inputs, outputs)
    #model = rnn(500, 300, 300, 0.05, 0.4, 0, epochs=200, batch_size=16, learning_rate=0.005)
    model.save('my_model.h5')
