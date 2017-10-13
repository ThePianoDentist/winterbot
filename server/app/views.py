from flask import request
from flask import jsonify

from app import app
from main import input_ids_to_categorical, HEROES, predict_last_pick


@app.route('/')
def display():
    print(app.basicNN)
    data = request.json
    #pick_ban = data["pick_ban"]
    pick_ban = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 60, 40]
    input_ = input_ids_to_categorical(pick_ban)
    #use_rnn = data["use_rnn"]
    use_rnn = False
    use_max_prob = data["use_max_prob"]
    allow_duplicates = data["allow_duplicates"]
    hero = 10
    out = HEROES[str(predict_last_pick(app.basicNN, full_input=input_))]["localized_name"]
    print(out)
    success = (use_rnn or len(pick_ban) == 19)
    message = "" if success else "Must select all first 19 picks/bans if not using Recurrent Neural Net"
    print(app.basicNN)
    pick_ban_out = []
    out = {"success": success, "message": message, "pick_ban": pick_ban_out}
    return jsonify(out)
