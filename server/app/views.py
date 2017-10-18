from flask import request
from flask import jsonify

from app import app
from main import input_ids_to_categorical, HEROES, predict_last_pick, HEROES_INVERT, generate_draft_rest


@app.route('/', methods=["POST"])
def display():
    success = True
    message = ""
    data = request.get_json()

    pick_ban = [pb for pb in data["pickbans"] if pb]  # get rid of empty strings
    try:
        pickban_ids = [HEROES_INVERT[name] for name in pick_ban]
        use_rnn = data.get("use_rnn", True)
        use_max = data.get("use_max", False)
        allow_dupes = data.get("allow_dupes", False)
        if not use_rnn and len(pick_ban) != 19:
            success = False
            message = "Must select all first 19 picks/bans if not using Recurrent Neural Net"
        if success:
            if use_rnn:
                pick_ban = [HEROES[pb] for pb in generate_draft_rest(
                    app.rnn, pickban_ids, pick_max=use_max, allow_duplicates=allow_dupes
                )]

            else:
                input_ = input_ids_to_categorical(pickban_ids)
                last_pick_id = predict_last_pick(
                    app.basicNN, full_input=input_, pick_max=use_max, allow_duplicates=allow_dupes
                )
                pick_ban.append(HEROES[last_pick_id])
    except KeyError as e:
        success = False
        message = "Incorrect hero name: %s" % e.args[0]

    out = {"success": success, "message": message, "pick_ban": pick_ban}
    return jsonify(out)
