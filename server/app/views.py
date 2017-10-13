from flask import request
from flask import jsonify

from app import app


@app.route('/')
def display():
    print(app.basicNN)
    data = request.json
    pick_ban = data["pick_ban"]
    use_rnn = data["use_rnn"]
    hero = 10
    success = (use_rnn or len(pick_ban) == 19)
    message = "" if success else "Must select all first 19 picks/bans if not using Recurrent Neural Net"
    print(app.basicNN)
    pick_ban_out = []
    out = {"success": success, "message": message, "pick_ban": pick_ban_out}
    return jsonify(out)
