from flask import Flask

from board import pages

# python -m flask --app ".\data_processing\data_structures\Flask\01_intro\board" run --port 8000 --debug

def create_app():
    app = Flask(__name__)

    app.register_blueprint(pages.bp)
    return app

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "Hello, World!"

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8000, debug=True)

    