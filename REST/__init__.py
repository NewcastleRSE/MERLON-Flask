import os
from flask import Flask, request
from . import schedule, forecast

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY='dev',)

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    
    os.makedirs(app.instance_path, exist_ok=True)

    forecast.store = app.instance_path
    
    
    # usage: POST https://host:port/api/schedule?api_key=<SAMPLE API KEY HERE>
    #        with appropriate request json in body (see samples)
    @app.route("/api/<action>", methods=['POST'])
    def do(action):
        # check key against registered api keys (sample code only)
        key = request.args.get('api_key')
        if key != 'dj75sp1$-':
            return {'message': 'unknown api key provided'}, 401

        # check request has been sent with data
        if not request.is_json:
            return {'message': 'incorrect request data format supplied'}, 400
        # determine which action is required

        # prep the parameters and return dict to pass to modules
        if action == 'forecast':
            return forecast.forecast(request.json)
        # dispatch body to scheduling or forecasting module
        elif action == 'schedule':
            try:
                result = schedule.schedule(request.json)
            except ValueError:
                result =  {'message': 'You have provided an inappropriate value in your scheduling request.'}, 400
            except:
                result = {'message': 'An error occurred processing your request. Please report this to the service adminstrators.'}, 500
            finally:
                return result
        elif action == 'train':
            return forecast.train(request.json)

    return app