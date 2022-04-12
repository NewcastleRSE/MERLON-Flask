from .utilities import forecasting as fc
import numpy as np
from .metadata import definitions, defaults

store = "."

def interpolate(zeropoint, input, window, instep=30, outstep=1):
    data = [zeropoint] + input
    input_times = range(0, len(data)*instep, instep)
    output_times = range(outstep,(window+1)*outstep,outstep)
    return np.interp(output_times, input_times, data)

def forecast(data, debug=None):
    metadata = { 
        'Site': data['Site'], 
        'period': (data['horizon'] * definitions[data['horizonType']]),
        'step': (data['isp'] * definitions[data['ispType']])
    }

    metadata['window'] = metadata['period'] // metadata['step']

    load = loadforecast(data['HistoricLoad'], metadata)
    generation = generationforecast(data['HistoricGeneration'], metadata, debug)

    return {
        'Site': data['Site'],
        'ForecastStart': data['ForecastPeriodStart'],
        'Forecast': {
            'Load': load,
            'Generation': generation
        }
    }

def loadforecast(data, metadata, debug=None):
    debugging = False

    if isinstance(debug, dict):
        debugging = True
        localdebug = {}

    lags, window = getDefaults(metadata['Site'], 'load')

    result = {}

    for bus in data.keys():
        model, scalers = fc.getTrainedLoadModel(metadata['Site'], bus, store,lags, window, debug=debug if debug is None else localdebug)
        d = data[bus].copy()

        d.sort(key=lambda e: e['Date'])
        inL = np.array([d[l]['Load'] for l in lags]).reshape(1,-1)
        outL = scalers['output'].inverse_transform(model.predict({'main_input': scalers['input'].transform(inL)})).reshape(-1).tolist()

        # model produces a 24-hour-ahead half-hourly forecast
        # if the request is for anything else, interpolate the results:
        if metadata['window'] != window or metadata['step'] != 30:
            outL = interpolate(float(d[-1]['Load']), outL, metadata['window'], outstep=metadata['step'])

        result[bus] = [{'interval': i+1, 'load': str(l)} for i,l in enumerate(outL)]

        if debugging:
            localdebug[bus] = {}
            localdebug[bus]['inL'] = inL
            localdebug[bus]['outL'] = outL
    
    if debugging:
        debug['loadforecast'] = localdebug

    return result

def generationforecast(data, metadata, debug=None):
    debugging = False

    if isinstance(debug, dict):
        debugging = True
        localdebug = {}

    lags, window = getDefaults(metadata['Site'], 'generation')


    result = {}
    for bus in data.keys():
        if debugging:
            localdebug[bus] = {}
        
        model, scalers = fc.getTrainedGenerationModel(metadata['Site'], bus, store, lags, window, debug=localdebug[bus] if debugging else None)
        d = data[bus].copy()

        d.sort(key=lambda e: e['Date'])
        inW = np.array([d[l]['Temperature'] for l in lags]).reshape(1,-1)
        inWs = scalers['inW'].transform(inW).reshape(1,1,-1)
        inG = np.array([d[l]['Generation'] for l in lags]).reshape(1,-1)
        inGs = scalers['inG'].transform(inG).reshape(1,1,-1)
        indict = {'load_input': inGs, 'temp_input': inWs}
        
        outGs = model.predict(indict)
        outG = scalers['outG'].inverse_transform(outGs).reshape(-1).tolist()

        # model produces a 24-hour-ahead half-hourly forecast
        # if the request is for anything else, interpolate the results:
        if metadata['window'] != window or metadata['step'] != 30:
            outG = interpolate(float(d[-1]['Generation']), outG, metadata['window'], outstep=metadata['step'])
            

        result[bus] = [{'interval': i+1, 'generation': str(g) if g > 0 else '0'} for i,g in enumerate(outG)]

        if debugging:
            localdebug[bus]['inG'] = inG
            localdebug[bus]['inW'] = inW
            localdebug[bus]['outL'] = outG

    if debugging:
        debug['generationforecast'] = localdebug

    return result

def train(data, debug=None):
    site = data['Site']
    trainers = data['TrainingRequired']

    if "Generation" in trainers:
        for bus in defaults[site]['busses']['genbuses']:
            if bus in data['data']:
                fc.getTrainedGenerationModel(site, bus, store=store, rawdata=data['data'][bus])

    if "Load" in trainers:
        for bus in defaults[site]['busses']['loadbuses']:
            if bus in data['data']:
                fc.getTrainedLoadModel(site, bus, store=store, rawdata=data['data'][bus])

    import os
    from datetime import date
    import json
    datapath = os.path.join(store, 'forecast', 'training', site, date.today().isoformat())
    os.makedirs(datapath, exist_ok=True)

    with open(os.path.join(datapath, 'training.json'), "w") as f:
        json.dump(data['data'], f)

    return {"status": "OK", "message": "Forecast Training Complete"}

def getDefaults(site, forecast):
    return defaults[site]['forecasting'][forecast]['lags'], defaults[site]['forecasting'][forecast]['window']