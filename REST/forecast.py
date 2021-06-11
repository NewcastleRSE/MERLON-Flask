from .utilities import forecasting as fc
import numpy as np

defaults = {
    'at-strem': {
        'generation': {
            'lags': np.array([0,4,5,23,45,46,47]),
            'window': 48
        },
        'load': {
            'lags': np.array([0,6,15,31,43,44,45,46,47]),
            'window': 48
        }
    }
}

definitions = {
    'minutes': 1,
    'hours': 60
}

store = "."

def forecast(data, debug=None):
    metadata = { 
        'Site': data['Site'], 
        'window': (data['horizon'] * definitions[data['horizonType']]) // (data['isp'] * definitions[data['ispType']])
    }

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
    if window != metadata['window']:
        raise ValueError()

    result = {}

    for bus in data.keys():
        model, scalers = fc.getTrainedLoadModel(metadata['Site'], bus, store,lags, window, debug=debug if debug is None else localdebug)
        d = data[bus].copy()

        d.sort(key=lambda e: e['Date'])
        inL = np.array([d[l]['Load'] for l in lags]).reshape(1,-1)
        outL = scalers['output'].inverse_transform(model.predict({'main_input': scalers['input'].transform(inL)})) 
        result[bus] = [{'interval': i+1, 'load': str(l)} for i,l in enumerate(outL.reshape(-1))]

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
    if window != metadata['window']:
        raise ValueError()

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
        outG = scalers['outG'].inverse_transform(outGs)

        result[bus] = [{'interval': i+1, 'generation': str(g) if g > 0 else '0'} for i,g in enumerate(outG.reshape(-1))]

        if debugging:
            localdebug[bus]['inG'] = inG
            localdebug[bus]['inW'] = inW
            localdebug[bus]['outL'] = outG

    if debugging:
        debug['generationforecast'] = localdebug

    return result

def getDefaults(site, forecast):
    return defaults[site][forecast]['lags'], defaults[site][forecast]['window']