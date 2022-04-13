def getGenerationModel(inputshape, wdw, store):
    """Creates an ANN using the LTSM architecture"""

    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, Flatten, concatenate, Permute, multiply
    from tensorflow.keras import Input

    import os
    shapeforpath = "-".join([str(x) for x in inputshape])
    modelpath = os.path.join(store, 'forecast', 'models', 'generic', shapeforpath, str(wdw), 'generation.tf')

    
    if os.path.exists(modelpath):
        return load_model(modelpath)
    

    model_in1 = Input(shape=inputshape, name='load_input')
    model_hid1 = LSTM(32, return_sequences=True)(model_in1)
    model_hid11 = Permute((2,1), name='Permute_x11')(model_hid1)
    model_hid11 = Dense(inputshape[0], activation='softmax', name='dense_x1')(model_hid11)
    model_hid12 = Permute((2,1), name='Permute_x12')(model_hid11)
    model_x1_out = multiply([model_hid1, model_hid12], name='merge_x1')
    model_x1_out = Flatten(name='Output_x1')(model_x1_out)
    
    model_in2 = Input(shape=inputshape, name='temp_input')
    model_hid2 = LSTM(32, return_sequences=True)(model_in2)
    model_hid21 = Permute((2,1), name='Permute_x21')(model_hid2)
    model_hid21 = Dense(inputshape[0], activation='softmax', name='dense_x2')(model_hid21)
    model_hid22 = Permute((2,1), name='Permute_x22')(model_hid21)
    model_x2_out = multiply([model_hid2, model_hid22], name='merge_x2')
    model_x2_out = Flatten(name='Output_x2')(model_x2_out)
    
    model_conc = concatenate([model_x1_out, model_x2_out], axis=1)

    model_hid3 = Dense(512, activation='relu')(model_conc)
    model_out = Dense(wdw, name='main_output')(model_hid3)
    
    model = Model(inputs=[model_in1, model_in2], outputs=[model_out])
    
    model.compile(loss='mse', optimizer='RMSProp', metrics=['mae', 'msle', 'mse'])
    model.save(modelpath, save_format='tf')

    return model

def getLoadModel(inputshape, wdw, store):
    """Creates an ANN using a Dense architecture"""

    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout

    import os
    shapeforpath = "-".join([str(x) for x in inputshape])
    modelpath = os.path.join(store, 'forecast', 'models', 'generic', shapeforpath, str(wdw), 'load.tf')

    if os.path.exists(modelpath):
        return load_model(modelpath)

    # Initialize the constructor
    model = Sequential()
    # Add an input layer
    model.add(Dense(256, activation='relu', input_shape=inputshape, name="main"))
    # Add first hidden layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(wdw, name="main_output"))
    model.compile(loss='mse', optimizer='Nadam', metrics=['mae', 'msle', 'mse'])
    model.save(modelpath, save_format='tf')
    return model

def getTrainedGenerationModel(site, bus, store=".", lags=None, window=None, rawdata=None, debug=None):
    from tensorflow.keras.models import load_model
    import os
    import pickle

    debugging = False

    if isinstance(debug, dict):
        debugging = True
        localdebug = {}
    
    defaults = getForecastDefaults(site)['generation']
    if lags is None:
        lags = defaults['lags']
    if window is None:
        window = defaults['window']
    
    shapeforpath = "-".join([str(x) for x in lags.shape])

    modelpath = os.path.join(store, 'forecast', 'models', site, bus, shapeforpath, str(window), 'generation.tf')
    scalerpath = os.path.join(store, 'forecast', 'scalers', site, bus, shapeforpath, str(window), 'generation.pkl')
    
    if os.path.exists(modelpath) and rawdata is None:
        model = load_model(modelpath)
        with open(scalerpath, "rb") as f:
            scalers = pickle.load(f)

        return model, scalers
    else:
        # ensure directory creation for saving model/scalers later
        os.makedirs(os.path.dirname(modelpath), exist_ok=True)
        os.makedirs(os.path.dirname(scalerpath), exist_ok=True)
    
    # no trained model, so we need to format & scale data and train the model
    if rawdata is None:
        import json
        # site training data should be: a json-dumped list of dicts containing:
        # Yr, Mt, Dy, Hr, Mn (date parts), <Bus1> ... <BusN> (by name, generation on each bus), Temp (erature)
        with open(os.path.join(store, 'forecast', 'data', site, 'generation', 'training.json'), "r") as f:
            trainingdata = json.load(f)

            rawdata = [ {
                'Date': extractdate(d),
                'Generation': d[bus],
                'Temperature': d['Temp']
            } for d in trainingdata]
 
    model = getGenerationModel(lags.reshape(1,-1).shape, window, store)
    data = formatGenerationTrainingData(rawdata, lags, window)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    scalers = {}
    scalers['inW'] = StandardScaler().fit(data['inW'])
    inWs = scalers['inW'].transform(data['inW'])
    scalers['inG'] = StandardScaler().fit(data['inG'])
    inGs = scalers['inG'].transform(data['inG'])
    scalers['outG'] = StandardScaler().fit(data['outG'])
    outGs = scalers['outG'].transform(data['outG'])
    
    import numpy as np
    inWtrain, inWtest, inGtrain, inGtest, outGtrain, outGtest = train_test_split(inWs, inGs, outGs, test_size=0.1, random_state=42)
    inTdict = { 'load_input': np.array(inGtrain).reshape(inGtrain.shape[0], 1,-1), 'temp_input': np.array(inWtrain).reshape(inWtrain.shape[0],1,-1)}
    outTdict = { 'main_output': outGtrain }
    inVdict = {'load_input': np.array(inGtest).reshape(inGtest.shape[0], 1,-1), 'temp_input': np.array(inWtest).reshape(inWtest.shape[0], 1,-1) }
    outVdict = {'main_output': outGtest}
    
    model.fit(inTdict, outTdict, epochs=50, batch_size=200, verbose=0, 
        validation_data=(inVdict, outVdict))
    
    model.save(modelpath)
    with open(scalerpath, "wb") as pkl:
        pickle.dump(scalers, pkl)
    
    if debugging:
        localdebug['rawdata'] = rawdata
        localdebug['testraindata'] = data
        debug['getTrainedGenerationModel'] = localdebug
    
    return model, scalers

def getTrainedLoadModel(site, bus, store=".", lags=None, window=None, rawdata=None, debug=None):
    from tensorflow.keras.models import load_model
    import os
    import pickle

    defaults = getForecastDefaults(site)['load']
    if lags is None:
        lags = defaults['lags']
    if window is None:
        window = defaults['window']

    shapeforpath = "-".join([str(x) for x in lags.shape])

    modelpath = os.path.join(store, 'forecast', 'models', site, bus, shapeforpath, str(window), 'load.tf')
    scalerpath = os.path.join(store, 'forecast', 'scalers', site, bus, shapeforpath, str(window), 'load.pkl')

    if os.path.exists(modelpath) and rawdata is None:
        model = load_model(modelpath)
        with open(scalerpath, "rb") as f:
            scalers = pickle.load(f)

        return model, scalers
    else:
        # ensure directory creation for saving model/scalers later
        os.makedirs(os.path.dirname(modelpath), exist_ok=True)
        os.makedirs(os.path.dirname(scalerpath), exist_ok=True)
    
    # no trained model, so we need to format & scale data and train the model
    # training data is currently stored as a json-dumped list of dicts, containing:
    # Yr, Mt, Dy, Hr, Mn (date/time details), <Bus1> ... <BusN> (by name, load on each bus)
    if rawdata is None:
        import json
        with open(os.path.join(store, 'forecast', 'data', site, 'load', 'training.json'), "r") as f:
            trainingdata = json.load(f)

            rawdata = [ {
                'Date': extractdate(d),
                'Load': d[bus],
            } for d in trainingdata]

    model = getLoadModel(lags.shape, window, store)
    data = formatLoadTrainingData(rawdata, lags, window)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    scalers = {}
    scalers['input'] = StandardScaler().fit(data['inL'])
    scalers['output'] = StandardScaler().fit(data['outL'])

    inLtrain, inLtest, outLtrain, outLtest = train_test_split(scalers['input'].transform(data['inL']), scalers['output'].transform(data['outL']), test_size=0.2, random_state=42)
    model.fit({'main_input': inLtrain,}, {'main_output': outLtrain }, epochs=50, batch_size=200, verbose=0, validation_data= ({'main_input': inLtest}, {'main_output': outLtest}))
    model.save(modelpath)
    with open(scalerpath, "wb") as pkl:
        pickle.dump(scalers, pkl)

    return model, scalers

def getForecastDefaults(site):
    from ..metadata import defaults
    return defaults[site]['forecasting']

def formatGenerationTrainingData(rawdata, lags, window):
    # 13-04-2022: add handlers for non-sequential data - allows training to data to have gaps.
    # n.b. gaps in data smaller than 2* targetdays will 
    
    from datetime import date

    def checkgap(date1, date2, targetdays):
        return abs((date.fromisoformat(date1) - date.fromisoformat(date2)).days) == targetdays
    
    # assumes rawdata is a list drawn from json in the forecastRequest format, approx. format
    # { Date: '...', 'Temperature': 11.5, 'Generation': 35.2 }
    # assume Radiation is not available (as per 28/08/2020)
    rawdata.sort(key=lambda e: e['Date'])
    Gkey = 'Generation'
    Wkey = 'Temperature'
    samples = len(rawdata)-window*2
    
    results = {}
    results['inW'] = [[rawdata[x+l][Wkey] for l in lags] for x in range(samples) if checkgap(rawdata[x]['Date'][:10], rawdata[x+window]['Date'][:10], 1)]
    results['inG'] = [[rawdata[x+l][Gkey] for l in lags] for x in range(samples) if checkgap(rawdata[x]['Date'][:10], rawdata[x+window]['Date'][:10], 1)]
    results['outG'] = [[rawdata[x+w][Gkey] for w in range(window)] for x in range(window,samples+window) if checkgap(rawdata[x]['Date'][:10], rawdata[x-window]['Date'][:10], 1)]
    return results

def formatLoadTrainingData(rawdata, lags, window):
    # assumes rawdata is a list of historic load, drawn from json in the forecastRequest format, approx:
    # { Date: '...', 'Load': 34.7 }
    rawdata.sort(key=lambda e: e['Date'])
    samples = len(rawdata) - window*2
    results = {}
    results['inL'] = [[rawdata[x+l]['Load'] for l in lags] for x in range(samples)]
    results['outL'] = [[rawdata[x+w]['Load'] for w in range(window)] for x in range(window,samples+window)]
    return results

def extractdate(record):
    return "T".join(["-".join(["{:0>4}".format(record['Yr']), "{:0>2}".format(record['Mt']), "{:0>2}".format(record['Dy'])]), ":".join(["{:0>2}".format(record['Hr']), "{:0>2}".format(record['Mn']), "00"])])
