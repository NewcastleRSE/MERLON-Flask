from .utilities import scheduling as sc
from .metadata import defaults, definitions

def schedule(data):
# this data set up step should be common to all schedules:
    import datetime as dt
    site = data['Site']
    scenario = data['scenario']

    meta = defaults[site]['scheduling']

    # check this scenario is valid for this site (e.g. at-strem does not support islanding)
    if scenario not in meta['scenarios']:
        raise ValueError

    # set up inputs to scheduling problem: 
    busses = defaults[site]['busses']
    window = (data['horizon'] * definitions[data['horizonType']]) // (data['isp'] * definitions[data['ispType']])
    steplength = data['isp']/(60 if data['ispType'] == 'minutes' else 1)
    
    batt_ini = data['Battery']['ChargeState'] * meta['batt_cap'] #assumes battery charge as a fraction of full
    load = sc.formatLoadForecast(data['Forecast']['Load'], window, steplength, busses['loadbuses'])
    prod = sc.formatGenerationForecast(data['Forecast']['Generation'], window, steplength, busses['genbuses'])

    # flex_price will come back filled with 0 if the flexibility cost is not provided
    # even if flex costing is provided, it will be ignored during model evaluation if scenario is not 'market'
    flex_up, flex_down, flex_price = sc.formatFlexibilityData(data['Flexibility'], window, steplength, busses['flexbuses'])

    if scenario == 'market':
        ele_price = sc.formatPriceData(data['Pricing'], window)[0]
    else:
        ele_price = None

    result = sc.buildAndOptimiseModel(
        site, 
        scenario, 
        window, 
        steplength, 
        load, 
        prod, 
        flex_up, 
        flex_down, 
        flex_price,
        ele_price,
        batt_ini,
        meta,
        busses
    )

    return {
        'ScheduleStartDate': data['SchedulePeriodStart'],
        'horizon': data['horizon'],
        'horizonType': data['horizonType'],
        'isp': data['isp'],
        'ispType': data['ispType'],
        'Schedule': result
    }