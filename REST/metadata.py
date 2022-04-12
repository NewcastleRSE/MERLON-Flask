import numpy as np

definitions = {
    'minutes': 1,
    'hours': 60
}

defaults = {
    'at-strem': {
        'scheduling': {
            'scenarios': ['constraint', 'market'],
            'batt_cap': 0.25,
            'batt_pow': 0.25,
            'nch': 0.95,
            'ndis': 0.95,
            'fr': {
                'start': (0,0,0),
                'end': (6,0,0),
                'min': 0.4 * 0.25,
                'max': 0.6 * 0.25
            }
        },
        'forecasting': {
            'generation': {
                'lags': np.array([0,4,5,23,45,46,47]),
                'window': 48
            },
            'load': {
                'lags': np.array([0,6,15,31,43,44,45,46,47]),
                'window': 48
            }
        },
        'busses': {
            'loadbuses': ['T241','T261','T265','T264','T262'],
            'genbuses': ['T241','T261','T267','T265','T266','T264','T262'],
            'flexbuses': ['T241','T261', 'T265', 'T264','T262']
        }
    },
    'es-crevillent': {
        'scheduling': {
            'scenarios': ['constraint', 'islanding', 'market'],
            'batt_cap': 0.242,
            'batt_pow': 0.1,
            'nch': 0.95,
            'ndis': 0.95,
            'fr': {
                'start': (0,0,0),
                'end': (6,0,0),
                'min': 0.4 * 0.242,
                'max': 0.6 * 0.242
            }
        },
        'forecasting': {
            'generation': {
                'lags': np.array([0,4,5,23,45,46,47]),
                'window': 48
            },
            'load': {
                'lags': np.array([0,6,15,31,43,44,45,46,47]),
                'window': 48
            }
        },
        'busses': {
            'loadbuses': ['bus2'],
            'genbuses': ['bus2'],
            'flexbuses': ['bus2']
        }
    }
}
