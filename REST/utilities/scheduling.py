def getAdmittanceMatrix(site):
    if site != 'at-strem':
        raise ValueError()

    import numpy as np
    # Load network data
    # inputs of overhead transmission lines (6 types in total), these data are provide by the pilot site partners. The per-unit system is adopted here. The base parameters are shown as follows:
    # base power 1MW
    # base voltage 20kV
    # base current 50A
    # base impedance 400 ohms
    # base admittance 2.5 mS

    # resistance and reactance for each type of line
    para={"R_OH150": (0.2069)/400,
        "X_OH150": (0.3602)/400,
        "R_OH50": (0.6296)/400,
        "X_OH50": (0.3956)/400,
        "R_C240": (0.13)/400,
        "X_C240": (0.113)/400,
        "R_C95": (0.323)/400,
        "X_C95": (0.132)/400,
        "R_C50": (0.39)/400,
        "X_C50": (0.146)/400,
        "R_C35": (0.527)/400,
        "X_C35": (0.153)/400,
        }

    line={"s0100": (para["R_C240"]+1j*para["X_C240"])*0.12,
        "s0200": (para["R_C240"]+1j*para["X_C240"])*0.354,
        "s0201": (para["R_OH150"]+1j*para["X_OH150"])*0.948,
        "b0100": (para["R_C50"]+1j*para["X_C50"])*0.84,
        "b02001":(para["R_C50"]+1j*para["X_C50"])*0.195,
        "b02002":(para["R_OH50"]+1j*para["X_OH50"])*0.128,
        "b02003":(para["R_OH50"]+1j*para["X_OH50"])*0.73,
        "b02004":(para["R_OH50"]+1j*para["X_OH50"])*1.443,
        "b03001":(para["R_OH50"]+1j*para["X_OH50"])*0.18,
        "b03002":(para["R_OH50"]+1j*para["X_OH50"])*0.207,
        "b03003": (para["R_C95"]+1j*para["R_C95"])*0.112,
        "b3101": (para["R_OH50"]+1j*para["X_OH50"])*0.076,
        "b3100": (para["R_OH50"]+1j*para["X_OH50"])*0.274,
        
        "b03a001": (para["R_C95"]+1j*para["R_C95"])*0.132,
        "b03a002": (para["R_OH50"]+1j*para["X_OH50"])*0.747,
        "b04001": (para["R_OH50"]+1j*para["X_OH50"])*1.323,
        "b04002": (para["R_OH50"]+1j*para["X_OH50"])*0.948,
        "b05001": (para["R_OH50"]+1j*para["X_OH50"])*0.145,
        "b05002": (para["R_OH50"]+1j*para["X_OH50"])*0.491,
        
        "b3301": (para["R_OH50"]+1j*para["X_OH50"])*0.664,
        "b3300": (para["R_C95"]+1j*para["R_C95"])*0.32,
        
        "b4100": (para["R_OH50"]+1j*para["X_OH50"])*0.27,
        
        "b42001": (para["R_OH50"]+1j*para["X_OH50"])*0.635,
        
        "b42002": (para["R_C95"]+1j*para["R_C95"])*0.54,
        "b42003": (para["R_C95"]+1j*para["R_C95"])*1.1,
        
        "b0600": (para["R_C95"]+1j*para["R_C95"])*0.614,
        
        "b07001": (para["R_C95"]+1j*para["R_C95"])*0.614,
        
        "b07002": (para["R_C95"]+1j*para["R_C95"])*0.04,
        
        "b07003": (para["R_C95"]+1j*para["R_C95"])*0.04,
        
        "b0202": (para["R_OH50"]+1j*para["X_OH50"])*0.74,
        "b0901": (para["R_OH50"]+1j*para["X_OH50"])*0.08,
        "b08001": (para["R_C95"]+1j*para["R_C95"])*1.897,
        
        "b08002": (para["R_C95"]+1j*para["R_C95"])*1.9,
        
        "b08003": (para["R_C95"]+1j*para["R_C95"])*2.051,
        
        "b08004": (para["R_OH50"]+1j*para["X_OH50"])*0.542,
        "b0801": (para["R_C95"]+1j*para["R_C95"])*0.115,
        
        "b0900": (para["R_OH50"]+1j*para["X_OH50"])*0.475,
        "b1000": (para["R_OH50"]+1j*para["X_OH50"])*0.443,
        "b11001": (para["R_OH50"]+1j*para["X_OH50"])*0.517,
        "b11011": (para["R_OH50"]+1j*para["X_OH50"])*0.777,
        
        "b11002": (para["R_OH50"]+1j*para["X_OH50"])*0.426,
        
        "b11012": (para["R_OH50"]+1j*para["X_OH50"])*1.7,
        "b11013": (para["R_OH50"]+1j*para["X_OH50"])*0.55,
        "b12001": (para["R_OH50"]+1j*para["X_OH50"])*0.985,
        "b1201": (para["R_OH50"]+1j*para["X_OH50"])*0.961,
        "b12002": (para["R_OH50"]+1j*para["X_OH50"])*0.253,
        
        "b1103": (para["R_OH50"]+1j*para["X_OH50"])*0.23,
        "b11300": (para["R_OH50"]+1j*para["X_OH50"])*1.05,
        
        "b1102": (para["R_OH50"]+1j*para["X_OH50"])*0.057,
        }


    Y={"branch1_2": 1/(line["s0100"]+line["s0200"]+line["s0201"]+line["b0100"]+line["b02001"]+line["b02002"]+line["b02003"]+line["b02004"]+line["b03001"]+line["b03002"]+line["b03003"]),
            "branch2_3": 1/(line["b03a001"]+line["b03a002"]+line["b04001"]+line["b04002"]+line["b42001"]),
            "branch3_4": 1/(line["b42003"]+line["b42002"]),
            "branch4_5": 1/(line["b07003"]),
            "branch5_6": 1/(line["b07002"]),
            "branch6_7": 1/(line["b07001"]),
            "branch7_8": 1/(line["b0600"]),
            "branch2_8": 1/(line["b03a001"]+line["b03a002"]+line["b04001"]+line["b04002"]+line["b05001"]+line["b05002"]),
            
            }


    # construct the admittance matrix, it is done manually here
    Y_matrix=np.zeros([8,8],dtype=complex)

    Y_matrix[0,0]=Y["branch1_2"]

    Y_matrix[1,1]=Y["branch1_2"]+Y["branch2_3"]+Y["branch2_8"]
    Y_matrix[2,2]=Y["branch2_3"]+Y["branch3_4"]
    Y_matrix[3,3]=Y["branch3_4"]
    Y_matrix[4,4]=Y["branch5_6"]

    Y_matrix[5,5]=Y["branch5_6"]+Y["branch6_7"]
    Y_matrix[6,6]=Y["branch6_7"]+Y["branch7_8"]
    Y_matrix[7,7]=Y["branch7_8"]+Y["branch2_8"]

    Y_matrix[0,1]=-Y["branch1_2"]
    Y_matrix[1,0]=-Y["branch1_2"]

    Y_matrix[1,2]=-Y["branch2_3"]
    Y_matrix[2,1]=-Y["branch2_3"]

    Y_matrix[2,3]=-Y["branch3_4"]
    Y_matrix[3,2]=-Y["branch3_4"]

    Y_matrix[4,5]=-Y["branch5_6"]
    Y_matrix[5,4]=-Y["branch5_6"]

    Y_matrix[5,6]=-Y["branch6_7"]
    Y_matrix[6,5]=-Y["branch6_7"]

    Y_matrix[6,7]=-Y["branch7_8"]
    Y_matrix[7,6]=-Y["branch7_8"]

    Y_matrix[1,7]=-Y["branch2_8"]
    Y_matrix[7,1]=-Y["branch2_8"]

    #G=np.real(Y_matrix)
    #B=np.imag(Y_matrix)

    return np.real(Y_matrix), np.imag(Y_matrix) # G, B

def formatLoadForecast(data, steps, length, buses):
    return formatData(data, steps, length, buses, 'load', lambda v: float(v)*length*1e-3)

def formatGenerationForecast(data, steps, length, buses):
    return formatData(data, steps, length, buses, 'generation', lambda v: float(v)*length*1e-3)

def formatFlexibilityData(data, steps, length, buses):
    flexup = formatData(data, steps, length, buses, 'upwards', lambda v: float(v)*length*1e-3)
    flexdown = formatData(data,steps,length,buses,'downwards', lambda v: float(v)*length*1e-3)
    flexprice = formatData(data,steps,length,buses,'cost',lambda v: float(v))

    return flexup, flexdown, flexprice

def formatPriceData(data, steps, length):
    return formatData(data, steps, length, ['Data'], 'price', lambda v: float(v))

def formatData(data, steps, length, buses, valuekey, converter=lambda v: v):
    for bus in buses:
        if bus not in data:
            data[bus] = [{ "interval": i, valuekey: 0} for i in range(steps)]
        
        if len(data[bus]) < steps:
            raise ValueError

        data[bus].sort(key=lambda e: e['interval'])
    
    outdata = [[converter(data[bus][i][valuekey]) for i in range(steps)] for bus in buses]
    return outdata

def applyFrequencyResponse(lower, upper, settings, startDate, steps, steplength):
    import datetime as dt
    sh, sm, ss = settings['start']
    startTime = dt.time(sh,sm,ss)
    eh,em,es = settings['end']
    endTime = dt.time(eh,em,es)
    timestep = dt.timedelta(hours=steplength)

    if startTime > endTime:
        predicate = lambda t: not endTime <= (startDate + (timestep*t)).time() < startTime
    else:
        predicate = lambda t: startTime <= (startDate + (timestep*t)).time() < endTime

    for s in range(steps):
        if predicate(s):
            upper[0,s] = settings['max']
            lower[0,s] = settings['min']
    


