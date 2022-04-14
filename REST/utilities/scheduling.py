def getAdmittanceMatrix(site):
    if site == 'at-strem':

        # Load network data
        # inputs of overhead transmission lines (6 types in total), these data are provide by the pilot site partners. The per-unit system is adopted here. The base parameters are shown as follows:
        # base power 1MW
        # base voltage 20kV
        # base current 50A
        # base impedance 400 ohms
        # base admittance 2.5 mS

        # resistance and reactance for each type of line
        para={
            "R_OH150": (0.2069)/400,
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


        return {
            "branch1_2": 1/(line["s0100"]+line["s0200"]+line["s0201"]+line["b0100"]+line["b02001"]+line["b02002"]+line["b02003"]+line["b02004"]+line["b03001"]+line["b03002"]+line["b03003"]),
            "branch2_3": 1/(line["b03a001"]+line["b03a002"]+line["b04001"]+line["b04002"]+line["b42001"]),
            "branch3_4": 1/(line["b42003"]+line["b42002"]),
            "branch4_5": 1/(line["b07003"]),
            "branch5_6": 1/(line["b07002"]),
            "branch6_7": 1/(line["b07001"]),
            "branch7_8": 1/(line["b0600"]),
            "branch2_8": 1/(line["b03a001"]+line["b03a002"]+line["b04001"]+line["b04002"]+line["b05001"]+line["b05002"]),
        }

    if site == "es-crevillent":
        return {"branch1_2": ((0.115+0.103j)/0.16),}

    raise ValueError

def formatLoadForecast(data, steps, length, buses):
    return formatData(data, steps, buses, 'load', lambda v: float(v)*length*1e-3)

def formatGenerationForecast(data, steps, length, buses):
    return formatData(data, steps, buses, 'generation', lambda v: float(v)*length*1e-3)

def formatFlexibilityData(data, steps, length, buses):
    flexup = formatData(data, steps, buses, 'upwards', lambda v: float(v)*length*1e-3)
    flexdown = formatData(data, steps, buses,'downwards', lambda v: float(v)*length*1e-3)
    flexprice = formatData(data, steps, buses,'cost', lambda v: float(v))

    return flexup, flexdown, flexprice

def formatPriceData(data, steps):
    return formatData(data, steps, ['Data'], 'price', lambda v: float(v))

def formatData(data, steps, buses, valuekey, converter=lambda v: v):
    for bus in buses:
        if bus not in data:
            data[bus] = [{ "interval": i, valuekey: 0} for i in range(steps)]
        
        if len(data[bus]) < steps:
            raise ValueError

        data[bus].sort(key=lambda e: e['interval'])
    
    outdata = [[converter(data[bus][i].get(valuekey, 0)) for i in range(steps)] for bus in buses]
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
    
def buildAndOptimiseModel(site, scenario, t, steplength, load, prod, flex_up, flex_down, flex_price, ele_price, batt_ini, meta, busses, retry=0):
    from gurobipy import GRB, Model, LinExpr, QuadExpr
    from math import tan, acos

    if scenario == "market" and ele_price is None:
        raise ValueError

    # Reactive power demand (MVar), assuming power factor to be 0.97
    Rload=[[v*tan(acos(0.97)) for v in bus] for bus in load]
    # Reactive power output from generation, assuming power factor to be 0.97
    Rprod=[[v*tan(acos(0.97)) for v in bus] for bus in prod]

    Z = getAdmittanceMatrix(site)

    # various 'magic' values that help set up the next few variables.
    # note that the dictionary here is *immediately* dereferenced by site.
    values = {
        "at-strem": {
            "v_delta": 0.02, # permitted variance of voltage
            "var_length": 7, # base length of model variables
        },
        "es-crevillent": {
            "v_delta": 0.07,
            "var_length": 1,
        },
    }[site]

    # constraining values - adds additional variance if we are retrying the schedule
    v_min = (1-(values['v_delta'] + (0.01*retry)))**2
    v_max = (1+(values['v_delta'] + (0.01*retry)))**2

    # define lower and upper boundary
    v_lb=[[v_min] * t] * (values['var_length']+1)
    v_lb[0][:]=[1]*t
    v_ub=[[v_max] * t] * (values['var_length']+1)
    v_ub[0][:]=[1]*t

    m = Model("socp")

    ### set up the variables that support the model optimization
    # for a result set we need only:
    # Pbcha, Pbdisa, Pbrea, Eb - battery operations
    # Ut_flex_up, Ut_flex_down - utilised flexibility
    # Pact, Prea - grid active and reactive power.

    # ancilary variables
    # v is the system voltage? P, Q as described below. L?
    v=m.addVars(values['var_length']+1,t,lb=v_lb,ub=v_ub,name="v")
    L=m.addVars(values['var_length'],t,lb=0,ub=GRB.INFINITY,name="L")
    # active power flow (MW)
    P=m.addVars(values['var_length'],t,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="P")
    # reactive power flow (MVar)
    Q=m.addVars(values['var_length'],t,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="Q")

    # Pact and Prea are the active and reactive power required from the grid to input to the distribution network (MW)
    Pact=m.addVars(1,t,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="Pact")
    Prea=m.addVars(1,t,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="Prea")

    # curtail energy availability for islanding scenarios:
    
    if scenario == 'islanding':
        prod_curt = m.addVars(values['var_length'], t, lb=0, ub=prod, name="prod_curt")

    # Utilised flexibility (MW)
    Ut_flex_up=m.addVars(len(flex_up),t,lb=0,ub=flex_up,name="Ut_flex_up")
    Ut_flex_down=m.addVars(len(flex_down),t,lb=0,ub=flex_down,name="Ut_flex_down")


    # battery active charging power (MW)
    Pbcha=m.addVars(1,t,lb=0,ub=meta['batt_pow'],name="Pbcha")
    # battery reactive charging/discharging power, positive means charging, negative means discharging (MVar)
    Pbrea=m.addVars(1,t,lb=-meta['batt_pow'],ub=meta['batt_pow'],name="Pbrea")
    # battery active discharging power (MW)
    Pbdisa=m.addVars(1,t,lb=0,ub=meta['batt_pow'],name="Pbdisa")
    # battery binary variable to decide charging and discharging, 1 for charging, 0 for discharging
    B_bin=m.addVars(1,t,lb=0,ub=1,vtype=GRB.BINARY,name="B_bin")

    # battery energy level
    Eb=m.addVars(1,t,lb=[[0] * t],ub=[[meta['batt_cap']]*t],name="Eb")

    expr = QuadExpr(0) if scenario == 'islanding' else LinExpr(0)

####
####

# CHECK ALL ARRAY INDEXERS: prod and load are NOT np.Array or gurobi variables, they're ordinary nested arrays!!!!
# therefore they should be indexed by prod[x][y], not prod[x,y]
####
####

    ## site specific expression building:
    if site == "at-strem":
        # define objective function
        for k in range(t):
            if scenario == 'constraint':
                expr.addTerms(Z["branch1_2"].real,L[0,k])
                expr.addTerms(Z["branch2_3"].real,L[1,k])
                expr.addTerms(Z["branch3_4"].real,L[2,k])
                expr.addTerms(Z["branch5_6"].real,L[3,k])
                expr.addTerms(Z["branch6_7"].real,L[4,k])
                expr.addTerms(Z["branch7_8"].real,L[5,k])
                expr.addTerms(Z["branch2_8"].real,L[6,k])
            else:
                expr.addTerms(ele_price[k]*steplength,Pact[0,k])
                expr.addTerms(flex_price[0][k]*steplength,Ut_flex_up[0,k])
                expr.addTerms(flex_price[1][k]*steplength,Ut_flex_up[1,k])
                expr.addTerms(flex_price[2][k]*steplength,Ut_flex_up[2,k])
                expr.addTerms(flex_price[3][k]*steplength,Ut_flex_up[3,k])
                expr.addTerms(flex_price[4][k]*steplength,Ut_flex_up[4,k])
                
                expr.addTerms(flex_price[0][k]*steplength,Ut_flex_down[0,k])
                expr.addTerms(flex_price[1][k]*steplength,Ut_flex_down[1,k])
                expr.addTerms(flex_price[2][k]*steplength,Ut_flex_down[2,k])
                expr.addTerms(flex_price[3][k]*steplength,Ut_flex_down[3,k])
                expr.addTerms(flex_price[4][k]*steplength,Ut_flex_down[4,k])
                
        m.setObjective(expr, GRB.MINIMIZE)

        for k in range(t):
            # active power flow constraints
            m.addConstr(Pact[0,k]-P[0,k]==0)
            m.addConstr(P[1,k]-P[0,k]+P[6,k]+Z["branch1_2"].real*L[0,k]-prod[0][k]+Ut_flex_up[0,k]-Ut_flex_down[0,k]==-load[0][k])
            m.addConstr(P[2,k]-P[1,k]+Z["branch2_3"].real*L[1,k]-prod[1][k]+Ut_flex_up[1,k]-Ut_flex_down[1,k]==-load[1][k])
            m.addConstr(-P[2,k]+Z["branch3_4"].real*L[2,k]-prod[2][k]==0)
            m.addConstr(-P[3,k]+Z["branch5_6"].real*L[3,k]-prod[3][k]+Ut_flex_up[2,k]-Ut_flex_down[2,k]==-load[2][k])
            m.addConstr(P[3,k]-P[4,k]+Z["branch6_7"].real*L[4,k]-prod[4][k]==0)
            m.addConstr(P[4,k]-P[5,k]+Z["branch7_8"].real*L[5,k]-prod[5][k]+Pbcha[0,k]-Pbdisa[0,k]+Ut_flex_up[3,k]-Ut_flex_down[3,k]==-load[3][k])
            m.addConstr(P[5,k]-P[6,k]+Z["branch2_8"].real*L[6,k]-prod[6][k]+Ut_flex_up[4,k]-Ut_flex_down[4,k]==-load[4][k])
            
            # reactive power flow constraints
            m.addConstr(Prea[0,k]-Q[0,k]==0)
            m.addConstr(Q[1,k]-Q[0,k]+Q[6,k]+Z["branch1_2"].imag*L[0,k]-Rprod[0][k]==-Rload[0][k])
            m.addConstr(Q[2,k]-Q[1,k]+Z["branch2_3"].imag*L[1,k]-Rprod[1][k]==-Rload[1][k])
            m.addConstr(-Q[2,k]+Z["branch3_4"].imag*L[2,k]-Rprod[2][k]==0)
            m.addConstr(-Q[3,k]+Z["branch5_6"].imag*L[3,k]-Rprod[3][k]==-Rload[2][k])
            m.addConstr(Q[3,k]-Q[4,k]+Z["branch6_7"].imag*L[4,k]-Rprod[4][k]==0)
            m.addConstr(Q[4,k]-Q[5,k]+Z["branch7_8"].imag*L[5,k]-Rprod[5][k]+Pbrea[0,k]==-Rload[3][k])
            m.addConstr(Q[5,k]-Q[6,k]+Z["branch2_8"].imag*L[6,k]-Rprod[6][k]==-Rload[4][k])
            
            # voltage constraints
            m.addConstr(v[0,k]-v[1,k]-2*Z["branch1_2"].real*P[0,k]-2*Z["branch1_2"].imag*Q[0,k]+(Z["branch1_2"].real**2+Z["branch1_2"].imag**2)*L[0,k]==0)
            m.addConstr(v[1,k]-v[2,k]-2*Z["branch2_3"].real*P[1,k]-2*Z["branch2_3"].imag*Q[1,k]+(Z["branch2_3"].real**2+Z["branch2_3"].imag**2)*L[1,k]==0)
            m.addConstr(v[2,k]-v[3,k]-2*Z["branch3_4"].real*P[2,k]-2*Z["branch3_4"].imag*Q[2,k]+(Z["branch3_4"].real**2+Z["branch3_4"].imag**2)*L[2,k]==0)
            m.addConstr(v[5,k]-v[4,k]-2*Z["branch5_6"].real*P[3,k]-2*Z["branch5_6"].imag*Q[3,k]+(Z["branch5_6"].real**2+Z["branch5_6"].imag**2)*L[3,k]==0)
            m.addConstr(v[6,k]-v[5,k]-2*Z["branch6_7"].real*P[4,k]-2*Z["branch6_7"].imag*Q[4,k]+(Z["branch6_7"].real**2+Z["branch6_7"].imag**2)*L[4,k]==0)
            m.addConstr(v[7,k]-v[6,k]-2*Z["branch7_8"].real*P[5,k]-2*Z["branch7_8"].imag*Q[5,k]+(Z["branch7_8"].real**2+Z["branch7_8"].imag**2)*L[5,k]==0)
            m.addConstr(v[7,k]-v[6,k]-2*Z["branch7_8"].real*P[5,k]-2*Z["branch7_8"].imag*Q[5,k]+(Z["branch7_8"].real**2+Z["branch7_8"].imag**2)*L[5,k]==0)
            m.addConstr(v[1,k]-v[7,k]-2*Z["branch2_8"].real*P[6,k]-2*Z["branch2_8"].imag*Q[6,k]+(Z["branch2_8"].real**2+Z["branch2_8"].imag**2)*L[6,k]==0)
            
            # constraints for battery
            
            m.addConstr(Pbcha[0,k]<=B_bin[0,k]*meta['batt_pow'])
            m.addConstr(Pbdisa[0,k]<=(1-B_bin[0,k])*meta['batt_pow'])
            
            m.addConstr(Pbcha[0,k]**2+Pbrea[0,k]**2<=meta['batt_pow']**2)
            m.addConstr(Pbdisa[0,k]**2+Pbrea[0,k]**2<=meta['batt_pow']**2)
            
            if k==0:
                m.addConstr(batt_ini+Pbcha[0,k]*steplength*meta['nch']-Pbdisa[0,k]*steplength/meta['ndis']==Eb[0,k])
            else:
                m.addConstr(Eb[0,k-1]+Pbcha[0,k]*meta['nch']*steplength-Pbdisa[0,k]*steplength/meta['ndis']==Eb[0,k])
            
            m.addConstr(P[0,k]**2+Q[0,k]**2<=L[0,k]*v[0,k])
            m.addConstr(P[1,k]**2+Q[1,k]**2<=L[1,k]*v[1,k])
            m.addConstr(P[2,k]**2+Q[2,k]**2<=L[2,k]*v[2,k])
            m.addConstr(P[3,k]**2+Q[3,k]**2<=L[3,k]*v[5,k])
            m.addConstr(P[4,k]**2+Q[4,k]**2<=L[4,k]*v[6,k])
            m.addConstr(P[5,k]**2+Q[5,k]**2<=L[5,k]*v[7,k])
            m.addConstr(P[6,k]**2+Q[6,k]**2<=L[6,k]*v[1,k])
            
    elif site == "es-crevillent":
        for k in range(t):
            if scenario == 'islanding':
                expr.addTerms(1,Pact[0,k],Pact[0,k])
            elif scenario == 'market':
                expr.addTerms(ele_price[k]*steplength,Pact[0,k])
                expr.addTerms(flex_price[k]*steplength,Ut_flex_up[0,k])
                expr.addTerms(flex_price[k]*steplength,Ut_flex_down[0,k])
            else:
                expr.addTerms(Z["branch1_2"].real,L[0,k])
            
        # "select" is found in the tupledict class on Gurobi website
        m.setObjective(expr, GRB.MINIMIZE)

        # Define constraints
        for k in range(t):
            # active power flow constraints
            m.addConstr(Pact[0,k]-P[0,k]==0)
            if scenario == 'islanding':
                m.addConstr(-P[0,k]+Z["branch1_2"].real*L[0,k]+Ut_flex_up[0,k]-Ut_flex_down[0,k]+prod_curt[0,k]-prod[0][k]+Pbcha[0,k]-Pbdisa[0,k]==-load[0][k])
            else:
                m.addConstr(-P[0,k]+Z["branch1_2"].real*L[0,k]+Ut_flex_up[0,k]-Ut_flex_down[0,k]-prod[0][k]+Pbcha[0,k]-Pbdisa[0,k]==-load[0][k])
            
            # reactive power flow constraints
            m.addConstr(Prea[0,k]-Q[0,k]==0)
            m.addConstr(-Q[0,k]+Z["branch1_2"].imag*L[0,k]-Rprod[0][k]+Pbrea[0,k]==-Rload[0][k])
            
            
            # voltage constraints
            m.addConstr(v[0,k]-v[1,k]-2*Z["branch1_2"].real*P[0,k]-2*Z["branch1_2"].imag*Q[0,k]+(Z["branch1_2"].real**2+Z["branch1_2"].imag**2)*L[0,k]==0)
            
            # constraints for battery
            
            m.addConstr(Pbcha[0,k]<=B_bin[0,k]*meta['batt_pow'])
            m.addConstr(Pbdisa[0,k]<=(1-B_bin[0,k])*meta['batt_pow'])
            
            m.addConstr(Pbcha[0,k]**2+Pbrea[0,k]**2<=meta['batt_pow']**2)
            m.addConstr(Pbdisa[0,k]**2+Pbrea[0,k]**2<=meta['batt_pow']**2)
            
            if k==0:
                m.addConstr(batt_ini+Pbcha[0,k]*steplength*meta['nch']-Pbdisa[0,k]*steplength/meta['ndis']==Eb[0,k])
            else:
                m.addConstr(Eb[0,k-1]+Pbcha[0,k]*meta['nch']*steplength-Pbdisa[0,k]*steplength/meta['ndis']==Eb[0,k])
            
            
            m.addConstr(P[0,k]**2+Q[0,k]**2<=L[0,k]*v[0,k])

    else:
        raise ValueError

    # set acceptable optimality gap:
    m.Params.mipgap=0.01

    # optimize problem:
    m.optimize()

    status = m.getAttr('Status')
    # format and return scheduling result:

    if status == GRB.OPTIMAL:
        result = {}
        result['BatteryOperations'] = [
            { 
                'interval': i+1, 
                'ActiveCharging': 0 if Pbcha[0,i].X < 5e-6 else (Pbcha[0,i].X/steplength)*1e3, 
                'ActiveDischarging': 0 if Pbdisa[0,i].X < 5e-6 else (Pbdisa[0,i].X/steplength)*1e3,
                'ReactivePower': 0 if abs(Pbrea[0,i].X) < 5e-6 else (Pbrea[0,i].X/steplength)*1e3,
                'ExpectedChargeStatus': 0 if Eb[0,i].X/meta['batt_cap'] < 0.01 else Eb[0,i].X/meta['batt_cap']
            }
        for i in range(t)]

        result['FlexibilityUtilisation'] = {
            bus: [
                {
                    'interval': i+1,
                    'upwards': 0 if Ut_flex_up[busi,i].X < 5e-6 else (Ut_flex_up[busi,i].X/steplength)*1e3,
                    'downwards': 0 if Ut_flex_down[busi,i].X < 5e-6 else (Ut_flex_down[busi,i].X/steplength)*1e3
                }
            for i in range(t)]
        for busi, bus in enumerate(busses['flexbuses'])}

        result['GridPower'] = [
            {
                'interval': i+1,
                'activePower': 0 if abs(Pact[0,i].X) < 5e-6 else (Pact[0,i].X/steplength)*1e3,
                'reactivePower': 0 if abs(Prea[0,i].X) < 5e-6 else (Prea[0,i].X/steplength)*1e3
            }
        for i in range(t)]

        return True, result
    elif status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
        m.computeIIS()
        result = {}

        for name in ["IISConstr", "IISLB", "IISUB"]:
            attr = m.getAttr(name)
            result[name] = []

            searching = True
            found = -1

            while searching:
                try:
                    found = attr.index(1, found+1)
                    result[name].append(found)
                except ValueError:
                    searching = False

        return False, result
