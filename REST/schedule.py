from .utilities import scheduling as sc

definitions = {
    'minutes': 1,
    'hours': 60
}

defaults = {
    'at-strem': {
        'batt_cap': 0.25,
        'batt_pow': 0.25,
        'nch': 0.95,
        'ndis': 0.95,
        'fr': {
            'start': (0,0,0),
            'end': (6,0,0),
            'min': 0.4 * 0.25,
            'max': 0.6 * 0.25
        },
        'loadbuses': ['T205','T261','T265','T264','T262'],
        'genbuses': ['T205','T261','T267','T265','T266','T264','T262'],
        'flexbuses': ['T205','T261', 'T265', 'T264','T262']
    }
}

def schedule(data):
    import datetime as dt
    site = data['Site']
    window = (data['horizon'] * definitions[data['horizonType']]) // (data['isp'] * definitions[data['ispType']])
    steplength = data['isp']/(60 if data['ispType'] == 'minutes' else 1)
    startDate = dt.datetime.fromisoformat(data['SchedulePeriodStart'])

    meta = defaults[site]

    batt_ini = data['Battery']['ChargeState'] * meta['batt_cap'] #assumes battery charge as a fraction of full
    load = sc.formatLoadForecast(data['Forecast']['Load'], window, steplength, meta['loadbuses'])
    prod = sc.formatGenerationForecast(data['Forecast']['Generation'], window, steplength, meta['genbuses'])
    flex_up, flex_down, flex_price = sc.formatFlexibilityData(data['Flexibility'], window, steplength, meta['flexbuses'])

    ele_price = sc.formatPriceData(data['Pricing'], window, steplength)[0]

    G, B = sc.getAdmittanceMatrix(site)

    import gurobipy as gp
    from gurobipy import GRB

    # Build the optimisation model
    m = gp.Model("socp")
    
    import math
    import numpy as np

    t = window

    u_min = np.square(0.98)/math.sqrt(2) 
    u_max = np.square(1.02)/math.sqrt(2)
    u_1=1/math.sqrt(2)
    
    # define lower and upper boundary
    u_lb=np.ones([8,t])*u_min
    u_lb[0,:]=u_1
    u_ub=np.ones([8,t])*u_max
    u_ub[0,:]=u_1
    # u R T are the variables related to bus voltage, and power flow between buses
    u=m.addVars(8,t,lb=u_lb,ub=u_ub,name="u")
    R=m.addVars(8,t,lb=0,ub=GRB.INFINITY,name="R")
    T=m.addVars(8,t,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="T")

    # Eact and Erea are the active and reactive power required from the grid to input to the distribution network
    Eact=m.addVars(1,t,lb=0,ub=GRB.INFINITY,name="Eact")
    Erea=m.addVars(1,t,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="Erea")

    # Utilised flexibility
    Ut_flex_up=m.addVars(5,t,lb=0,ub=flex_up,name="Ut_flex_up")
    Ut_flex_down=m.addVars(5,t,lb=0,ub=flex_down,name="Ut_flex_down")

    # renewable energy input
    Erenew=m.addVars(7,t,lb=0,ub=prod,name="Erenew")

    # battery active charging power
    Ebcha=m.addVars(1,t,lb=0,ub=meta['batt_pow']*steplength,name="Ebcha")
    # battery reactive charging/discharging power, positive means charging, negative means discharging
    Ebrea=m.addVars(1,t,lb=-meta['batt_pow']*steplength,ub=meta['batt_pow']*steplength,name="Ebrea")
    # battery active discharging power
    Ebdisa=m.addVars(1,t,lb=0,ub=meta['batt_pow']*steplength,name="Ebdisa")

    # battery capacity
    Eb_lb=np.zeros([1,t])
    Eb_ub=np.ones([1,t])*meta['batt_cap']

    sc.applyFrequencyResponse(Eb_lb, Eb_ub, meta['fr'], startDate, window, steplength)

    # battery energy level
    Eb=m.addVars(1,t,lb=Eb_lb,ub=Eb_ub,name="Eb")


    expr=gp.LinExpr(0)
    # define objective function
    for k in range(t):
        expr.addTerms(ele_price[k],Eact[0,k])
        expr.addTerms(ele_price[k],Erenew[0,k])
        expr.addTerms(ele_price[k],Erenew[1,k])
        expr.addTerms(ele_price[k],Erenew[2,k])
        expr.addTerms(ele_price[k],Erenew[3,k])
        expr.addTerms(ele_price[k],Erenew[4,k])
        expr.addTerms(ele_price[k],Erenew[5,k])
        expr.addTerms(ele_price[k],Erenew[6,k])
        
        expr.addTerms(flex_price[0][k],Ut_flex_up[0,k])
        expr.addTerms(flex_price[1][k],Ut_flex_up[1,k])
        expr.addTerms(flex_price[2][k],Ut_flex_up[2,k])
        expr.addTerms(flex_price[3][k],Ut_flex_up[3,k])
        expr.addTerms(flex_price[4][k],Ut_flex_up[4,k])
        
        expr.addTerms(flex_price[0][k],Ut_flex_down[0,k])
        expr.addTerms(flex_price[1][k],Ut_flex_down[1,k])
        expr.addTerms(flex_price[2][k],Ut_flex_down[2,k])
        expr.addTerms(flex_price[3][k],Ut_flex_down[3,k])
        expr.addTerms(flex_price[4][k],Ut_flex_down[4,k])
        
        
        
    # "select" is found in the tupledict class on Gurobi website
    m.setObjective(expr, GRB.MINIMIZE)

    for k in range(t):
        m.addConstr(u[0,k]*math.sqrt(2)*G[0,0]+R[0,k]*G[0,1]+T[0,k]*B[0,1]-Eact[0,k]==0)
        m.addConstr(u[1,k]*math.sqrt(2)*G[1,1]+R[0,k]*G[1,0]+R[1,k]*G[1,2]+R[7,k]*G[1,7]-T[0,k]*B[1,0]+T[1,k]*B[1,2]-T[7,k]*B[1,7]-Erenew[0,k]+Ut_flex_up[0,k]-Ut_flex_down[0,k]==-load[0][k])
        m.addConstr(u[2,k]*math.sqrt(2)*G[2,2]+R[1,k]*G[1,2]+R[2,k]*G[2,3]-T[1,k]*B[1,2]+T[2,k]*B[2,3]-Erenew[1,k]+Ut_flex_up[1,k]-Ut_flex_down[1,k]==-load[0][k])
        m.addConstr(u[3,k]*math.sqrt(2)*G[3,3]+R[2,k]*G[2,3]+R[3,k]*G[3,4]-T[2,k]*B[2,3]+T[3,k]*B[3,4]-Erenew[2,k]==0)
        m.addConstr(u[4,k]*math.sqrt(2)*G[4,4]+R[3,k]*G[3,4]+R[4,k]*G[4,5]-T[3,k]*B[3,4]+T[4,k]*B[4,5]-Erenew[3,k]+Ut_flex_up[2,k]-Ut_flex_down[2,k]==-load[2][k])
        m.addConstr(u[5,k]*math.sqrt(2)*G[5,5]+R[4,k]*G[4,5]+R[5,k]*G[5,6]-T[4,k]*B[4,5]+T[5,k]*B[5,6]-Erenew[4,k]+Ebcha[0,k]-Ebdisa[0,k]==0)
        m.addConstr(u[6,k]*math.sqrt(2)*G[6,6]+R[5,k]*G[5,6]+R[6,k]*G[6,7]-T[5,k]*B[5,6]+T[6,k]*B[6,7]-Erenew[5,k]+Ut_flex_up[3,k]-Ut_flex_down[3,k]==-load[3][k])
        m.addConstr(u[7,k]*math.sqrt(2)*G[7,7]+R[6,k]*G[6,7]+R[7,k]*G[7,1]-T[6,k]*B[6,7]+T[7,k]*B[7,1]-Erenew[6,k]+Ut_flex_up[4,k]-Ut_flex_down[4,k]==-load[4][k])
        
        m.addConstr(-u[0,k]*math.sqrt(2)*B[0,0]-R[0,k]*B[0,1]+T[0,k]*G[0,1]-Erea[0,k]==0)
        m.addConstr(-u[1,k]*math.sqrt(2)*B[1,1]-R[0,k]*B[1,0]-R[1,k]*B[1,2]-R[7,k]*B[1,7]-T[0,k]*G[1,0]+T[1,k]*G[1,2]-T[7,k]*G[1,7]-Erenew[0,k]*math.tan(math.acos(0.9))==-load[0][k]*math.tan(math.acos(0.97)))
        m.addConstr(-u[2,k]*math.sqrt(2)*B[2,2]-R[1,k]*B[1,2]-R[1,k]*B[2,3]-T[1,k]*G[1,2]+T[2,k]*G[2,3]-Erenew[1,k]*math.tan(math.acos(0.9))==-load[1][k]*math.tan(math.acos(0.97)))
        m.addConstr(-u[3,k]*math.sqrt(2)*B[3,3]-R[2,k]*B[2,3]-R[2,k]*B[3,4]-T[2,k]*G[2,3]+T[3,k]*G[3,4]-Erenew[2,k]*math.tan(math.acos(0.9))==0)
        m.addConstr(-u[4,k]*math.sqrt(2)*B[4,4]-R[3,k]*B[3,4]-R[3,k]*B[4,5]-T[3,k]*G[3,4]+T[4,k]*G[4,5]-Erenew[3,k]*math.tan(math.acos(0.9))==-load[2][k]*math.tan(math.acos(0.97)))
        m.addConstr(-u[5,k]*math.sqrt(2)*B[5,5]-R[4,k]*B[4,5]-R[4,k]*B[5,6]-T[4,k]*G[4,5]+T[5,k]*G[5,6]-Erenew[4,k]*math.tan(math.acos(0.9))+Ebrea[0,k]==0)
        m.addConstr(-u[6,k]*math.sqrt(2)*B[6,6]-R[5,k]*B[5,6]-R[5,k]*B[6,7]-T[5,k]*G[5,6]+T[6,k]*G[6,7]-Erenew[5,k]*math.tan(math.acos(0.9))==-load[3][k]*math.tan(math.acos(0.97)))
        m.addConstr(-u[7,k]*math.sqrt(2)*B[7,7]-R[6,k]*B[6,7]-R[7,k]*B[7,1]-T[6,k]*G[6,7]+T[7,k]*G[7,1]-Erenew[6,k]*math.tan(math.acos(0.9))==-load[4][k]*math.tan(math.acos(0.97)))
        
        m.addConstr(np.square(Ebcha[0,k])+np.square(Ebrea[0,k])<=np.square(meta['batt_pow']*steplength))
        m.addConstr(np.square(Ebdisa[0,k])+np.square(Ebrea[0,k])<=np.square(meta['batt_pow']*steplength))
        
        if k==0:
            m.addConstr(batt_ini+Ebcha[0,k]*meta['nch']-Ebdisa[0,k]/meta['ndis']==Eb[0,k])
        else:
            m.addConstr(Eb[0,k-1]+Ebcha[0,k]*meta['nch']-Ebdisa[0,k]/meta['ndis']==Eb[0,k])
            
        m.addConstr(np.square(R[7,k])+np.square(T[7,k])<=2*u[1,k]*u[7,k])
        for i in range(7):
            m.addConstr(np.square(R[i,k])+np.square(T[i,k])<=2*u[i,k]*u[i+1,k])

            

    m.optimize()
    pass

    result = {}
    result['BatteryOperations'] = [
        { 
            'interval': i+1, 
            'ActiveCharging': 0 if Ebcha[0,i].X < 5e-6 else (Ebcha[0,i].X/steplength)*1e3, 
            'ActiveDischarging': 0 if Ebdisa[0,i].X < 5e-6 else (Ebdisa[0,i].X/steplength)*1e3,
            'ReactivePower': 0 if abs(Ebrea[0,i].X) < 5e-6 else (Ebrea[0,i].X/steplength)*1e3,
            'ExpectedChargeStatus': 0 if Eb[0,i].X/meta['batt_cap'] < 0.01 else Eb[0,i].X/meta['batt_cap']
        }
    for i in range(window)]

    result['FlexibilityUtilisation'] = {
        bus: [
            {
                'interval': i+1,
                'upwards': 0 if Ut_flex_up[busi,i].X < 5e-6 else (Ut_flex_up[busi,i].X/steplength)*1e3,
                'downwards': 0 if Ut_flex_down[busi,i].X < 5e-6 else (Ut_flex_down[busi,i].X/steplength)*1e3
            }
        for i in range(window)]
    for busi, bus in enumerate(meta['flexbuses'])}

    result['Generation'] = {
        bus: [
            {
                'interval': i+1,
                'activePower': 0 if Erenew[busi,i].X < 5e-6 else (Erenew[busi,i].X/steplength)*1e3
            }
        for i in range(window)]
    for busi, bus in enumerate(meta['genbuses'])}

    result['GridPower'] = [
        {
            'interval': i+1,
            'activePower': 0 if abs(Eact[0,i].X) < 5e-6 else (Eact[0,i].X/steplength)*1e3,
            'reactivePower': 0 if abs(Erea[0,i].X) < 5e-6 else (Erea[0,i].X/steplength)*1e3
        }
    for i in range(window)]

    return {
        'ScheduleStartDate': data['SchedulePeriodStart'],
        'horizon': data['horizon'],
        'horizonType': data['horizonType'],
        'isp': data['isp'],
        'ispType': data['ispType'],
        'Schedule': result
    }