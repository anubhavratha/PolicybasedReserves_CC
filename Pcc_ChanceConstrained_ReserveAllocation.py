''' Case 1: Variable participation factor, SOCP -> Linear Implementation due to assumptions:
a. Uncertain vectors are independent of each other
b. Normal distribution of zero mean
c. Participation factor aligns with the up/down regulation needed
More description to come here '''

import numpy as np
import pandas as pd
import gurobipy as gb
from scipy.stats import invgauss,norm
import matplotlib.pyplot as plt
import time

# Helper Functions
def units_in_node(data,key):
    #This function provides the unit numbers for nodes
    # Mapping
    temp_dict= {}
    for k, v in data.items():
           for k2, v2 in v.items():
                      temp_dict[(k, k2)] = v2
    length = sum(value == key for value in temp_dict.values())
    solution=[]

    for g in range(length):
            solution.append([k for k, v in temp_dict.items() if v == key][g][0])

    return solution

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

# Uncertainty Modelling -> Errors in forecast is cast as independent random variables centered at mu_val 0 and with a fixed sigma_val
def generate_wind_error_variance(data,T,Sigma_baseline_Multiplier):
    #Function to generate wind forecasts for the different wind farm: Baseline standard deviation is considered as 7.5% of installed capacity,
    #In the absence of historical data, the
    K = [k for k, k_info in data['WindFarm'].items()]
    sigma_k_hourly=[]
    var_omega_hourly = []
    for k in K:
        sigma_k_baseline = Sigma_baseline_Multiplier*data['WindFarm'][k]['Wmax']     #Base line Std Deviation is considered 7.5% of the max capacity
        sigma_k_hourly=np.array(np.random.normal(sigma_k_baseline,0.2*sigma_k_baseline, len(T)))    #Hourly standard deviation is considered to be around the baseline at 20% STD Dev
        var_omega_hourly.append(np.square(sigma_k_hourly))
    var_omega = np.sum(var_omega_hourly,axis=0)
    #print(np.cov(var_omega_hourly[0],var_omega_hourly[1]))
    return var_omega

''' ----- DAY-AHEAD MARKET : FUNCTION DEFINITION ----'''
def clearDAMarket_and_Reserves_CC(data,Sigma_baseline_Multiplier):
    # Create a new GurobiPy model
    m = gb.Model()
    #Create datasets -> indices
    G = [g for g, g_info in data['FlexGens'].items()]
    K = [k for k, k_info in data['WindFarm'].items()]
    D = [d for d, d_info in data['Demand'].items()]
    T = [t for t, t_info in data['Demand'].items()]
    N = list(data['BusNums'])
    #Call the uncertainty model
    var_omega = generate_wind_error_variance(data,T,Sigma_baseline_Multiplier)
    print(var_omega)
    # Create variables
    p = {}
    alpha_val={}
    for g in G:
        for t in T:
            p[(g,t)] = m.addVar(lb=data['FlexGens'][g]['Pmin'], ub=data['FlexGens'][g]['Pmax'], name='Generation of power plant; {}; {};'.format(g,t))
            alpha_val[(g,t)] = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='Flexibility Participation of power plants; {}; {};'.format(g,t))
    p
    w = {}
    Wind_F={}
    for k in K:
        for t in T:
            w[(k,t)] = m.addVar(lb=0.0, ub=data['WindFarm'][k]['Wmax']*data['WindFactor'][t][k], name='Generation of wind farm; {}; {}; '.format(k,t))
    w
    m.update()
    #Objective function
    m.setObjective(gb.quicksum(data['FlexGens'][g]['LinCost']*p[(g,t)] + data['FlexGens'][g]['QuadCost']*(p[(g,t)]*p[(g,t)] + var_omega[int(t.replace('t',''))-1]*alpha_val[(g,t)]*alpha_val[(g,t)]) for g in G for t in T), gb.GRB.MINIMIZE)
    m.update()
    #Constraints:
    for n in N:
        A_G = units_in_node(data['FlexGens'],n)
        A_K = units_in_node(data['WindFarm'],n)
        A_DE = units_in_node(data['ELDemand'],n)
        for t in T:
            var_omega_t = var_omega[int(t.replace('t',''))-1]
            #DayAhead and Recourse Power Balance Constraint
            m.addConstr(gb.quicksum(p[(g,t)] for g  in A_G) + gb.quicksum(w[(k,t)] for k in A_K),gb.GRB.EQUAL, gb.quicksum(data['ELDemand'][d]['share']*data['Demand'][t]['EL'] for d in A_DE) ,name="power balance node; {}; {};".format(n,t))
            m.addConstr(gb.quicksum(alpha_val[(g,t)] for g in A_G),gb.GRB.EQUAL,1.0, name="recourse equality {}".format(t))
            #Flexible Generators Constraints: Linearized Chance Constraints for production and ramping
            for g in G:
                m.addConstr(alpha_val[(g,t)],gb.GRB.GREATER_EQUAL,0.0, name="alpha non negative {}{}".format(g,t))
                m.addConstr(p[(g,t)] + norm.ppf(1-data['FlexGens'][g]['epsilon'])*alpha_val[(g,t)]*np.sqrt(var_omega_t),gb.GRB.LESS_EQUAL,data['FlexGens'][g]['Pmax'],name="Production Max CC FlexGens {}{}".format(g,t))
                m.addConstr(p[(g,t)] - norm.ppf(1-data['FlexGens'][g]['epsilon'])*alpha_val[(g,t)]*np.sqrt(var_omega_t),gb.GRB.GREATER_EQUAL,data['FlexGens'][g]['Pmin'],name="Production Min CC FlexGens {}{}".format(g,t))
                m.addConstr(norm.ppf(1-data['FlexGens'][g]['epsilon'])*alpha_val[(g,t)]*np.sqrt(var_omega_t),gb.GRB.LESS_EQUAL,data['FlexGens'][g]['Rmax'],name="Ramping UP CC FlexGens {}{}".format(g,t))
                m.addConstr(-norm.ppf(1-data['FlexGens'][g]['epsilon'])*alpha_val[(g,t)]*np.sqrt(var_omega_t),gb.GRB.LESS_EQUAL,data['FlexGens'][g]['Rmax'],name="Ramping DN Max CC FlexGens {}{}".format(g,t))
    m.optimize()
    #Debugging Infeasibility
    #print("~~~__=========__~~~")
    #print(m.getConstrs())
    #print("~~~__=========__~~~")
    m.write('Pcc_CC_FullModel.lp')
    if(m.Status != 2):
        m.computeIIS()
        m.write('Pcc_infeas_data.ilp')
        Model_Status = 'Pcc_DA_Infeasible'
        return Model_Status
    #Retrieving Optimal Values for Generation Dispatch and Reserves Allocated
    Nom_gen_values=([v.x for v in m.getVars() if 'g' in v.varName and "Generation" in v.varName ])
    Alpha_gen_values=([v.x for v in m.getVars() if 'g' in v.varName and "Participation" in v.varName ])
    Wind_prod_values=([v.x for v in m.getVars() if 'k' in v.varName and "Generation" in v.varName])
    #Pre-processing of the values to obtain more readable values
    GenProd = split_list(Nom_gen_values,wanted_parts=len(Nom_gen_values)//len(T))
    WindForecast = split_list(Wind_prod_values,wanted_parts=len(Wind_prod_values)//len(T))
    AlphaG = split_list(Alpha_gen_values,wanted_parts=len(Alpha_gen_values)//len(T))
    #Retrieving Duals for the power balance constraints for recovering cleared DA Market Price (lambda_DA) and cleared Reserve Provision Price (lambda_RE)
    duals={}
    for c in m.getConstrs():
        duals[c.constrName] = c.getAttr('Pi')
    lambda_DA_vals = ([value for key, value in duals.items() if 'power' in key.lower()])
    lambda_RE_vals = ([value for key, value in duals.items() if 'recourse' in key.lower()])
    NetLoad=[]
    for t in T:
        NetLoad.append(data['Demand'][t]['EL'])
    #Split-up total DA cost into cost of energy and cost of Reserves
    EnergyCostsPerGenerator = []
    ReservePolicyProcurementCostsPerGenerator = []
    for iter in range(len(GenProd)):
        EnergyCostsPerGenerator.append([np.inner(GenProd[iter],lambda_DA_vals)])
        ReservePolicyProcurementCostsPerGenerator.append([np.inner(AlphaG[iter],lambda_RE_vals)])
    TotalDAEnergyCosts = np.sum(EnergyCostsPerGenerator)
    TotalDAReservePolicyProcurementCosts = np.sum(ReservePolicyProcurementCostsPerGenerator)
    return m.objVal,m.status,m.getVars(),lambda_DA_vals,lambda_RE_vals,GenProd,AlphaG,WindForecast,NetLoad, TotalDAEnergyCosts, TotalDAReservePolicyProcurementCosts,EnergyCostsPerGenerator,ReservePolicyProcurementCostsPerGenerator

''''----- REAL TIME OPERATION PROGRAM (Not Optimization) ---- '''
def runRealTimeSimulations(data,DeltaVals,DA_Opt_Dispatch,DA_Opt_AlphaVals):
    VOLL = 500.0 # Value of Lost Load
    #Create datasets -> indices
    G = [g for g, g_info in data['FlexGens'].items()]
    K = [k for k, k_info in data['WindFarm'].items()]
    D = [d for d, d_info in data['Demand'].items()]
    T = [t for t, t_info in data['Demand'].items()]
    N = list(data['BusNums'])
    # Create variables
    P_net = np.zeros([len(G),len(T)])
    P_Adjustments = np.zeros([len(G),len(T)])
    Cost_gen_net = np.zeros([len(G),len(T)])
    for g in G:
        for t in T:
            G_index=int(g.strip('g'))-1
            T_index=int(t.strip('t'))-1
            P_net[G_index][T_index] = DA_Opt_Dispatch[(g,t)] - DA_Opt_AlphaVals[(g,t)]*DeltaVals[(t)]
            P_Adjustments[G_index][T_index] = - DA_Opt_AlphaVals[(g,t)]*DeltaVals[(t)]
            Cost_gen_net[G_index][T_index] = data['FlexGens'][g]['LinCost']*P_net[G_index][T_index] + data['FlexGens'][g]['QuadCost']*(P_net[G_index][T_index]**2)
    P_net
    w_spill=np.zeros([len(K),len(T)])
    wind_forecast=np.zeros([len(K),len(T)])
    for k in K:
        for t in T:
            K_index=int(k.strip('k'))-1
            T_index=int(t.strip('t'))-1
            w_spill[K_index][T_index] = 0.0
            wind_forecast[K_index][T_index] = data['WindFarm'][k]['Wmax']*data['WindFactor'][t][k]
    w_spill
    d_shed={}
    for n in N:
        for t in T:
            d_shed[(n,t)] = 0.0
    #Calculating the Cost of Real-Time scenario execution
    for n in N:
        A_G = units_in_node(data['FlexGens'],n)
        A_DE = units_in_node(data['ELDemand'],n)
        A_K = units_in_node(data['WindFarm'],n)
        Surplus_Gen=[]
        for t in T:
            #Realtime-Balance Criteria
            T_index=int(t.strip('t'))-1
            #print('Load in hour: {}'.format(sum(data['ELDemand'][d]['share']*data['Demand'][t]['EL'] for d in A_DE)))
            Surplus_Gen.append(np.sum(P_net,axis = 0)[T_index] + (np.sum(wind_forecast, axis = 0)[T_index] + DeltaVals[(t)]) - sum(data['ELDemand'][d]['share']*data['Demand'][t]['EL'] for d in A_DE))
        print('Total Flex Generation: {}'.format(np.sum(P_net,axis=0)))
        print('Total Wind Forecast:{}'.format(np.sum(wind_forecast, axis = 0)))
        #print('Total Wind Realization = {}'.format((np.sum(wind_forecast, axis = 0) + DeltaVals[(t)])))
        print('Surplus Generation: {}'.format(np.round(Surplus_Gen,2)))
        if(any(item > 0.001 for item in Surplus_Gen)):
            print('WindSpillage')
        if(any(item < -0.001 for item in Surplus_Gen)):
            print('LoadShedding')
        #Calculating Total wind spilled and load shedding in this scenario
        WindSpillage_This_Scenario = sum(Surplus_Gen[i] for i in range(len(Surplus_Gen)) if Surplus_Gen[i] > 0.001)
        LoadShedding_This_Scenario = sum(Surplus_Gen[i] for i in range(len(Surplus_Gen)) if Surplus_Gen[i] < -0.001)
        Total_RT_Generation_Cost = np.sum(np.sum(Cost_gen_net,axis=0))
        VOLL_Cost = LoadShedding_This_Scenario*VOLL
        Total_RT_Operation_Cost = Total_RT_Generation_Cost + VOLL_Cost
        print('Total RT Generation cost: {:.2e}'.format(Total_RT_Operation_Cost))
        return Total_RT_Operation_Cost,P_net,WindSpillage_This_Scenario,WindSpillage_This_Scenario,P_Adjustments
