''' Deterministic Benchmark for Reserve Procurement and Activation using Minimum Reserve Margin (MRR) '''
import numpy as np
import pandas as pd
import gurobipy as gb
from scipy.stats import invgauss,norm
import matplotlib.pyplot as plt
import time

# Defining Helper Functions used by both optimization programs
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


''' ----- DAY-AHEAD MARKET : FUNCTION DEFINITION ----'''
def clearDAMarket_and_Reserves(data,MRR):
    # Create a new GurobiPy model
    m = gb.Model()
    #Create datasets -> indices
    G = [g for g, g_info in data['FlexGens'].items()]
    K = [k for k, k_info in data['WindFarm'].items()]
    D = [d for d, d_info in data['Demand'].items()]
    T = [t for t, t_info in data['Demand'].items()]
    N = list(data['BusNums'])
    # Create variables
    p = {}
    R={}
    for g in G:
        for t in T:
            p[(g,t)] = m.addVar(lb=data['FlexGens'][g]['Pmin'], ub=data['FlexGens'][g]['Pmax'], name='Generation Scheduleof power plant;{};{}; '.format(g,t))
            R[(g,t)] = m.addVar(lb=0.0, ub=data['FlexGens'][g]['Rmax'], name='Upwards/Downwards Reserve Allocation for power plant; {}; {}; '.format(g,t))
    p
    w = {}
    Wind_F={}
    for k in K:
        for t in T:
            w[(k,t)] = m.addVar(lb=0.0, ub=data['WindFarm'][k]['Wmax']*data['WindFactor'][t][k], name='Generation of wind farm; {}; {}; '.format(k,t))
    w
    m.update()
    #Objective function
    m.setObjective(gb.quicksum(data['FlexGens'][g]['LinCost']*p[(g,t)] + data['FlexGens'][g]['QuadCost']*(p[(g,t)]*p[(g,t)] + data['FlexGens'][g]['Cost_Res_Proc']*R[(g,t)]) for g in G for t in T), gb.GRB.MINIMIZE)
    m.update()
    #Constraints:
    for n in N:
        A_G = units_in_node(data['FlexGens'],n)
        A_K = units_in_node(data['WindFarm'],n)
        A_DE = units_in_node(data['ELDemand'],n)
        for t in T:
            #DayAhead and Recourse Power Balance Constraint
            m.addConstr(gb.quicksum(p[(g,t)] for g  in A_G) + gb.quicksum(w[(k,t)] for k in A_K),gb.GRB.EQUAL, gb.quicksum(data['ELDemand'][d]['share']*data['Demand'][t]['EL'] for d in A_DE) ,name="power balance node{}{}".format(n,t))
            m.addConstr(gb.quicksum(R[(g,t)] for g in A_G),gb.GRB.GREATER_EQUAL,MRR, name="MRR allocation {}".format(t))
            #Flexible Generators Constraints: Linearized Chance Constraints for production and ramping
            for g in G:
                m.addConstr(p[(g,t)] + R[(g,t)],gb.GRB.LESS_EQUAL,data['FlexGens'][g]['Pmax'],name="Production+Reserves Max CC FlexGens {}{}".format(g,t))
                m.addConstr(p[(g,t)] - R[(g,t)],gb.GRB.GREATER_EQUAL,data['FlexGens'][g]['Pmin'],name="Production+Reserves Min CC FlexGens {}{}".format(g,t))
    m.optimize()
    if(m.Status != 2):
        m.computeIIS()
        m.write('Pdet_infeas_data.ilp')
        Model_Status = 'Pdet_DA_Infeasible'
        return Model_Status
    #Retrieving Optimal Values for Generation Dispatch and Reserves Allocated
    Nom_gen_values=([v.x for v in m.getVars() if 'g' in v.varName and "Generation" in v.varName ])
    Reserve_gen_values=([v.x for v in m.getVars() if 'g' in v.varName and "Reserve" in v.varName ])
    Wind_prod_values=([v.x for v in m.getVars() if 'k' in v.varName and "Generation" in v.varName])
    #Pre-processing of the values to obtain more readable values
    GenProd = split_list(Nom_gen_values,wanted_parts=len(Nom_gen_values)//len(T))
    WindForecast = split_list(Wind_prod_values,wanted_parts=len(Wind_prod_values)//len(T))
    Reserves = split_list(Reserve_gen_values,wanted_parts=len(Reserve_gen_values)//len(T))
    #Retrieving Duals for the power balance constraints for recovering cleared DA Market Price (lambda_DA) and cleared Reserve Provision Price (lambda_RE)
    duals={}
    for c in m.getConstrs():
        duals[c.constrName] = c.getAttr('Pi')
    lambda_DA_vals = ([value for key, value in duals.items() if 'power' in key.lower()])
    lambda_RE_vals = ([value for key, value in duals.items() if 'allocation' in key.lower()])
    #Obtaining the value of netload for plotting
    NetLoad=[]
    for t in T:
        NetLoad.append(data['Demand'][t]['EL'])
    #Split-up total DA cost into cost of energy and cost of Reserves
    EnergyCostsPerGenerator = []
    ReserveProcurementCostsPerGenerator = []
    for iter in range(len(GenProd)):
        EnergyCostsPerGenerator.append([np.inner(GenProd[iter],lambda_DA_vals)])
        ReserveProcurementCostsPerGenerator.append([np.inner(Reserves[iter],np.tile(data['FlexGens']['g'+str(iter+1)]['Cost_Res_Proc'],len(T)))])
    TotalDAEnergyCosts = np.sum(EnergyCostsPerGenerator)
    TotalDAReserveProcurementCosts = np.sum(ReserveProcurementCostsPerGenerator)
    return m.objVal,m.status,m.getVars(),lambda_DA_vals,lambda_RE_vals,GenProd,Reserves,WindForecast,NetLoad,TotalDAEnergyCosts,TotalDAReserveProcurementCosts, EnergyCostsPerGenerator, ReserveProcurementCostsPerGenerator

''' ----- REAL-TIME BALANCING MARKET : FUNCTION DEFINITION ----'''
def clearRTBalancingMarket(data,DeltaVals,DA_Opt_Dispatch,DA_Opt_Reserves):
    VOLL = 500.0 # Value of Lost Load
    rt_model = gb.Model()
    #Create datasets -> indices
    G = [g for g, g_info in data['FlexGens'].items()]
    K = [k for k, k_info in data['WindFarm'].items()]
    D = [d for d, d_info in data['Demand'].items()]
    T = [t for t, t_info in data['Demand'].items()]
    N = list(data['BusNums'])
    # Create variables
    r = {}
    for g in G:
        for t in T:
            r[(g,t)] = rt_model.addVar(lb=-DA_Opt_Reserves[(g,t)], ub=DA_Opt_Reserves[(g,t)], name='Reserves activated from the power plant;{};{}; '.format(g,t))
    r
    w_spill={}
    for k in K:
        for t in T:
            w_spill[(k,t)] = rt_model.addVar(lb=0.0, ub=data['WindFarm'][k]['Wmax']*data['WindFactor'][t][k], name='Wind spillage from wind farm; {}; {}; '.format(k,t))
    w_spill
    d_shed={}
    for n in N:
        for t in T:
            d_shed[(n,t)] = rt_model.addVar(lb=0.0,ub=gb.GRB.INFINITY, name='Load shedding activated this scenario at bus n in period t: {} {} '.format(n,t))
    rt_model.update()
    #Objective function
    rt_model.setObjective(gb.quicksum(data['FlexGens'][g]['LinCost']*(DA_Opt_Dispatch[(g,t)] + r[(g,t)]) + data['FlexGens'][g]['QuadCost']*((DA_Opt_Dispatch[(g,t)]+r[(g,t)])*(DA_Opt_Dispatch[(g,t)]+r[(g,t)])) for g in G for t in T) + gb.quicksum(VOLL*d_shed[(n,t)] for n in N for t in T), gb.GRB.MINIMIZE)
    rt_model.update()
    for n in N:
        A_G = units_in_node(data['FlexGens'],n)
        A_DE = units_in_node(data['ELDemand'],n)
        A_K = units_in_node(data['WindFarm'],n)
        for t in T:
            #Realtime-Balance Criteria
            rt_model.addConstr(gb.quicksum(r[(g,t)] for g  in A_G) + d_shed[(n,t)] - gb.quicksum(w_spill[(k,t)] for k in A_K) + DeltaVals[(t)], gb.GRB.EQUAL, 0.0 ,name="balance for deviation {}{}".format(n,t))
            rt_model.addConstr(d_shed[(n,t)],gb.GRB.LESS_EQUAL, gb.quicksum(data['ELDemand'][d]['share']*data['Demand'][t]['EL'] for d in A_DE), name='Load Shedding Maximum {}'.format(n,t))
    rt_model.optimize()
    if(rt_model.Status != 2):
        rt_model.computeIIS()
        rt_model.write('Pdet_RT_infeas_data.ilp')
    #Extract optimal values of Optimization Variable
    Adjustment_gen_values = ([v.x for v in rt_model.getVars() if "Reserves" in v.varName ])
    Load_shed_values = ([v.x for v in rt_model.getVars() if "Load" in v.varName ])
    Wind_spillage_values = ([v.x for v in rt_model.getVars() if "Wind" in v.varName ])
    #Making easier to display
    Adjustments = split_list(Adjustment_gen_values,wanted_parts=len(Adjustment_gen_values)//len(T))
    LoadSheds = split_list(Load_shed_values,wanted_parts=len(Load_shed_values)//len(T))
    WindSpills = split_list(Wind_spillage_values,wanted_parts=len(Wind_spillage_values)//len(T))
    PrintableAdjustments = [[np.round(float(i), 4) for i in nested] for nested in Adjustments]
    PrintableLoadSheds = [[np.round(float(i), 4) for i in nested] for nested in LoadSheds]
    PrintableWindSpills = [[np.round(float(i), 4) for i in nested] for nested in WindSpills]
    #Recovering Balancing Price
    duals={}
    for c in rt_model.getConstrs():
        duals[c.constrName] = c.getAttr('Pi')
    lambda_RT_vals = ([value for key, value in duals.items() if 'balance' in key.lower()])
    #Calculating Total wind spilled and load shedding in this scenario
    WindSpillage_This_Scenario = np.sum(WindSpills)
    LoadShedding_This_Scenario = np.sum(LoadSheds)
    return rt_model.objVal,rt_model.status,rt_model.getVars(),lambda_RT_vals,WindSpillage_This_Scenario,LoadShedding_This_Scenario,PrintableAdjustments
