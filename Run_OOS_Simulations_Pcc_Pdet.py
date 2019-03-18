'''
Running OOSs:
A. Biased Simulations: OOS scenarios generated from Normal distribution with the same values of sigma (as used by Pcc Chance Constraints)
are used to compare Pdet and Pcc in terms of their OOS performance: Total End of Day Costs, Profits for Generators, etc.
B. Unbiased Simulations:
    B1: Normal distribution with variations in values of sigma - to test the performance of both Pdet and Pcc in terms of the paramters above
'''

import numpy as np
import pandas as pd
import gurobipy as gb
from scipy.stats import invgauss,norm
import matplotlib.pyplot as plt
import time

#Importing Market Clearing Functions from the two models
from Pdet_Benchmark_Deterministic_Reserves import clearDAMarket_and_Reserves,clearRTBalancingMarket
from Pcc_ChanceConstrained_ReserveAllocation import clearDAMarket_and_Reserves_CC,runRealTimeSimulations

# Function to create Input Data used by the OOS Simulations
def input():
    solution = {

    'BusNums' : {'n1'},

    'FlexGens':{
    'g1':{'Pmin':0, 'Pmax':152, 'BusNum':'n1', 'Rmax':40, 'LinCost':13.32, 'QuadCost':0.02, 'Cost_Res_Proc':15, 'epsilon':0.05},
    'g2':{'Pmin':0, 'Pmax':152, 'BusNum':'n1', 'Rmax':40, 'LinCost':13.32, 'QuadCost':0.08, 'Cost_Res_Proc':15, 'epsilon':0.05},
    'g3':{'Pmin':0, 'Pmax':350, 'BusNum':'n1', 'Rmax':70, 'LinCost':20.7, 'QuadCost':0.03, 'Cost_Res_Proc':10, 'epsilon':0.05},
    'g4':{'Pmin':0, 'Pmax':591, 'BusNum':'n1', 'Rmax':180, 'LinCost':20.93, 'QuadCost':0.0037, 'Cost_Res_Proc':8, 'epsilon':0.05},
    'g5':{'Pmin':0, 'Pmax':60, 'BusNum':'n1', 'Rmax':60, 'LinCost':26.11, 'QuadCost':0.2, 'Cost_Res_Proc':7, 'epsilon':0.05},
    'g6':{'Pmin':0, 'Pmax':155, 'BusNum':'n1', 'Rmax':30, 'LinCost':10.52, 'QuadCost':0.015, 'Cost_Res_Proc':16, 'epsilon':0.05},
    'g7':{'Pmin':0, 'Pmax':155, 'BusNum':'n1', 'Rmax':30, 'LinCost':10.52, 'QuadCost':0.015, 'Cost_Res_Proc':16, 'epsilon':0.05},
    'g8':{'Pmin':0, 'Pmax':400, 'BusNum':'n1', 'Rmax':0, 'LinCost':6.02, 'QuadCost':0.01, 'Cost_Res_Proc':0, 'epsilon':0.05},
    'g9':{'Pmin':0, 'Pmax':400, 'BusNum':'n1', 'Rmax':0, 'LinCost':5.47, 'QuadCost':0.01, 'Cost_Res_Proc':0, 'epsilon':0.05},
    'g10':{'Pmin':0, 'Pmax':300, 'BusNum':'n1', 'Rmax':0, 'LinCost':0, 'QuadCost':0.01, 'Cost_Res_Proc':0, 'epsilon':0.05},
    'g11':{'Pmin':0, 'Pmax':310, 'BusNum':'n1', 'Rmax':60, 'LinCost':10.52, 'QuadCost':0.015, 'Cost_Res_Proc':17, 'epsilon':0.05},
    'g12':{'Pmin':0, 'Pmax':350, 'BusNum':'n1', 'Rmax':40, 'LinCost':10.89, 'QuadCost':0.015, 'Cost_Res_Proc':16, 'epsilon':0.05}
    },

    'WindFarm':{
    'k1':{'Wmax':200, 'BusNum': 'n1'},
    'k2':{'Wmax':200, 'BusNum': 'n1'},
    'k3':{'Wmax':200, 'BusNum': 'n1'},
    'k4':{'Wmax':200, 'BusNum': 'n1'},
    'k5':{'Wmax':200, 'BusNum': 'n1'},
    'k6':{'Wmax':200, 'BusNum': 'n1'}
    },

    'WindFactor':{
	't1':{'k1':0.8,'k2':0.9,'k3':0.8,'k4':0.9,'k5':0.8,'k6':0.9},
	't2':{'k1':0.9,'k2':0.85,'k3':0.9,'k4':0.85,'k5':0.9,'k6':0.85},
	't3':{'k1':0.85,'k2':0.75,'k3':0.85,'k4':0.75,'k5':0.85,'k6':0.75},
	't4':{'k1':0.6,'k2':0.7,'k3':0.85,'k4':0.75,'k5':0.85,'k6':0.75},
	't5':{'k1':0.7,'k2':0.8,'k3':0.7,'k4':0.8,'k5':0.7,'k6':0.8},
	't6':{'k1':0.68,'k2':0.58,'k3':0.68,'k4':0.58,'k5':0.68,'k6':0.58},
	't7':{'k1':0.5,'k2':0.56,'k3':0.5,'k4':0.56,'k5':0.5,'k6':0.56},
	't8':{'k1':0.3,'k2':0.4,'k3':0.3,'k4':0.4,'k5':0.3,'k6':0.4},
	't9':{'k1':0.45,'k2':0.2,'k3':0.45,'k4':0.2,'k5':0.45,'k6':0.2},
	't10':{'k1':0.85,'k2':0.75,'k3':0.85,'k4':0.75,'k5':0.85,'k6':0.75},
	't11':{'k1':0.7,'k2':0.75,'k3':0.7,'k4':0.75,'k5':0.7,'k6':0.75},
	't12':{'k1':0.15,'k2':0.35,'k3':0.15,'k4':0.35,'k5':0.15,'k6':0.35},
	't13':{'k1':0.2,'k2':0.25,'k3':0.2,'k4':0.25,'k5':0.2,'k6':0.25},
	't14':{'k1':0.4,'k2':0.3,'k3':0.4,'k4':0.3,'k5':0.4,'k6':0.3},
	't15':{'k1':0.35,'k2':0.25,'k3':0.35,'k4':0.25,'k5':0.35,'k6':0.25},
	't16':{'k1':0.65,'k2':0.7,'k3':0.65,'k4':0.7,'k5':0.65,'k6':0.7},
	't17':{'k1':0.5,'k2':0.2,'k3':0.5,'k4':0.2,'k5':0.5,'k6':0.2},
	't18':{'k1':0.25,'k2':0.1,'k3':0.25,'k4':0.1,'k5':0.25,'k6':0.1},
	't19':{'k1':0.1,'k2':0.15,'k3':0.1,'k4':0.15,'k5':0.1,'k6':0.15},
	't20':{'k1':0.2,'k2':0.15,'k3':0.2,'k4':0.15,'k5':0.2,'k6':0.15},
	't21':{'k1':0.15,'k2':0.1,'k3':0.15,'k4':0.1,'k5':0.15,'k6':0.1},
	't22':{'k1':0.15,'k2':0.1,'k3':0.15,'k4':0.1,'k5':0.15,'k6':0.1},
	't23':{'k1':0.5,'k2':0.7,'k3':0.5,'k4':0.7,'k5':0.5,'k6':0.7},
	't24':{'k1':0.65,'k2':0.77,'k3':0.65,'k4':0.77,'k5':0.65,'k6':0.77}},

    'Demand':{
      't1':{'EL':1775.835},
      't2':{'EL':1669.815,'prev':'t1'},
      't3':{'EL':1590.3,'prev':'t2'},
      't4':{'EL':1563.795,'prev':'t3'},
      't5':{'EL':1563.795,'prev':'t4'},
      't6':{'EL':1590.3,'prev':'t5'},
      't7':{'EL':1961.37,'prev':'t6'},
      't8':{'EL':2279.43,'prev':'t7'},
      't9':{'EL':2517.975,'prev':'t8'},
      't10':{'EL':2544.48,'prev':'t9'},
      't11':{'EL':2544.48,'prev':'t10'},
      't12':{'EL':2517.975,'prev':'t11'},
      't13':{'EL':2517.975,'prev':'t12'},
      't14':{'EL':2517.975,'prev':'t13'},
      't15':{'EL':2464.965,'prev':'t14'},
      't16':{'EL':2464.965,'prev':'t15'},
      't17':{'EL':2623.995,'prev':'t16'},
      't18':{'EL':2650.5,'prev':'t17'},
      't19':{'EL':2650.5,'prev':'t18'},
      't20':{'EL':2544.48,'prev':'t19'},
      't21':{'EL':2411.955,'prev':'t20'},
      't22':{'EL':2199.915,'prev':'t21'},
      't23':{'EL':1934.865,'prev':'t22'},
      't24':{'EL':1669.815,'prev':'t23'}},

    'ELDemand':{
	'd1' : {'share' : 1.0,'node': 'n1'}},

    'Lines': {
    'l1' : {'From' : 'n1', 'To' : 'n2','B' : 0.0146, 'capacity' : 175},
    'l2' : {'From' : 'n1', 'To' : 'n3','B' : 0.2253, 'capacity' : 175},
    'l3' : {'From' : 'n1', 'To' : 'n5','B' : 0.0907, 'capacity' : 500},
    'l4' : {'From' : 'n2', 'To' : 'n4','B' : 0.1356, 'capacity' : 175},
    'l5' : {'From' : 'n2', 'To' : 'n6','B' : 0.205, 'capacity' : 175},
    'l6' : {'From' : 'n3', 'To' : 'n9','B' : 0.1271, 'capacity' : 175},
    'l7' : {'From' : 'n3', 'To' : 'n24','B' : 0.084, 'capacity' : 400},
    'l8' : {'From' : 'n4', 'To' : 'n9','B' : 0.111, 'capacity' : 175},
    'l9' : {'From' : 'n5', 'To' : 'n10','B' : 0.094, 'capacity' : 500},
    'l10' : {'From' : 'n6', 'To' : 'n10','B' : 0.0642, 'capacity' : 175},
    'l11' : {'From' : 'n7', 'To' : 'n8','B' : 0.0652, 'capacity' : 1000},
    'l12' : {'From' : 'n8', 'To' : 'n9','B' : 0.1762, 'capacity' : 175},
    'l13' : {'From' : 'n8', 'To' : 'n10','B' : 0.1762, 'capacity' : 175},
    'l14' : {'From' : 'n9', 'To' : 'n11','B' : 0.084, 'capacity' : 400},
    'l15' : {'From' : 'n9', 'To' : 'n12','B' : 0.084, 'capacity' : 400},
    'l16' : {'From' : 'n10', 'To' : 'n11','B' : 0.084, 'capacity' : 400},
    'l17' : {'From' : 'n10', 'To' : 'n12','B' : 0.084, 'capacity' : 400},
    'l18' : {'From' : 'n11', 'To' : 'n13','B' : 0.0488, 'capacity' : 500},
    'l19' : {'From' : 'n11', 'To' : 'n14','B' : 0.0426, 'capacity' : 500},
    'l20' : {'From' : 'n12', 'To' : 'n13','B' : 0.0488, 'capacity' : 500},
    'l21' : {'From' : 'n12', 'To' : 'n23','B' : 0.0985, 'capacity' : 500},
    'l22' : {'From' : 'n13', 'To' : 'n23','B' : 0.0884, 'capacity' : 500},
    'l23' : {'From' : 'n14', 'To' : 'n16','B' : 0.0594, 'capacity' : 500},
    'l24' : {'From' : 'n15', 'To' : 'n16','B' : 0.0172, 'capacity' : 500},
    'l25' : {'From' : 'n15', 'To' : 'n21','B' : 0.0249, 'capacity' : 500},
    'l26' : {'From' : 'n15', 'To' : 'n24','B' : 0.0529, 'capacity' : 500},
    'l27' : {'From' : 'n16', 'To' : 'n17','B' : 0.0263, 'capacity' : 500},
    'l28' : {'From' : 'n16', 'To' : 'n19','B' : 0.0234, 'capacity' : 500},
    'l29' : {'From' : 'n17', 'To' : 'n18','B' : 0.0143, 'capacity' : 500},
    'l30' : {'From' : 'n17', 'To' : 'n22','B' : 0.1069, 'capacity' : 500},
    'l31' : {'From' : 'n18', 'To' : 'n21','B' : 0.0132, 'capacity' : 1000},
    'l32' : {'From' : 'n19', 'To' : 'n20','B' : 0.0203, 'capacity' : 1000},
    'l33' : {'From' : 'n20', 'To' : 'n23','B' : 0.0112, 'capacity' : 1000},
    'l34' : {'From' : 'n21', 'To' : 'n22','B' : 0.0692, 'capacity' : 500}},
    }
    return solution


# Program for generating RT wind realization scenarios
def generate_wind_RT_realizations(data,T,NumSimCases,bias,Sigma_baseline_Multiplier):
    #Function to generate wind realizations to be picked up from a given probability distribution, based on whether biased is set to 1
    K = [k for k, k_info in data['WindFarm'].items()]
    #column_names = ['Wind Farm Number', 'Hour of the Day', 'Simulation Number','WPP Forecast (MW)', 'WPP Actual (MW)']
    column_names =['Wind Farm Number', 'Hour of the Day', 'Scenario Number','WPP Forecast(MW)','WPP Actual(MW)','Error']
    df_main = pd.DataFrame(columns = column_names)
    for k in K:
        sigma_k_baseline = Sigma_baseline_Multiplier*data['WindFarm'][k]['Wmax']     #Std Deviation is considered 7.5% of the max capacity
        for hour in T:
            PointForecast = data['WindFarm'][k]['Wmax']*data['WindFactor'][hour][k]
            sigma_k_hourly = np.random.normal(sigma_k_baseline,0.1*sigma_k_baseline,1)
            DataToAdd = np.random.normal(PointForecast,sigma_k_hourly,NumSimCases)
            DataToAdd=np.clip(DataToAdd,0.0,data['WindFarm'][k]['Wmax'])
            ForcError = DataToAdd - np.tile(PointForecast,NumSimCases)
            df2 = pd.DataFrame({'Wind Farm Number': np.tile(k,NumSimCases), 'Hour of the Day':np.tile(hour,NumSimCases),'Scenario Number':range(NumSimCases),'WPP Forecast(MW)':np.tile(PointForecast,NumSimCases),'WPP Actual(MW)':DataToAdd,'Error':ForcError})
            df_main = df_main.append(df2,ignore_index=True,sort=False)
    return df_main


''' ---- GATHERING THE SYSTEM DATA and EXTRACTING NEEDED DIMENSIONS FROM SYSTEM DATA ---- '''
SystemData = input()
G = [g for g, g_info in SystemData['FlexGens'].items()]
K = [k for k, k_info in SystemData['WindFarm'].items()]
D = [d for d, d_info in SystemData['Demand'].items()]
T = [t for t, t_info in SystemData['Demand'].items()]
N = list(SystemData['BusNums'])


'''---- A. BIASED OOS ----'''
SBM = 0.1
bias = True
NumSimCases = 1000

#Importing Stored Scenarios, if available
OOS_Scenarios = pd.read_csv('WindForecast_Errors_1000Scenarios.csv')   #Uncomment if reading from a given set of scenarios

# Uncomment if Generating Scenarios
#OOS_Scenarios = generate_wind_RT_realizations(SystemData,T,NumSimCases,bias,Sigma_baseline_Multiplier=SBM)
#FileName = 'WindForecast_Errors_1000Scenarios.csv'                #Saving Scenarios_to_CSV_File
#export_csv = OOS_Scenarios.to_csv(FileName,index=None,header=True)

#Running Pdet: Deterministic Testbench for the given data and scenarios: Both Day Ahead (once) and Real-time Optimization Problems (NumSimCases times)  need to be called in sequence
Pdet_RTRedispatchCost_with_MRR = []
MRR=200     #Value of Minimum Reserve Requirement set 
Pdet_DACost,status,varData,Pdet_lambda_DA_vals,Pdet_lambda_RE_vals,Pdet_GenProd,Pdet_Reserves,WindForecast,NetLoad,Pdet_DAEnergyCost,Pdet_DAReserveProcCost,Pdet_DA_EnergyCostPerGen,Pdet_DA_ReserveCostPerGen = clearDAMarket_and_Reserves(SystemData,MRR)
print('Pdet_EnergyEarningsPerGenerator: {}'.format(Pdet_DA_EnergyCostPerGen))
print('Pdet_ReservePolicyCostsPerGenerator: {}'.format(Pdet_DA_ReserveCostPerGen))

#Collecting the optimal dispatch and reserve allocation from DA stage
Pdet_DA_Opt_Dispatch={}
Pdet_DA_Opt_Reserves={}
for g in G:
    for t in T:
        G_index=int(g.strip('g'))-1
        T_index=int(t.strip('t'))-1
        Pdet_DA_Opt_Dispatch[(g,t)] = Pdet_GenProd[G_index][T_index]
        Pdet_DA_Opt_Reserves[(g,t)] = Pdet_Reserves[G_index][T_index]
#Running the Real-time Optimization program NumSimCases times
Pdet_RTRedispatchCost = []       #List to store the Redispatch Cost
Pdet_Total_WindSpillage = []
Pdet_Total_LoadShedding = []
Pdet_TotalRTAdjustmentEnergyCosts = []
for s in range(NumSimCases):
    print('Scenario Num: {}'.format(s))
    Delta={}
    for t in T:
        Delta[t]=OOS_Scenarios[(OOS_Scenarios['Hour of the Day'] == t) & (OOS_Scenarios['Scenario Number'] == s)]['Error'].sum()
    Pdet_RTCost,Status,VariablesValue,Pdet_RT_Price,Pdet_WindSpilled,Pdet_LoadShed,Pdet_Adjustments = clearRTBalancingMarket(SystemData,Delta,Pdet_DA_Opt_Dispatch,Pdet_DA_Opt_Reserves)
    Pdet_RTRedispatchCost.append(Pdet_RTCost)
    print('Done! \n -----')
    Pdet_Total_WindSpillage.append(Pdet_WindSpilled)
    Pdet_Total_LoadShedding.append(Pdet_LoadShed)
    #Calculating Per Generator Adjustment Payments
    Pdet_AdjustmentCostsPerGenerator = []
    for iter in range(len(Pdet_Adjustments)):
        Pdet_AdjustmentCostsPerGenerator.append([np.inner(Pdet_Adjustments[iter],Pdet_lambda_DA_vals)])
    Pdet_TotalRTAdjustmentEnergyCosts.append(np.sum(Pdet_AdjustmentCostsPerGenerator))
    print('Pdet_Adjustment Payments Made to Generators: {}'.format(Pdet_AdjustmentCostsPerGenerator))
Pdet_RTRedispatchCost_with_MRR.append(Pdet_RTRedispatchCost)

plt.boxplot(Pdet_RTRedispatchCost_with_MRR)
plt.show()

#Running Pcc: Chance Constrained Co-optimization for the given data and scenarios: Day Ahead (once) and Real-time Optimization Settlemetn (NumSimCases times)
Pcc_DACost,status,varData,Pcc_lambda_DA_vals,Pcc_lambda_RE_vals,Pcc_GenProd,Pcc_AlphaG,WindForecast,NetLoad,Pcc_DAEnergyCost,Pcc_DAReserveProcCost,Pcc_DA_EnergyCostPerGen,Pcc_DA_ReserveCostPerGen = clearDAMarket_and_Reserves_CC(SystemData,Sigma_baseline_Multiplier=0.3)
print('------------')
print('Pcc_EnergyEarningsPerGenerator: {}'.format(Pcc_DA_EnergyCostPerGen))
print('Pcc_ReservePolicyCostsPerGenerator: {}'.format(Pcc_DA_ReserveCostPerGen))

#Collecting the optimal dispatch and reserve allocation from DA stage
Pcc_DA_Opt_Dispatch = {}
Pcc_DA_Opt_AlphaVals = {}
for g in G:
    for t in T:
        G_index=int(g.strip('g'))-1
        T_index=int(t.strip('t'))-1
        Pcc_DA_Opt_Dispatch[(g,t)] = Pcc_GenProd[G_index][T_index]
        Pcc_DA_Opt_AlphaVals[(g,t)] = Pcc_AlphaG[G_index][T_index]
#Running the Real-time settlement program NumSimCases times
Pcc_RTRedispatchCost = []
Pcc_Total_WindSpillage = []
Pcc_Total_LoadShedding =[]
Pcc_TotalRTAdjustmentEnergyCosts = []
for s in range(NumSimCases):
    Delta={}
    for t in T:
        Delta[t]=OOS_Scenarios[(OOS_Scenarios['Hour of the Day'] == t) & (OOS_Scenarios['Scenario Number'] == s)]['Error'].sum()
    print('Deviation = {}'.format(Delta))
    Pcc_RTCost, Pcc_NetGeneration, Pcc_WindSpilled,Pcc_LoadShed,Pcc_Adjustments = runRealTimeSimulations(SystemData,Delta,Pcc_DA_Opt_Dispatch,Pcc_DA_Opt_AlphaVals)
    Pcc_RTRedispatchCost.append(Pcc_RTCost)
    Pcc_Total_WindSpillage.append(Pcc_WindSpilled)
    Pcc_Total_LoadShedding.append(Pcc_LoadShed)
    #Calculating Per Generator Adjustment Payments
    Pcc_AdjustmentCostsPerGenerator = []
    for iter in range(len(Pcc_Adjustments)):
        Pcc_AdjustmentCostsPerGenerator.append([np.inner(Pcc_Adjustments[iter],Pcc_lambda_DA_vals)])
    Pcc_TotalRTAdjustmentEnergyCosts.append(np.sum(Pcc_AdjustmentCostsPerGenerator))
    print('Pcc_Adjustment Payments Made to Generators: {}'.format(Pcc_AdjustmentCostsPerGenerator))

''' --------PLOTTING RESULTS and COMPARISONS ------- '''
#Plot1: Scenario-wise comparison of total cost end of day between Pdet and Pcc
plt.plot(Pdet_RTRedispatchCost, color='blue', linewidth=2, alpha=0.8, label='Deterministic')
plt.plot(Pcc_RTRedispatchCost, color='red', linewidth=2, alpha=0.8, label='Chance-Constrained')
plt.legend(loc='upper right')
plt.show()


#Table 1: End_of_Day Expected Cost
print('=========Pdet Costs : DAY AHEAD========')
print('Pdet_DA_ReserveCost = {}'.format(Pdet_DAReserveProcCost*1e-3))
print('Pdet_DA_EnergyCost = {}'.format(Pdet_DAEnergyCost*1e-3))
print('Pdet_DA_ObjectiveValue = {}'.format(Pdet_DACost*1e-3))
print('Pdet_DA_Prices Lambda_DA:{}'.format(np.round(Pdet_lambda_DA_vals,2)))
print('=========Pdet Costs : REAL TIME ADJUSTMENTS - EXPECTATION========')
print('Pdet_RT_Redispatch_Cost = {}'.format(np.mean(Pdet_RTRedispatchCost)*1e-3))
print('Pdet_Expected_Total_System_Costs = {}'.format((np.mean(Pdet_RTRedispatchCost)+Pdet_DAReserveProcCost)*1e-3))
print('Pdet_Total_RT_Adjustment_Payments_to_Gens = {}'.format(Pdet_TotalRTAdjustmentEnergyCosts))
print('=========================')
print('=========Pdet SYSTEM PARAMETERS========')
print('Pdet_Total_LoadShed: {}'.format(Pdet_Total_LoadShedding))
print('Pdet_Total_WindSpilled: {}'.format(Pdet_Total_WindSpillage))


print('=========Pcc Costs========')
print('Pcc_DA_ReserveCost = {}'.format(Pcc_DAReserveProcCost*1e-3))
print('Pcc_DA_EnergyCost = {}'.format(Pcc_DAEnergyCost*1e-3))
print('Pcc_DA_ObjectiveValue = {}'.format(Pcc_DACost*1e-3))
print('Pcc_DA_Prices Lambda_DA:{}'.format(np.round(Pcc_lambda_DA_vals,2)))
print('Pcc_ReservePolicy_Prices Lambda_RE:{}'.format(np.round(Pcc_lambda_RE_vals,2)))
print('=========Pcc Costs : REAL TIME ADJUSTMENTS - EXPECTATION========')
print('Pcc_RT_Redispatch_Cost = {}'.format(np.mean(Pcc_RTRedispatchCost)*1e-3))
print('Pcc_Expected_Total_System_Costs = {}'.format((np.mean(Pcc_RTRedispatchCost)+Pcc_DAReserveProcCost)*1e-3))
print('Pcc_Total_RT_Adjustment_Payments_to_Gens = {}'.format(Pcc_TotalRTAdjustmentEnergyCosts))
print('=========================')
print('=========Pcc SYSTEM PARAMETERS========')
print('Pcc_Total_LoadShed: {}'.format(np.sum(Pcc_Total_LoadShedding)))
print('Pcc_Total_WindSpilled: {}'.format(np.sum(Pcc_Total_WindSpillage)))
