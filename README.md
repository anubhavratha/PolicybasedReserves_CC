# Policy-based reserves in day-ahead electricity markets
This repository contains the python scripts accompanying the paper: "Exploring Market Properties of Policy-based Reserve Procurement for Power Systems" https://ieeexplore.ieee.org/abstract/document/9029777

If you are re-using parts of the code, please cite the work as:

A. Ratha, J. Kazempour, A. Virag and P. Pinson, "_Exploring Market Properties of Policy-based Reserve Procurement for Power Systems_," 2019 IEEE 58th Conference on Decision and Control (CDC), Nice, France, 2019, pp. 7498-7505, doi: 10.1109/CDC40024.2019.9029777.

## Instructions for navigating the programs:
1. The excel sheet "Generators_and_Load_Data.xlsx" contains the cost and operational constraints related to the generators and demand in the single-node electricity network considered in the case study.
2. The python scripts "Pcc_ChanceConstrained_ReserveAllocation.py" and "Pdet_Benchmark_Deterministic_Reserves.py" contain the optimization models corresponding to the proposed chance-constrained energy and reserve co-optimization problem and a deterministic co-optimization benchmark, respectively. Both these optimization problems are formulated as linear programs and are solved using Gurobi solver.
3. The python script "Run_OOS_Simulations_Pcc_Pdet.py" performs biased and unbiased out-of-sample simulations on the chance-constrained and deterministic benchmark and collects the results that compare these two approaches. The CSV file "WindForecast_Errors_1000Scenarios.csv" contains the synthetic forecast error scenarios used. The function "generate_wind_RT_realizations" can be used to generate new scenarios based on the preferred dispersion criteria, parametrized by the input float 'sigma_baseline_multiplier'.
