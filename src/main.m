%% main.m
% ===========================================================================
%  CONAC-CVL CONTROL SIMULATION MAIN SCRIPT
%  Original Author: Myeongseok Ryu
%  Modified by: Samuel Naol Samuel
%  Last Modified: 2025:12:08
% ===========================================================================   
%% 
clear;
clc;
close all;
addpath("utils")

%% SIMULATION SETTING
paramSim.saveResult  =   0;
paramSim.saveFigFile =   0;
paramSim.saveNetwork =   0;

% paramSim.seed_num =  1;
paramSim.seed_num =  130;

%% SIMULATION PARAMETERS
paramSim.dt = 1e-2;                                                 % SAMPLING TIME STEP
paramSim.T = 5;                                                     % TERMINAL TIME
t = 0:paramSim.dt:paramSim.T;
rpt_dt = 1;

x = [0 0]';                                                         % INITIAL STATE, in R^2
u = [0 0]';                                                         % INITIAL INPUT, in R^2

%% SYSTEM DECLARE
sat = @(u) min(max(u, -100), 100);                                  % SATURATION FUNCTION
f_x = @(x) [                                                        % DRIFT DYNAMICS
    x(1)*x(2)*tanh(x(2))+sech(x(1))
    sech(x(1)+x(2))^2-sech(x(2))^2
];
g_x = @(x) [                                                        % INPUT GAIN MATRIX, POSITIVE DEFINITE
    1 , 0;
    0, 1 
];
grad_x = @(x, u, t) f_x(x) + g_x(x) * sat(u);


%% PASSIVE PARAMETERS
rng(paramSim.seed_num);
paramSim.exp_name = datetime('now','TimeZone','local', ...
    'Format','yyMMdd_HHmmss');

%% REFERENCE
%ref_Traj = @(t) [0;0];
% 
 ref_Traj = @(t)[ % desired trajectory
     sin(2*t)-cos(1.5*t)
     2*cos(2*t) + 1.5 * sin(1.5*t)
     ] * 1e0;    

%% NEURAL NETWORK DECLARE
NN = paramCtrl_load(paramSim);
NN = init_NN(NN);
NN.paramCtrl.f_x = f_x;                                             % STORE f_x IN NN PARAMCTRL
NN.paramCtrl.g_x = g_x;
%% REPORT SIMULATION SETTING
reportSim(NN, paramSim);

%% RECORDER
recordPrepare     
                                                                    % PREPARE TRAJECTORY RECORDERS
%% MAIN LOOP
dataset_x = zeros( ...                                              % BUFFER FROM WHICH CVL INPUTS ARE SAMPLED
    NN.paramCtrl.size_CVL_input(1)*int64(NN.paramCtrl.input_dt/NN.paramCtrl.dt), ...
    NN.paramCtrl.size_CVL_input(2));

fprintf("===========================================\n")
fprintf("             SIMULATION START              \n")
fprintf("===========================================\n")
fprintf("\n")
    
try 
    for t_idx = 2:1:length(t)
        %% ERROR CALCULATION
        xd = ref_Traj(t(t_idx));
        e = x - xd;
    
        %% CONTROL LAW CALCULATION
        [NN_Out, NN, dataset_x] = NNforward(NN, x, xd, u, dataset_x, t(t_idx));
        u = -1 * NN_Out;                                         %  u = -Phi_hat

        %% SYSTEM STEP
        x = x + grad_x(x, u, t(t_idx)) * paramSim.dt;
    
        %% NEURAL NETWORK TRAINING
        NN = NNtrain(NN, e);
    
        %% RECORDING
        result.X_hist(:, t_idx) = x;
        result.XD_hist(:, t_idx) = xd;
        result.U_hist(:, t_idx) = u;
        result.E_hist(:, t_idx) = e;
       
        if NN.paramCtrl.CVLon                                       % RECORD WEIGHTS NORM OF CVL: Om and Om_B combined 
            for Om_idx = 1:1:NN.paramCtrl.CVL_num+1
                Om = NN.("Omega"+string(Om_idx-1));
                B = NN.("Omega_B"+string(Om_idx-1));
                Om_flat = Om(:);
                B_flat = B(:);
                OmB_flat = [Om_flat; B_flat];
                result.Om_hist.("Om_Combined"+string(Om_idx-1))(1, t_idx) = ...
                    norm(OmB_flat, "fro");
            end
        end
        
        for V_idx = 1:1:NN.paramCtrl.FCL_num+1                      % RECORD WEIGHTS NORM OF FCL: V
            result.V_hist(V_idx, t_idx) = norm(NN.("V"+string(V_idx-1)), "fro");
        end
       %% REPORTING
        if rem(t(t_idx)/paramSim.dt, rpt_dt/paramSim.dt) == 0
            fprintf("[INFO] Simulation Step %.2f/%.2fs (%.3f%%)\r", ...
                t(t(t_idx)/paramSim.dt) / paramSim.T, t(t_idx), paramSim.T, t(t_idx)/paramSim.T*100);
        end
        if isnan(x(1))
            error("states Inf")
        end
    end
catch whyStop
    fprintf("[ERROR] %s\n", whyStop.message)

    if paramSim.saveResult
        diary off
    end
    result.t = t;
    result.t_idx = t_idx;
    result.NN = NN;
    
    resultReportPlot(result, paramSim);
    return
end
%% PLOT AND REPORT
result.t = t;
result.t_idx = t_idx;
result.NN = NN;

fprintf("[INFO] Plotting Results\n\n")
resultReportPlot(result, paramSim);

%% DIARY OFF
if paramSim.saveResult
    diary off
end

%% NETWORK SAVE
if paramSim.saveResult && paramSim.saveNetwork
    NN.gradTape = [];                                                 % CLEAR GRADIENT TAPE BEFORE SAVING
    save(result_dir + "/NN.mat","NN")
    fprintf("[INFO] Network Saved\n\n")
end

%% TERMINATION
fprintf("===========================================\n")
fprintf("           SIMULATION TERMINATED           \n")
fprintf("===========================================\n")
fprintf("\n")