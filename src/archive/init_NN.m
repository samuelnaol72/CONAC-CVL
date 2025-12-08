function NN = init_NN(NN)
    %% NEURAL NETWORK WEIGHTS INITIALIZATION
    % XAVIER'S METHOD NOT APPLIED HERE.

    %% CVL INITIALIZATION
    if NN.paramCtrl.CVLon
        % UNIFORMLY INITIALIZED IN [-INIT_RANGE, INIT_RANGE] (XAVIER'S METHOD)
        initRangeCVL = NN.paramCtrl.initRangeCVL;

        for nn_idx = 1:1:NN.paramCtrl.CVL_num+1
            n = NN.paramCtrl.CVL_Node(nn_idx, 2);                   % INPUT WIDTH
            q = NN.paramCtrl.CVL_Node(nn_idx, 3);                   % FILTER HEIGHT
            r = NN.paramCtrl.CVL_Node(nn_idx, 4);                   % FILTER NUMBER
    
            % INITIALIZE THE FILTERS AND BIASES
            NN.("Omega"+string(nn_idx-1)) =...                      % OMEGA IS SET OF FILTERS (W)
            (rand(q,n,r) -0.5) * 2 * initRangeCVL;
            NN.("Omega_B"+string(nn_idx-1)) = ...                   % BIAS INITIALIZATION
            (rand(r, 1) -0.5) * 2 * initRangeCVL;

            % ACCUMULATE CVL WEIGHT NUMBERS
            NN.paramCtrl.CVL_weight_num = ...
            NN.paramCtrl.CVL_weight_num + n*q*r + r;
        end
    end

    %% FCL INITIALIZATION
    for nn_idx = 1:1:NN.paramCtrl.FCL_num+1
        n = NN.paramCtrl.FCL_Node(nn_idx);                         % INPUT SIZE
        m = NN.paramCtrl.FCL_Node(nn_idx+1);                       % OUTPUT SIZE
        
        % INITIALIZE
        initRangeFCL = NN.paramCtrl.initRangeFCL;
        NN.("V"+string(nn_idx-1)) = ...                            % V INITIALIZATION
        (rand(n+1,m) -0.5) * 2 * initRangeFCL;                     % V INCLUDES BIAS TERM, ROW OF V += 1
        
        % ACCUMULATE FCL WEIGHT NUMBERS
        NN.paramCtrl.FCL_weight_num = NN.paramCtrl.FCL_weight_num + n*m + m;
    end
end