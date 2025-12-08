
%{

function NN = NNtrain(NN, error)
    %% PREPARE
    paramCtrl = NN.paramCtrl;                                                                 % PARAMETER CONTROL STRUCTURE
    gradTape = NN.gradTape;                                                                   % GRADIENT TAPE FROM FORWARD PASS
    CVLon = paramCtrl.CVLon;                                                                  % CVL ON/OFF
    dt = paramCtrl.dt;                                                                        % SIMULATION TIME STEP
    out_num = paramCtrl.size_FCL_output;                                                      % NUMBER OF OUTPUT NODES
    FCL_num = paramCtrl.FCL_num;                                                              % INDEX OF FCL WEIGHTS: 0 TO FCL_num
    Phi=paramCtrl.NN_Out;                                                                     % NN OUTPUT FROM FORWARD PASS
    if CVLon
        CVL_num = paramCtrl.CVL_num;
        CVL_Node = paramCtrl.CVL_Node;
    end
    Gamma = paramCtrl.Gamma;                                                                  % LEARNING RATE   
    Beta= paramCtrl.Beta;                                                                     % LAGRANGE MULTIPLIER LEARNING RATE
    Om_norms = paramCtrl.Om_norms;
    V_norms = paramCtrl.V_norms;
    NN_Out_norm = paramCtrl.NN_Out_norm;
    Lamda_Om= NN.paramCtrl.Lambda_Om;                                                         % LAMBDA for CVL weights, initialized to zero
    Lamda_V= NN.paramCtrl.Lambda_V;                                                           % LAMBDA for FCL weights, initialized to zero
    Lambda_Out= NN.paramCtrl.Lambda_Out;                                                        % LAMBDA for Output norm, initialized to zero
    

    %% BACK-PROPAGATION
    % FCL train ====================================================
    dPhidphi = 1;
    for nn_idx = flip(0:1:FCL_num) 
        % --- PREPARE NETWORK VARIABLES AND CONSTANTS ---
        V = NN.("V"+string(nn_idx));
        [n,m] = size(V);
        phi = gradTape.("FCL_phi"+string(nn_idx));                                            % INPUT TO CURRENT WEIGHT MATRIX
        
        % ACCUMULATE DERIVATIVE CHAIN dPhi / dphi
        if nn_idx ~= FCL_num
            V_next = NN.("V"+string(nn_idx+1)); 
            phi_dot = gradTape.("FCL_phi_dot"+string(nn_idx+1));
            dPhidphi = dPhidphi * (V_next'*phi_dot);
        end
        
        % --- GRADIENT CALCULATION AND WEIGHT UPDATE ---
        dPhi_dvecV_T = dPhidphi * kron(eye(m), phi');                                          % MULTIPLY WITH ACCUMULATED DERIVATIVE CHAIN
        vecV = reshape(V, [], 1);
        
        grad_vecV= - Gamma*(dPhi_dvecV_T * error-...                                           % GRADIENT CALCULATION`
        Lamda_V(nn_idx)*vecV- Lambda_Out* dPhi_dvecV_T * Phi); 
        
        Lamda_V(nn_idx)= ...                                                                   % UPDATE LAGRANGE MULTIPLIER FOR V
        max(0, Lamda_V(nn_idx) + Beta * constraint_slack(vecV, V_norms(nn_idx)));
        
        vecV = vecV + grad_vecV * dt;                                                          % UPDATE WEIGHTS
        NN.("V"+string(nn_idx)) = reshape(vecV, n,m);
    end
   % CVL train ====================================================
    if CVLon
        % --- INITIALIZE FCL GRADIENT FOR CVL BACK-PROP ---
        dPhi_dFCLInput = dPhidphi * NN.V0'; 
        dPhi_dFCLInput = dPhi_dFCLInput(:,1:end-1);                                            % REMOVE BIAS TERM
     
        

        % INITIALIZE ACCUMULATORS
        CVLJacobians = struct(); 
        
        % LOOP BACKWARD THROUGH CVL LAYERS
        for idx = flip(0:1:CVL_num)
            
            % --- PREPARE NETWORK VARIABLES AND CONSTANTS ---
            phi = gradTape.("CVL_phi"+string(idx));                                            % jc INDEX REMOVED FOR BREVITY     
            phi_dot= gradTape.("CVL_phi_dot" + string(idx));
            Om = NN.("Omega"+string(idx));                     
            B = NN.("Omega_B"+string(idx));                    
            
            % PREPARE CONSTANTS FOR ADAPTATION LAW
            Gamma_C = Gamma;                     
            lambda_C = Lamda_Om(idx+1);          
            Om_norm = Om_norms(idx+1);           
            
            filter_num = size(B, 1);             
            p_jc = size(Om, 1);                                                                 % FILTER HEIGHT

            n_jcPlus1=CVL_Node(idx,1);  
            m_jc=CVL_Node(idx,2);
            n_jcPlus1 = CVL_Node(idx+1,1);                                                      % HEIGHT OF NEXT LAYER (idx+1)
            m_jcPlus1 = CVL_Node(idx+1,2);                                                      % WIDTH  OF NEXT LAYER (idx+1)      
            
            
            % LOOP OVER NETWORK OUTPUTS
            for out_idx = 1:1:out_num                                                         
                if idx == CVL_num                                                               % SETUP THE STRUCTURE OF dPhi_i/dPhi_jc
                    CVLJacobians.("dPhi_"+string(out_idx)+"_dPhi_jc")= ...
                        reshape(dPhi_dFCLInput(:, out_idx), CVL_Node(end,1), CVL_Node(end,2));
                end

                dPhi_i_dPhi_jc= CVLJacobians.("dPhi_"+string(out_idx)+"_dPhi_jc");
                % --- JACOBIAN CALCULATION (dPhi_i / dOmega, dPhi_i / dB, dPhi_i/ dphi_jc) ---
                dPhi_i_dOm = zeros(size(Om));                                                   % JACOBIAN OF Phi_i w.r.t OMEGA
                dPhi_i_dB = zeros(size(B));                                                     % JACOBIAN OF Phi_i w.r.t B
                for filter_idx = 1:1:filter_num
                    dPhi_i_dOm_lk = zeros(size(Om(:,:,filter_idx)));                            % JACOBIAN OF Phi_i w.r.t OMEGA_lk
                    dPhi_i_dB_lk = 0;                                                           % JACOBIAN OF Phi_i w.r.t B_lk   
                    for l_i = 1:1:n_jcPlus1                                                 
                        % dPhi_i/dOmega_lk
                        dPhi_i_dOm_lk= dPhi_i_dOm_lk + ...      
                            dPhi_i_dPhi_jc(l_i,filter_idx) * phi(l_i:l_i+p_jc-1, :);            % p_jc: filter height
                        
                        % dPhi_i/dB_lk
                        dPhi_i_dB_lk = dPhi_i_dB_lk + dPhi_i_dPhi_jc(l_i,filter_idx);
                    end 
                    % STORE FILTER-LEVEL JACOBIANS
                    dPhi_i_dOm(:,:,filter_idx) = dPhi_i_dOm_lk;
                    dPhi_i_dB(filter_idx) = dPhi_i_dB_lk;
                end


                % dPhi_i/dphi_jc
                dPhi_i_dphi_jc = zeros(size(phi));   
                for l_j = 1:1:m_jcPlus1
                    for filter_idx = 1:1:filter_num
                        W = Om(:,:,filter_idx);                                                % FILTER Number (filter_idx)
                        for l_i = 1:1:n_jcPlus1
                            dPhi_i_dPhi_jc_lilj = dPhi_i_dPhi_jc(l_i, l_j);
                            tmp= zeros(n_jc, m_jc);
                            tmp(l_i:l_i+p_jc -1,:) = W;
                            dPhi_i_dphi_jc=dPhi_i_dphi_jc+ dPhi_i_dPhi_jc_lilj* tmp;
                        end
                    end
                end
                % --- STORE % dPhi_i/dOmega, dPhi_i/dB,  % dPhi_i/dphi_jc ---
                CVLJacobians.("dPhi_"+string(out_idx)+"_dOm")=dPhi_i_dOm;
                CVLJacobians.("dPhi_"+string(out_idx)+"_dB")    = dPhi_i_dB;
                CVLJacobians.("dPhi_"+string(out_idx)+"_dphi_jc")  = dPhi_i_dphi_jc;

                 % --- BACK PROPAGATION of dPhi_i/dPhi_jc FOR THE NEXT LOOP ---
                CVLJacobians.("dPhi_"+string(out_idx)+"_dPhi_jc")=...
                    dPhi_i_dPhi_jc.*phi_dot;

            end 
            
            % --- APPLY ADAPTATION LAW (Primal Update) ---
            vecOmega = reshape(Om, [], 1);
            theta_OmB = [vecOmega; B]; 
            
            % Vectorize the Objective Gradient (dJ/dtheta_T)
            dJ_dtheta_T = [reshape(-reshape(dPhi_dOmega_accum, [], 1) * error', 1, numel(Om)); -dPhi_dB_accum' * error]'; 
            dJ_dtheta = dJ_dtheta_T'; 
            
            % Vectorize the Lambda_Phi term
            dPhi_dtheta_T_Phi = [reshape(dPhi_dOmega_T_Phi_accum, [], 1); dPhi_dB_T_Phi_accum];
            
            % Calculate Lambda terms
            Lambda_Om_term = lambda_C * theta_C;
            Lambda_Phi_term = Lambda_Out * dPhi_dtheta_T_Phi; 
            
            % Final adaptation rule (Primal Update)
            grad_theta_C = Gamma_C * (dJ_dtheta - Lambda_Om_term - Lambda_Phi_term);
            
            % Update Weights
            theta_C = theta_C + grad_theta_C * dt;
            
            % Reshape and store
            Om_size = numel(Om);
            NN.("Omega"+string(idx)) = reshape(theta_C(1:Om_size), size(Om));
            NN.("Omega_B"+string(idx)) = theta_C(Om_size+1:end);
            
            % --- DUAL UPDATE ---
            Lamda_Om(idx+1) = max(0, lambda_C + Beta * constraint_slack(theta_C, Om_norm));
            Lambda_Out = max(0, Lambda_Out + Beta * constraint_slack(Phi, NN_Out_norm)); 
        end % End of idx loop (CVL_num down to 0)

        
        
        % STORE UPDATED LAGRANGE MULTIPLIERS
        NN.paramCtrl.Lambda_Om = Lamda_Om;
        NN.paramCtrl.Lambda_V = Lamda_V;
        NN.paramCtrl.Lambda_Out = Lambda_Out;
    end
    
end

%% LOCAL FUNCTIONS
function c= constraint_slack(x, radius)                                              % CONSTRAINT FUNCTION: c <= 0
    c = norm(x)^2 - radius^2;
    c=c/2;
end

%}




