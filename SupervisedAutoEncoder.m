classdef SupervisedAutoEncoder < handle
% Supervised AutoEncoder

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties(SetAccess = private)
        nVisible  = 0;
        nHidden   = 0;
        Epoch     = 10;
        BatchSize = 1;
        InputZeroMaskedFraction = 0.5;   
        Momentum  = 0.5;                 
        LearnRate = 0.1;
        WA        = [];
        WB        = [];
        PS1       = [];
        PS2       = [];
    end
    methods
        %% Constructor
        function obj = SupervisedAutoEncoder(nVisible,nHidden,Epoch,BatchSize,InputZeroMaskedFraction,Momentum,LearnRate)
            obj.nVisible  = nVisible;                                                 % Original dimension
            obj.nHidden   = nHidden;                                                  % Reduced dimension
            obj.Epoch     = Epoch;
            obj.BatchSize = BatchSize;
            obj.InputZeroMaskedFraction = InputZeroMaskedFraction;                    % Anti-noise parameters
            obj.Momentum  = Momentum;                                                 % Weight momentum factor
            obj.LearnRate = LearnRate;                                                % LearnRate
            obj.WA = (rand(nHidden,nVisible+1)-0.5)*8*sqrt(6/(nHidden+nVisible));     % Encoder Weight initialization
            obj.WB = (rand(nVisible,nHidden+1)-0.5)*8*sqrt(6/(nVisible+nHidden));     % Decoder Weight initialization
        end
        %% Train
        function train(obj,X,L)
            [X,obj.PS1] = mapminmax(X',0,1);
            [L,obj.PS2] = mapminmax(L',0,1);
            X     = X';
            L     = L';
            vW{1} = zeros(size(obj.WA));
            vW{2} = zeros(size(obj.WB));
            
             %Add noise to input (for use in denoising autoencoder)
            if(obj.InputZeroMaskedFraction ~= 0)
                theta = rand(size(X)) > obj.InputZeroMaskedFraction;      % Noise matrix
            else
                theta = true(size(X));
            end
            X_temp = X.*theta;
            X_temp = [ones(size(X,1),1),X_temp];      % Input of the network
            for i = 1 : obj.Epoch
                kk = randperm(size(X,1));            
                for batch = 1 : size(X,1)/obj.BatchSize
                    batch_x = X_temp(kk((batch-1)*obj.BatchSize+1:batch*obj.BatchSize),:);   
                    batch_l = L(kk((batch-1)*obj.BatchSize+1:batch*obj.BatchSize),:);         
                    batch_y = X(kk((batch-1)*obj.BatchSize+1:batch*obj.BatchSize),:);       

                    % Feedforward pass 
                    poshid = 1./(1+exp(-batch_x*obj.WA'));       
                    poshid1 = [ones(obj.BatchSize,1),batch_l];    
                    poshid2 = 1./(1+exp(-poshid1*obj.WB'));   

                    % BP
                    e1    = batch_y - poshid2;              % Error of D dimensions
                    e2    = batch_l - poshid;               % Error of K dimensions
                    
                    d{2}  = -e2.*(poshid.*(1-poshid));
                    d{3}  = -e1.*(poshid2.*(1-poshid2)); 
                    
                    for i = 1 : 2
                        if i+1 == 3
                            dW{i} = (d{i+1}'*poshid1/size(d{3},1));    %Weight update amount
                        else
%                             dW{i} = (d{i+1}(:,2:end)'*batch_x)/size(d{i+1},1);
                            dW{i} = (d{i+1}'*batch_x)/size(d{i+1},1);  %Weight update amount
                        end
                    end
                    for i = 1 : 2
                        dW{i} = obj.LearnRate*dW{i};
                         % Error momentum learning rate
                        if obj.Momentum > 0
                            vW{i} = obj.Momentum*vW{i} + dW{i};
                            dW{i} = vW{i};
                        end
                         % Weight Update
                        if i == 1
                            obj.WA = obj.WA - dW{i};
                        else
                            obj.WB = obj.WB - dW{i};
                        end
                    end
                end
            end
        end
        %% Reduce
        function H = reduce(obj,X)
            X = mapminmax('apply',X',obj.PS1)';
            H = 1./(1+exp(-X*obj.WA(:,2:end)'-repmat(obj.WA(:,1)',size(X,1),1)));
            H = mapminmax('reverse',H',obj.PS2)';
        end
        %% Recover
        function X = recover(obj,H)
            H = mapminmax('apply',H',obj.PS2)';
            X = 1./(1+exp(-H*obj.WB(:,2:end)'-repmat(obj.WB(:,1)',size(H,1),1)));
            X = mapminmax('reverse',X',obj.PS1)';
        end
    end
end