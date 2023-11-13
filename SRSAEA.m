classdef SRSAEA < ALGORITHM
% <multi> <real> <expensive>
% Supervised Reconstruction Surrogate-assisted Evolutionary Computation
% Rs --- 5 --- The generation of weight optimization with DE
% G1 --- 50 --- The generation of weight optimization with DE

%------------------------------- Reference --------------------------------
% H. Li, J. Lin, . C. He, Q. Chen and L. Pan, Supervised Reconstruction for 
% High-dimensional Expensive Multiobjective Optimization, IEEE Transactions 
% on  Emerging Topics in Computational Intelligence, 2023.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jianqing Lin
    methods
        function main(Algorithm,Problem)
            warning('off')
            %% Parameter settings
            [Rs,G1]     = Algorithm.ParameterSet(5,50);
            Num_Sample  = 11*2*Rs-1;
            PopDec      = repmat((Problem.upper - Problem.lower),Num_Sample, 1) .* lhsdesign(Num_Sample, Problem.D) + repmat(Problem.lower, Num_Sample, 1);  % LHS
            A           = Problem.Evaluation(PopDec);
            
            %% Optimization
            while Algorithm.NotTerminated(A)
                A1    = WeightSurrogateOptimization(Problem,G1,A,Rs);
                A     = [A, A1];
            end
        end
    end
end