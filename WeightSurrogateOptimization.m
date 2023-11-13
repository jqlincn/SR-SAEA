function Arc= WeightSurrogateOptimization(Problem,G1,Population,Rs)
% Surrogate-assisted optimization

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jianqing Lin

	%% Choose Rs solutions as the reference solutions
    [RefPop,~,~] = EnvironmentalSelection(Population,Rs);   
    RBFPOP       = Population;
    
    %% Calculate the reference directions
	lower     = Problem.lower;
    upper     = Problem.upper;
	Direction = [sum((RefPop.decs-repmat(lower,Rs,1)).^2,2).^(0.5);sum((repmat(upper,Rs,1)-RefPop.decs).^2,2).^(0.5)];
	Direct    = [(RefPop.decs-repmat(lower,Rs,1));(repmat(upper,Rs,1)-RefPop.decs)]./repmat(Direction,1,Problem.D);  
	lmax      = sum((upper-lower).^2)^(0.5)*0.5;
    
    %% Initialize the lambda population
    [LPopNew,NewPop]    = problemRef(Problem,RBFPOP,Direct,Direction,RefPop,Rs);                  % Problem Reformulation

    %% Create rbf models
    LpopDecs = LPopNew.decs;
    LpopObjs = LPopNew.objs;
    for i = 1:Problem.M
        RBF_para{i} = RBFCreate(LpopDecs, LpopObjs(:,i), 'cubic');
    end
    
    %% Create DAE
    dae = SupervisedAutoEncoder(size(Population.decs,2),2*Rs,50,size(Population.decs,1),0.5,0.5,0.1);
    dae.train(Population.decs,NewPop);
    
    %% RM-MEDA
    Population = Surrogate_individual(LpopDecs,LpopObjs);
    for g = 1:G1
        % Parameter setting
        K = 5;
        PopDec = Population.decs;
        [N,D]  = size(PopDec);
        M      = length(Population(1).obj);

        % EDA Modeling
        [Model,probability] = LocalPCA(PopDec,M,K);

        % Reproduction
        OffspringDec = zeros(N,D);
        % Generate new trial solutions one by one
        for i = 1 : N
            % Select one cluster by Roulette-wheel selection
            k = find(rand<=probability,1);
            % Generate one offspring
            if ~isempty(Model(k).eVector)
                lower = Model(k).a - 0.25*(Model(k).b-Model(k).a);
                upper = Model(k).b + 0.25*(Model(k).b-Model(k).a);
                trial = rand(1,M-1).*(upper-lower) + lower;
                sigma = sum(abs(Model(k).eValue(M:D)))/(D-M+1);
                OffspringDec(i,:) = Model(k).mean + trial*Model(k).eVector(:,1:M-1)' + randn(1,D)*sqrt(sigma);
            else
                OffspringDec(i,:) = Model(k).mean + randn(1,D);
            end
        end
        OffspringObj = Surrogate_Predictor(OffspringDec, RBF_para, Problem.M);
        Offspring    = Surrogate_individual(OffspringDec,OffspringObj);
        Population   = EnvironmentalSelection([Population,Offspring],Problem.N);
    end

    %Update and store the non-dominated solutions
    [Arc,~,~] = EnvironmentalSelection(Population,Rs);
    Arc       = dae.recover(Arc.decs);
    Arc       = Problem.Evaluation(Arc);
end


function [LPopNew,NewPop] = problemRef(Problem,Population,Direct,ROTDist,RefPop,Rs)

    PopDec  = Population.decs;
    PopObj  = Population.objs;
    RefDec  = RefPop.decs;
    O    = Problem.lower;
    T    = Problem.upper;
    
    NewPop = [];
    for i = 1:size(PopDec,1)
        PRDec = [];
        for j = 1:Rs
            Proj  = projPointOnLine(PopDec(i,:),O,Direct(j,:));
            PRDec = [PRDec;Proj];
        end
        for j = Rs+1:2*Rs
            Proj  = projPointOnLine(PopDec(i,:),T,Direct(j,:));
            PRDec = [PRDec;Proj];
        end
        POTDist  = diag(pdist2(PRDec,[repmat(O,Rs,1);repmat(T,Rs,1)]));
        lDec     = ROTDist-POTDist;
        NewPop    = [NewPop;lDec'];
    end
    LPopNew = Surrogate_individual(NewPop,PopObj);
end

function ProjPoint = projPointOnLine(point,PassPoint,W)
% Point  ----------- candidate point
% PassPoint  ------- passing point
% W  --------------- direction vector

        % Convert the point into coordinates with PassPoint as the origin.
        point = point - PassPoint;
        
        B   = W';
        X   = point'; 
        BTB = B'*B;
        Y   = B*(BTB\(B'*X));
        ProjPoint = Y'+PassPoint;  
end






