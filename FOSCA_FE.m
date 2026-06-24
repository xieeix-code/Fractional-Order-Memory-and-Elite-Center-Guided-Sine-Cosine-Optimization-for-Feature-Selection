function [Destination_fitness, Destination_position, Convergence_curve] = FOSCA2_FE(N, MaxFE, lb, ub, dim, fobj, opts)
% FOSCA2_FE
% Fractional elite-memory SCA with real function-evaluation counting.
%
% Main changes from FOSCA_GPT1:
% 1) Use deterministic wrapper evaluation instead of stochastic transfer inside the optimizer.
% 2) Use an unnormalized short-memory fractional term to provide mild damping and sparsity pressure.
% 3) Replace single-best guidance with a dynamic elite-center guide in the early stage and gbest guidance in the late stage.
% 4) Use a faster-decaying r1 schedule to reduce late-stage oscillation.
%
% Interface:
%   [bestFitness,bestPosition,curve] = FOSCA2_FE(N,MaxFE,lb,ub,dim,fobj,opts)
%
% Notes:
% - Each call to fobj(.) counts as one FE.
% - Convergence_curve(k) records the best-so-far fitness after the k-th FE.
% - Destination_position is continuous; the FS wrapper converts it to a binary subset.

    if nargin < 7 || isempty(opts)
        opts = struct();
    end

    alpha = getOption(opts, 'alpha', 0.82);
    alpha = min(max(alpha, 0.05), 0.99);

    useGreedy      = getOption(opts, 'useGreedy', true);
    useEliteCenter = getOption(opts, 'useEliteCenter', true);

    lbVec = expandBound(lb, dim);
    ubVec = expandBound(ub, dim);

    X = initialization(N, dim, ub, lb);
    X_prev1 = X;
    X_prev2 = X;

    Objective_values = inf(N, 1);
    Destination_fitness = inf;
    Destination_position = zeros(1, dim);

    Convergence_curve = inf(1, MaxFE);
    FE = 0;

    for i = 1:N
        Objective_values(i) = fobj(X(i, :));
        FE = FE + 1;

        if Objective_values(i) < Destination_fitness
            Destination_fitness = Objective_values(i);
            Destination_position = X(i, :);
        end

        Convergence_curve(FE) = Destination_fitness;
        if FE >= MaxFE
            return;
        end
    end

    c0 = alpha;
    c1 = 0.5 * alpha * (1 - alpha);
    c2 = (1/6) * alpha * (1 - alpha) * (2 - alpha);

    while FE < MaxFE
        tau = FE / MaxFE;

        if useEliteCenter
            Guide = computeEliteGuide(X, Objective_values, Destination_position, tau);
        else
            Guide = Destination_position;
        end

        exponent = 1 / max(alpha, eps);
        r1 = 2 * (1 - tau)^exponent;

        X_old = X;

        X_mem = c0 .* X + c1 .* X_prev1 + c2 .* X_prev2;

        R2 = 2 * pi * rand(N, dim);
        R3 = 2 * rand(N, dim);
        R4 = rand(N, dim);

        GuideMat = repmat(Guide, N, 1);
        Distance = abs(R3 .* GuideMat - X);

        Trig = sin(R2);
        cosMask = R4 >= 0.5;
        Trig(cosMask) = cos(R2(cosMask));

        X_candidate = X_mem + r1 .* Trig .* Distance;
        X_candidate = min(max(X_candidate, lbVec), ubVec);

        for i = 1:N
            if FE >= MaxFE
                break;
            end

            candidate_fit = fobj(X_candidate(i, :));
            FE = FE + 1;

            if ~useGreedy || candidate_fit <= Objective_values(i)
                X(i, :) = X_candidate(i, :);
                Objective_values(i) = candidate_fit;

                if candidate_fit < Destination_fitness
                    Destination_fitness = candidate_fit;
                    Destination_position = X_candidate(i, :);
                end
            end

            Convergence_curve(FE) = Destination_fitness;
        end

        X_prev2 = X_prev1;
        X_prev1 = X_old;
    end

    if FE < MaxFE
        Convergence_curve(FE+1:end) = Destination_fitness;
    end
end


function Guide = computeEliteGuide(X, fitness, gbest, tau)
% Dynamic elite-center guide.
% Early stage: multiple elite solutions provide diversity.
% Late stage : the guide gradually collapses to gbest for exploitation.

    N = size(X, 1);
    [~, order] = sort(fitness, 'ascend');

    L0 = max(1, round(N / 2));
    L = max(1, round(L0 - (L0 - 1) * tau));
    eliteIdx = order(1:L);

    worstFit = max(fitness);
    w = worstFit - fitness(eliteIdx);

    if sum(w) <= eps || any(~isfinite(w))
        eliteCenter = mean(X(eliteIdx, :), 1);
    else
        w = w ./ sum(w);
        eliteCenter = w' * X(eliteIdx, :);
    end

    blendToBest = tau^1.5;
    Guide = (1 - blendToBest) .* eliteCenter + blendToBest .* gbest;
end


function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);

    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        X = zeros(SearchAgents_no, dim);
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end


function b = expandBound(v, dim)
    if isscalar(v)
        b = repmat(v, 1, dim);
    else
        b = reshape(v, 1, []);
        if numel(b) ~= dim
            error('The bound vector length must be equal to dim.');
        end
    end
end


function val = getOption(opts, fieldName, defaultVal)
    if isstruct(opts) && isfield(opts, fieldName) && ~isempty(opts.(fieldName))
        val = opts.(fieldName);
    else
        val = defaultVal;
    end
end
