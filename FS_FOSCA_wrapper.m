function [bestFitness, bestSubset, curve] = FS_FOSCA2_wrapper( ...
    X_train, Y_train, PopSize, MaxFE, kFold, kKNN)
% FS_FOSCA2_wrapper
% Feature-selection wrapper for FOSCA2 under real FE counting.
%
% Fitness protocol UNIFIED with FS_SCA_wrapper / FS_MPA_wrapper / etc.:
%   - cvpartition is rebuilt INSIDE fitnessFunction at every fobj call.
%   - This matches the noisy-fitness regime used by all baseline algorithms
%     (SCA, MPA, HHO, PSO, SSA, GA, GWO, WOA, DE, JADE, LSHADE).
%   - 30 independent runs absorb the CV noise; do NOT pre-fix CV here, that
%     would make FOSCA2 see a denoised objective while baselines see a noisy
%     one — a fairness violation reviewers will catch.
%
% Fitness:
%   0.99 * K-fold CV MCE + 0.01 * selected-feature ratio.

    dim = size(X_train, 2);
    lb  = 0;
    ub  = 1;

    fobj = @(x) wrapperFitness(x, X_train, Y_train, kFold, kKNN);

    opts.alpha = 0.82;
    opts.useGreedy = true;
    opts.useEliteCenter = true;

    [bestFitness, bestPos, curve] = FOSCA2_FE(PopSize, MaxFE, lb, ub, dim, fobj, opts);

    bestSubset = continuousToBinary(bestPos);
end


function subset = continuousToBinary(x)
    subset = x > 0.5;

    if sum(subset) == 0
        [~, idx] = max(x);
        subset(idx) = 1;
    end

    subset = double(subset);
end


function fit = wrapperFitness(x, X_train, Y_train, kFold, kKNN)
    subset = continuousToBinary(x);
    fit = fitnessFunction(subset, X_train, Y_train, kFold, kKNN);
end


function fit = fitnessFunction(subset, X_train, Y_train, kFold, kKNN)
% Identical noise regime to FS_SCA_wrapper: a new cvpartition is created on
% every call. This is intentional, not a bug.

    if sum(subset) == 0
        fit = 1e6;
        return;
    end

    Xs = X_train(:, logical(subset));
    cv = cvpartition(Y_train, 'KFold', kFold);

    mce_list = zeros(kFold, 1);

    for f = 1:kFold
        idxTr = training(cv, f);
        idxVa = test(cv, f);

        Xtr = Xs(idxTr, :);
        Ytr = Y_train(idxTr);

        Xva = Xs(idxVa, :);
        Yva = Y_train(idxVa);

        mdl = fitcknn(Xtr, Ytr, ...
            'NumNeighbors', kKNN, ...
            'Standardize', true);

        Ypred = predict(mdl, Xva);
        mce_list(f) = mean(Ypred ~= Yva);
    end

    MCE_cv    = mean(mce_list);
    featRatio = sum(subset) / size(X_train, 2);

    fit = 0.99 * MCE_cv + 0.01 * featRatio;
end
