clear;
close all;
clc;

%% ===== DATASET FILES =====
dataDir = '/Users/one/Study/HRTF modeling /Matlab and datasets/HUTUBS/HRIRs';

files = dir(fullfile(dataDir, 'pp*_HRIRs_measured.sofa'));

ids = arrayfun(@(x) sscanf(x.name, 'pp%d_'), files);
[ids, idx] = sort(ids);
files = files(idx);

% Keep last nTest subjects for testing
nTest = 92;
files = files(1:end-nTest);
ids = ids(1:end-nTest);

for i = 1:length(files)
    sofa = SOFAload(fullfile(dataDir, files(i).name));
    IR = sofa.Data.IR;

    hL(i,:,:) = squeeze(IR(:,1,:));
    hR(i,:,:) = squeeze(IR(:,2,:));
end


%% ===== HRTF (freq domain) =====

Nfft = 4096;
Nhalf = Nfft/2 + 1;

HL = fft(hL, Nfft, 3);
HR = fft(hR, Nfft, 3);

HL = HL(:,:,1:Nhalf);
HR = HR(:,:,1:Nhalf);

fs = sofa.Data.SamplingRate;

f = (0:Nhalf-1) * (fs/Nfft);


%% ===== SH =====

L = 16;

azimuth   = deg2rad(sofa.SourcePosition(:,1));
elevation = deg2rad(sofa.SourcePosition(:,2));

Y = real_sh_matrix(L, azimuth, elevation);

nSubjects = size(HL,1);
nCoeff = (L+1)^2;
nFreq = size(HL,3);

CoeffLeft  = zeros(nSubjects, nCoeff, nFreq);
CoeffRight = zeros(nSubjects, nCoeff, nFreq);

for i = 1:nSubjects

    HL_i = squeeze(HL(i,:,:));   % 440 x 2049
    HR_i = squeeze(HR(i,:,:));

    CoeffLeft(i,:,:)  = Y \ HL_i;
    CoeffRight(i,:,:) = Y \ HR_i;

end


%% ===== SH → spatial maps on Fibonacci sphere =====

fib_s_size = 1000;

[dirs, az, el] = fibonacci_sphere(fib_s_size);
Y_grid = real_sh_matrix(L, az, el);

% Representative frequencies [Hz]
freqTarget = logspace(log10(1000), log10(16000), 3);

% Find nearest FFT bins
freqBins = zeros(size(freqTarget));

for j = 1:length(freqTarget)
    [~, freqBins(j)] = min(abs(f - freqTarget(j)));
end

% Remove duplicate bins
freqBins = unique(freqBins);

% Actual frequencies corresponding to selected FFT bins
freqSelected = f(freqBins);

nSubjects = size(CoeffLeft, 1);
nFreqSelected = length(freqBins);

%% ===== Loop through subjects and selected frequencies =====

% ===== vMF settings =====

K_list = 1:20;

params.emMaxIter = 200;
params.emTol = 1e-7;
params.maxKappa = 120;
params.minKappa = 1e-4;
params.minComponentWeight = 1e-5;
params.initKappa = 12;
params.initMinSepDeg = 18;
params.reinitMinSepDeg = 12;

% ===== Evaluation setup =====

figDir = 'figures';

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

datasetDir = 'datasets';

if ~exist(datasetDir, 'dir')
    mkdir(datasetDir);
end


nRows = nSubjects * nFreqSelected * 2;  % 2 ears

stats = table( ...
    'Size', [nRows 10], ...
    'VariableTypes', { ...
        'double','string','double','double', ...
        'double','double','double','double', ...
        'double','double'}, ...
    'VariableNames', { ...
        'Subject','Ear','Frequency','K', ...
        'RMSE_dB','RelativeError','Correlation','JS', ...
        'BIC','MML'});

% RMSE_dB -> HRTF reconstruction accuracy	​
% JS -> normalized spatial-shape accuracy
% Correlation -> similarity of spatial pattern 
% BIC -> reconstruction accuracy vs model complexity
% MML -> vMF probability-density fit vs complexity

rowID = 0;


tri = convhull(dirs(:,1), dirs(:,2), dirs(:,3));

% ===== Store vMF models =====

models_L = cell(nSubjects, nFreqSelected);
models_R = cell(nSubjects, nFreqSelected);

scale_L = zeros(nSubjects, nFreqSelected);
scale_R = zeros(nSubjects, nFreqSelected);

% ====== NN =======
Kmax = max(K_list);

% Each row:
% Subject, Ear, Frequency, Scale, K,
% then Kmax * [pi, mu_x, mu_y, mu_z, kappa]

nNNRows = nSubjects * nFreqSelected * 2;
nNNCols = 5 + 5*Kmax;

nnData = zeros(nNNRows, nNNCols);
nnRow = 0;


for i = 1:nSubjects

    for j = 1:nFreqSelected

        k = freqBins(j);
        subjectID = ids(i);


        % SH coefficients for current subject and frequency
        C_L = squeeze(CoeffLeft(i,:,k)).';
        C_R = squeeze(CoeffRight(i,:,k)).';

        % Reconstruct spatial HRTF maps on Fibonacci sphere
        H_map_L = Y_grid * C_L;
        H_map_R = Y_grid * C_R;

        % Current frequency
        freqHz = f(k);

        % Magnitude spatial maps
        M_L = abs(H_map_L);
        M_R = abs(H_map_R);

        % Fit vMF mixtures + automatic K selection
        result_L = selectVMF_K(dirs, M_L, K_list, params);
        result_R = selectVMF_K(dirs, M_R, K_list, params);

        % Best models
        model_L = result_L.modelBest;
        model_R = result_R.modelBest;

        % Canonical component ordering FOR NN
        model_L = canonicalizeVMFModel(model_L);
        model_R = canonicalizeVMFModel(model_R);

        % ===== NN dataset =====
        
        for e = 1:2
        
            if e == 1
                model = model_L;
                scale = result_L.scale;
                earID = 0;      % left
            else
                model = model_R;
                scale = result_R.scale;
                earID = 1;      % right
            end
        
            nnRow = nnRow + 1;
        
            % Metadata / global parameters
            nnData(nnRow,1) = subjectID;
            nnData(nnRow,2) = earID;
            nnData(nnRow,3) = freqHz;
            nnData(nnRow,4) = scale;
            nnData(nnRow,5) = numel(model.pi);
        
            % vMF components
            for c = 1:numel(model.pi)
        
                base = 5 + (c-1)*5;
        
                nnData(nnRow,base+1) = model.pi(c);
                nnData(nnRow,base+2) = model.mu(c,1);
                nnData(nnRow,base+3) = model.mu(c,2);
                nnData(nnRow,base+4) = model.mu(c,3);
                nnData(nnRow,base+5) = model.kappa(c);
        
            end
        
        end
        
        % Store models
        models_L{i,j} = model_L;
        models_R{i,j} = model_R;
        
        % Store magnitude scale
        scale_L(i,j) = result_L.scale;
        scale_R(i,j) = result_R.scale;

        % Reconstructed magnitude maps
        Mrec_L = result_L.MrecBest;
        Mrec_R = result_R.MrecBest;

        % ===== Statistics =====
        
        idxBest_L = find(result_L.K_list == result_L.Kbest);
        idxBest_R = find(result_R.K_list == result_R.Kbest);
        
        % dB maps
        MdB_L    = 20*log10(M_L    + 1e-8);
        MrecDB_L = 20*log10(Mrec_L + 1e-8);
        
        MdB_R    = 20*log10(M_R    + 1e-8);
        MrecDB_R = 20*log10(Mrec_R + 1e-8);
        
        % Spatial correlation
        corr_L = corr(MdB_L, MrecDB_L);
        corr_R = corr(MdB_R, MrecDB_R);
        
        % ----- Left ear -----
        rowID = rowID + 1;
        
        stats.Subject(rowID)       = subjectID;
        stats.Ear(rowID)           = "L";
        stats.Frequency(rowID)     = freqHz;
        stats.K(rowID)             = result_L.Kbest;
        stats.RMSE_dB(rowID)       = result_L.rmseDB(idxBest_L);
        stats.RelativeError(rowID) = result_L.relError(idxBest_L);
        stats.Correlation(rowID)   = corr_L;
        stats.JS(rowID)            = result_L.jsDiv(idxBest_L);
        stats.BIC(rowID)           = result_L.bicScore(idxBest_L);
        stats.MML(rowID)           = result_L.mmlScore(idxBest_L);
        
        % ----- Right ear -----
        rowID = rowID + 1;
        
        stats.Subject(rowID)       = subjectID;
        stats.Ear(rowID)           = "R";
        stats.Frequency(rowID)     = freqHz;
        stats.K(rowID)             = result_R.Kbest;
        stats.RMSE_dB(rowID)       = result_R.rmseDB(idxBest_R);
        stats.RelativeError(rowID) = result_R.relError(idxBest_R);
        stats.Correlation(rowID)   = corr_R;
        stats.JS(rowID)            = result_R.jsDiv(idxBest_R);
        stats.BIC(rowID)           = result_R.bicScore(idxBest_R);
        stats.MML(rowID)           = result_R.mmlScore(idxBest_R);

        % ===== Save figures =====

        ears = {'L','R'};
        M_orig = {M_L, M_R};
        M_rec  = {Mrec_L, Mrec_R};
        results = {result_L, result_R};
        
        for e = 1:2
        
            M     = M_orig{e};
            Mrec  = M_rec{e};
            result = results{e};
            ear   = ears{e};
        
            errorDB = 20*log10(Mrec + 1e-8) - ...
                      20*log10(M + 1e-8);
        
            fig = figure('Visible','off', ...
                 'Position',[100 100 1500 520]);

            set(fig, 'CreateFcn', 'set(gcbo,''Visible'',''on'')');

            t = tiledlayout(fig, 1, 3, ...
                'TileSpacing','compact', ...
                'Padding','loose');
        
        
            % Original
            nexttile;
            trisurf(tri, dirs(:,1), dirs(:,2), dirs(:,3), ...
                M, 'EdgeColor','none');
            shading interp;
            axis equal off;
            view(3);
            colorbar;
            title('Original');
        
            % Reconstruction
            nexttile;
            trisurf(tri, dirs(:,1), dirs(:,2), dirs(:,3), ...
                Mrec, 'EdgeColor','none');
            shading interp;
            axis equal off;
            view(3);
            colorbar;
            title(sprintf('vMF reconstruction, K=%d', result.Kbest));
        
            % Error
            nexttile;
            trisurf(tri, dirs(:,1), dirs(:,2), dirs(:,3), ...
                errorDB, 'EdgeColor','none');
            shading interp;
            axis equal off;
            view(3);
            colorbar;
            title('Error [dB]');
        

            sgtitle(t, sprintf('Subject %d | %s ear | %.0f Hz', subjectID, ear, freqHz), 'FontWeight','bold');
        
            fileName = sprintf('S%03d_%s_%05dHz', ...
                subjectID, ear, round(freqHz));

        
            savefig(fig, fullfile(figDir, [fileName '.fig']));

            exportgraphics(fig, ...
                fullfile(figDir, [fileName '.png']), ...
                'Resolution',150);

            close(fig);
        
        end

    end


end
    

% ===== Save dataset =====

save(fullfile(datasetDir, 'vmf_dataset.mat'), ...
    'models_L', 'models_R', ...
    'scale_L', 'scale_R', ...
    'ids', 'freqSelected');


% ===== Save statistics =====

writetable(stats, fullfile(datasetDir, 'vmf_statistics.csv'));
save(fullfile(datasetDir, 'vmf_statistics.mat'), 'stats');

        
disp(stats);


% ===== Global statistics =====

fprintf('\n===== GLOBAL vMF STATISTICS =====\n');

fprintf('Number of HRTFs      : %d\n', height(stats));

fprintf('Mean RMSE            : %.3f dB\n', mean(stats.RMSE_dB));
fprintf('Median RMSE          : %.3f dB\n', median(stats.RMSE_dB));
fprintf('95th percentile RMSE : %.3f dB\n', prctile(stats.RMSE_dB,95));

fprintf('Mean JS              : %.5f\n', mean(stats.JS));
fprintf('Mean correlation     : %.4f\n', mean(stats.Correlation));

fprintf('Mean selected K      : %.2f\n', mean(stats.K));
fprintf('Median selected K    : %.1f\n', median(stats.K));
fprintf('Min selected K       : %d\n', min(stats.K));
fprintf('Max selected K       : %d\n', max(stats.K));

% ===== Statistics by frequency =====

freqStats = groupsummary( ...
    stats, ...
    'Frequency', ...
    'mean', ...
    {'RMSE_dB','JS','Correlation','K'});

disp(freqStats);

% ===== K distribution =====

Kstats = groupsummary(stats, 'K');

disp(Kstats);

% ==== Save for NN =====
varNames = {'Subject','Ear','Frequency','Scale','K'};

for c = 1:Kmax
    varNames{end+1} = sprintf('pi_%d', c);
    varNames{end+1} = sprintf('muX_%d', c);
    varNames{end+1} = sprintf('muY_%d', c);
    varNames{end+1} = sprintf('muZ_%d', c);
    varNames{end+1} = sprintf('kappa_%d', c);
end

nnTable = array2table(nnData, ...
    'VariableNames', varNames);

writetable(nnTable, ...
    fullfile(datasetDir, 'vmf_nn_dataset.csv'));

save(fullfile(datasetDir, 'vmf_nn_dataset.mat'), ...
    'nnData', 'varNames', 'Kmax');



%%

function model = fitWeightedVMFMixtureOnGrid(X, w, K, params)
    w = max(w(:), 0);
    if sum(w) <= eps
        error('Pesi nulli nel fitting vMF.');
    end
    w = w / sum(w);

    N = size(X,1);
    if size(X,2) ~= 3
        error('X deve essere [N x 3].');
    end

    [mu, pi_k, kappa] = initVMFPeaks(X, w, K, params);

    llPrev = -inf;
    llHist = nan(params.emMaxIter,1);

    for iter = 1:params.emMaxIter
        logProb = zeros(N, K);
        for k = 1:K
            dots = X * mu(k,:).';
            logProb(:,k) = log(max(pi_k(k), realmin)) + ...
                           logCvmf3(kappa(k)) + ...
                           kappa(k) * dots;
        end

        logDen = logsumexp(logProb, 2);
        resp = exp(logProb - logDen);
        wr = resp .* w;
        Nk = sum(wr, 1);

        ll = sum(w .* logDen);
        llHist(iter) = ll;

        pi_k = max(Nk(:), params.minComponentWeight);
        pi_k = pi_k / sum(pi_k);

        for k = 1:K
            if Nk(k) < params.minComponentWeight
                [mu(k,:), pi_k(k), kappa(k)] = reinitOneComponent(X, w, mu, idxExcluding(K,k), params);
                continue;
            end

            R = wr(:,k).' * X;
            Rn = norm(R);

            if Rn <= 1e-12
                [mu(k,:), pi_k(k), kappa(k)] = reinitOneComponent(X, w, mu, idxExcluding(K,k), params);
                continue;
            end

            mu(k,:) = R / Rn;
            Rbar = min(max(Rn / Nk(k), 0), 0.999999);
            kappa(k) = estimateKappaFromRbar(Rbar, params.maxKappa);
            kappa(k) = min(max(kappa(k), params.minKappa), params.maxKappa);
        end

        [~, ord] = sort(pi_k(:), 'descend');
        pi_k = pi_k(ord);
        mu = mu(ord,:);
        kappa = kappa(ord);

        if iter > 1
            if abs(ll - llPrev) < params.emTol * max(1, abs(llPrev))
                llHist = llHist(1:iter);
                break;
            end
        end
        llPrev = ll;
    end

    model = struct();
    model.pi = pi_k(:);
    model.mu = mu;
    model.kappa = kappa(:);
    model.numIter = numel(llHist);
    model.llHist = llHist(:);
    model.finalObjective = llHist(end);
end

function [mu, pi_k, kappa] = initVMFPeaks(X, w, K, params)
    N = size(X,1);
    [~, ord] = sort(w(:), 'descend');

    chosen = zeros(0,1);
    minSepRad = deg2rad(params.initMinSepDeg);

    for ii = 1:numel(ord)
        idx = ord(ii);
        xi = X(idx,:);

        if isempty(chosen)
            chosen(end+1,1) = idx; %#ok<AGROW>
        else
            ok = true;
            for jj = 1:numel(chosen)
                dp = max(-1, min(1, dot(xi, X(chosen(jj),:))));
                ang = acos(dp);
                if ang < minSepRad
                    ok = false;
                    break;
                end
            end
            if ok
                chosen(end+1,1) = idx; %#ok<AGROW>
            end
        end

        if numel(chosen) >= K
            break;
        end
    end

    while numel(chosen) < K
        if isempty(chosen)
            [~, idx] = max(w);
        else
            simMax = -inf(N,1);
            for c = 1:numel(chosen)
                simMax = max(simMax, X * X(chosen(c),:).');
            end
            score = w .* (1 - simMax);
            [~, idx] = max(score);
        end
        chosen(end+1,1) = idx; %#ok<AGROW>
    end

    mu = X(chosen(1:K), :);
    kappa = params.initKappa * ones(K,1);

    sim = X * mu.';
    [~, labels] = max(sim, [], 2);

    pi_k = zeros(K,1);
    for k = 1:K
        pi_k(k) = sum(w(labels == k));
        if pi_k(k) <= 0
            pi_k(k) = params.minComponentWeight;
        end
    end
    pi_k = pi_k / sum(pi_k);

    for k = 1:K
        wk = w .* (labels == k);
        Nk = sum(wk);
        if Nk <= params.minComponentWeight
            continue;
        end
        R = wk.' * X;
        Rn = norm(R);
        if Rn > 0
            mu(k,:) = R / Rn;
            Rbar = min(max(Rn / Nk, 0), 0.999999);
            kappa(k) = estimateKappaFromRbar(Rbar, params.maxKappa);
        end
    end
end

function [mu_k, pi_k, kappa_k] = reinitOneComponent(X, w, muAll, activeIdx, params)
    N = size(X,1);

    if isempty(activeIdx)
        [~, idx] = max(w);
    else
        simMax = -inf(N,1);
        for j = activeIdx
            simMax = max(simMax, X * muAll(j,:).');
        end
        score = w .* (1 - simMax);
        [~, idx] = max(score);
    end

    mu_k = X(idx,:);
    pi_k = max(params.minComponentWeight, 1/size(muAll,1));
    kappa_k = min(max(params.initKappa, params.minKappa), params.maxKappa);

    minSepRad = deg2rad(params.reinitMinSepDeg);
    for trial = 1:8
        if isempty(activeIdx)
            break;
        end

        good = true;
        for j = activeIdx
            dp = max(-1, min(1, dot(mu_k, muAll(j,:))));
            ang = acos(dp);
            if ang < minSepRad
                good = false;
                break;
            end
        end

        if good
            break;
        end

        v = randn(1,3);
        v = v / norm(v);
        mu_k = mu_k + 0.15 * v;
        mu_k = mu_k / norm(mu_k);
    end
end

function idx = idxExcluding(K, k0)
    idx = setdiff(1:K, k0);
end


function pRec = reconstructVMFMixturePDF(X, model)
    K = numel(model.pi);
    N = size(X,1);

    logMix = -inf(N, K);
    for k = 1:K
        dots = X * model.mu(k,:).';
        logMix(:,k) = log(max(model.pi(k), realmin)) + ...
                      logCvmf3(model.kappa(k)) + ...
                      model.kappa(k) * dots;
    end

    logDen = logsumexp(logMix, 2);
    pRec = exp(logDen);
    pRec = max(pRec, 0);

    if sum(pRec) <= eps
        pRec = ones(size(pRec)) / numel(pRec);
    else
        pRec = pRec / sum(pRec);
    end
end



function kappa = estimateKappaFromRbar(Rbar, maxKappa)
    Rbar = min(max(Rbar, 0), 0.999999);

    if Rbar < 1e-8
        kappa = 0;
        return;
    end

    if Rbar < 0.53
        kappa = 2*Rbar + Rbar^3 + 5*Rbar^5/6;
    elseif Rbar < 0.85
        kappa = -0.4 + 1.39*Rbar + 0.43/(1 - Rbar);
    else
        kappa = 1 / max(Rbar^3 - 4*Rbar^2 + 3*Rbar, 1e-8);
    end

    kappa = min(max(kappa, 1e-8), maxKappa);

    for it = 1:10
        A = A3ofKappa(kappa);
        dA = dA3ofKappa(kappa);
        step = (A - Rbar) / max(dA, 1e-12);
        kappaNew = kappa - step;
        kappaNew = min(max(kappaNew, 0), maxKappa);

        if abs(kappaNew - kappa) < 1e-8 * max(1, kappa)
            kappa = kappaNew;
            break;
        end
        kappa = kappaNew;
    end
end



function A = A3ofKappa(kappa)
    if kappa < 1e-6
        A = kappa / 3;
        return;
    end

    if kappa < 50
        A = coth(kappa) - 1 / kappa;
    else
        A = 1 - 1 / kappa;
    end
end

function dA = dA3ofKappa(kappa)
    if kappa < 1e-6
        dA = 1/3;
        return;
    end

    if kappa < 50
        dA = -csch(kappa)^2 + 1 / (kappa^2);
    else
        dA = 1 / (kappa^2);
    end
end

function val = logCvmf3(kappa)
    if kappa < 1e-8
        val = -log(4*pi);
        return;
    end

    if kappa < 50
        val = log(kappa) - log(4*pi) - log(sinh(kappa));
    else
        val = log(kappa) - log(4*pi) - (kappa - log(2));
    end
end

function val = logsumexp(A, dim)
    if nargin < 2
        dim = 1;
    end
    Amax = max(A, [], dim);
    val = Amax + log(sum(exp(A - Amax), dim));
end
%for NN
function model = canonicalizeVMFModel(model)
    % Canonical ordering to build a deterministic flattened feature vector.
    %
    % Primary key:
    %   score = pi * (1 + log1p(kappa))
    % Secondary keys:
    %   azimuth, elevation of the mean direction

    mu = model.mu;
    [az, el, ~] = cart2sph(mu(:,1), mu(:,2), mu(:,3));
    score = model.pi(:) .* (1 + log1p(model.kappa(:)));

    T = table(score(:), az(:), el(:), (1:numel(score)).', 'VariableNames', {'score','az','el','idx'});

    T = sortrows(T, {'score','az','el'}, {'descend','ascend','ascend'});
    ord = T.idx;

    model.pi = model.pi(ord);
    model.mu = model.mu(ord,:);
    model.kappa = model.kappa(ord);
end

%test
function result = selectVMF_K(X, M, K_list, params)

    M = max(M(:), 0);

    if sum(M) <= eps
        error('Input magnitude M is zero.');
    end

    % Absolute magnitude scale
    scale = sum(M);

    % Normalized spatial distribution
    pTrue = M / scale;

    nK = numel(K_list);
    N = numel(M);

    rmseDB   = zeros(nK,1);
    relError = zeros(nK,1);
    jsDiv    = zeros(nK,1);

    logLEff  = zeros(nK,1);
    mmlScore = zeros(nK,1);
    bicScore = zeros(nK,1);

    models  = cell(nK,1);
    pRecAll = cell(nK,1);
    MrecAll = cell(nK,1);

    dbFloor   = 1e-8;
    probFloor = 1e-12;

    % Effective number of spatial samples for MML
    Neff = 1 / sum(pTrue.^2);

    for i = 1:nK

        K = K_list(i);

        % ===== Fit vMF mixture =====
        model = fitWeightedVMFMixtureOnGrid(X, pTrue, K, params);

        % ===== Reconstruction =====
        pRec = reconstructVMFMixturePDF(X, model);
        Mrec = scale * pRec;

        % ===== Relative error =====
        relError(i) = norm(M - Mrec) / norm(M);

        % ===== RMSE in dB =====
        MdB    = 20*log10(M    + dbFloor);
        MrecDB = 20*log10(Mrec + dbFloor);

        residualDB = MdB - MrecDB;

        rmseDB(i) = sqrt(mean(residualDB.^2));

        % ===== Jensen-Shannon divergence =====
        p = max(pTrue, probFloor);
        q = max(pRec,  probFloor);

        p = p / sum(p);
        q = q / sum(q);

        m = 0.5 * (p + q);

        KL_pm = sum(p .* log(p ./ m));
        KL_qm = sum(q .* log(q ./ m));

        jsDiv(i) = 0.5 * KL_pm + 0.5 * KL_qm;

        % ===== Number of free parameters =====
        pK = 4*K - 1;

        % ===== MML-like score =====
        logLEff(i) = model.finalObjective * Neff;

        mmlScore(i) = -logLEff(i) + 0.5 * pK * log(Neff);

        % ===== BIC from dB reconstruction residuals =====
        RSS = sum(residualDB.^2);

        bicScore(i) = N * log(RSS / N) + pK * log(N);

        % ===== Store =====
        models{i}  = model;
        pRecAll{i} = pRec;
        MrecAll{i} = Mrec;

    end

    % ===== Select K by minimum BIC =====

    [~, idx] = min(bicScore);

    Kbest = K_list(idx);

    % ===== Output =====

    result.Kbest = Kbest;
    result.K_list = K_list(:);

    result.rmseDB   = rmseDB;
    result.relError = relError;
    result.jsDiv    = jsDiv;

    result.bicScore = bicScore;

    result.logLEff  = logLEff;
    result.mmlScore = mmlScore;
    result.Neff     = Neff;

    result.models = models;
    result.modelBest = models{idx};

    result.pRec = pRecAll;
    result.Mrec = MrecAll;

    result.pRecBest = pRecAll{idx};
    result.MrecBest = MrecAll{idx};

    result.scale = scale;

end