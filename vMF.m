clear;
close all;
clc;
%% SOFA
sofa = SOFAload('pp1_HRIRs_measured.sofa');

% ============ SOFA ================
% ListenerPosition [0,0,0]
% ListenerView [1,0,0], ListenerUp [0,0,1] ->
% +x = front , +z = up , +y = left, −y = right
% ReceiverPosition:
%     1.  0    0.7400         0 → LEFT
%     2.  0   -0.7400         0 → RIGHT
% ===================================
%% ===== HRIR (time domain) =====
IR = sofa.Data.IR;
fs = sofa.Data.SamplingRate;

% ======== ALL DIRECTIONS 440*256 =========
hL_all = squeeze(IR(:,1,:)); 
hR_all = squeeze(IR(:,2,:));

%% ========================================= SKIP
m = 1; % position IR 440*2*256 [direction × receiver/ear × time samples]

hL = squeeze(IR(m,1,:));
hR = squeeze(IR(m,2,:));

% --- plot ---

t = (0:length(hL)-1)/fs;

figure;
plot(t,hL); hold on;
plot(t,hR);
xlabel('Time (s)');
ylabel('Amplitude');
legend('Left','Right');
title('HRIR (one position)');
grid on;

%% ===== HRTF (freq domain) ======

Nfft = 4096;
Nhalf = Nfft/2 + 1;
% Frequency axis for positive FFT bins
f = (0:Nhalf-1) * (fs/Nfft);


% ======== ALL DIRECTIONS 440*256 =========
HL_all = fft(hL_all, Nfft, 2);
HR_all = fft(hR_all, Nfft, 2);

HL_all = HL_all(:,1:Nhalf);
HR_all = HR_all(:,1:Nhalf);

%% ========================================= SKIP

% --- one direction for plot ---

HL = fft(hL, Nfft); 
HR = fft(hR, Nfft);

HL = HL(1:Nhalf);
HR = HR(1:Nhalf);

% -- plot --

magL = 20*log10(abs(HL) + 1e-12);
magR = 20*log10(abs(HR) + 1e-12);

figure;
semilogx(f, magL); hold on;
semilogx(f, magR);
xlim([20 20000]);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
legend('Left','Right');
title('HRTF magnitude (one position)');
grid on;
%% ==== SH ====
L = 16; %order

azimuth   = sofa.SourcePosition(:,1);
elevation = sofa.SourcePosition(:,2);

azimuth   = deg2rad(azimuth);
elevation = deg2rad(elevation);

Y = real_sh_matrix(L, azimuth, elevation); %SH basis
CoeffLeft  = Y \ HL_all;
CoeffRight = Y \ HR_all;

% reconstraction
HL_rec = Y * CoeffLeft;

%% --- SH test error SKIP

err = norm(HL_all - HL_rec, 'fro') / norm(HL_all, 'fro');
err_mag = norm(abs(HL_all) - abs(HL_rec), 'fro') / norm(abs(HL_all), 'fro');

disp(err_mag);
disp(err);

% --- SH test error different L
% the number of SH coefficients for order L is (L+1)^2
% L=20 -> 441 coefficients and we have 440 directions from HUTUBS
orders = [4 8 12 16 20];

figure; hold on;

for Lt = orders

    Ytest = real_sh_matrix(Lt, azimuth, elevation);
    Ctest = Ytest \ HL_all;
    Hrec  = Ytest * Ctest;

    err_f = vecnorm(HL_all-Hrec,2,1) ./ vecnorm(HL_all,2,1);

    plot(f, err_f, 'DisplayName', sprintf('Lt=%d',Lt));
end

xlim([20 20000]);
ylim([0 0.6]);
xlabel('Frequency (Hz)');
ylabel('Relative error');
legend;
grid on;
title('SH reconstruction error vs frequency');


%% ===== SH → spatial map on Fibonacci sphere. 

fib_s_size = 1000;

[dirs, az, el] = fibonacci_sphere(fib_s_size);
Y_grid = real_sh_matrix(L, az, el);

k = 100; % freq control, freq bin

H_map_L = Y_grid * CoeffLeft(:,k);
H_map_R = Y_grid * CoeffRight(:,k);
%% POINT RENDERING SKIP
figure;
scatter3(dirs(:,1), dirs(:,2), dirs(:,3), 25, abs(H_map_L), 'filled');
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title(sprintf('|HRTF| Left ear, %.0f Hz', f(k)));
colormap turbo;
colorbar;
grid on;
view(3);
%% triangulation of the spherical surface using the Fibonacci points
% Fibonacci points -> convhull -> triangular mesh

tri = convhull(dirs(:,1), dirs(:,2), dirs(:,3)); %need it one again below 
figure;
trisurf(tri, dirs(:,1), dirs(:,2), dirs(:,3), abs(H_map_L), 'EdgeColor','none'); % triangular surface
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title(sprintf('|HRTF| Left ear, %.0f Hz', f(k)));
shading interp; %visual interpolation for rendering
colormap turbo;
colorbar;
view(3);



%% vMF func from Main_confronto.m


M_L = abs(H_map_L);

%w = M_L; %for manual K
%w = w / (sum(w) + eps);

K_list = 1:20;

rmseThresholdDB = 2.0;


%K = 3;

params.emMaxIter = 200; % max iterations number
params.emTol = 1e-7; % error, current likelihood with previous

params.maxKappa = 120; %max min concentration
params.minKappa = 1e-4;

params.minComponentWeight = 1e-5; %min Pi component

params.initKappa = 12;
params.initMinSepDeg = 18;% initial centers ≥ 18°
params.reinitMinSepDeg = 12;

result = selectVMF_K(dirs,M_L, K_list, params, rmseThresholdDB);
K = result.Kbest;
model = result.modelBest;


% dirs - 1000 x 3 [x y z] , w - 1000 x 1 - points' weights , K - number of vMF
% components, 
% model = fitWeightedVMFMixtureOnGrid(dirs, w, K, params); % for setting K manually

% ======== print results 
fprintf('\n===== vMF MIXTURE RESULT =====\n');
fprintf('K = %d\n\n', K);

for kj = 1:K

    mu = model.mu(kj,:);
    kap = model.kappa(kj);
    pi_k = model.pi(kj);

    az = atan2(mu(2), mu(1));
    el = asin(mu(3));

    theta50 = acos(max(-1,min(1, 1 + log(0.5)/kap)));

    fprintf('Component %d\n', kj);
    fprintf('  pi      = %.6f\n', pi_k);
    fprintf('  kappa   = %.6f\n', kap);

    fprintf('  mu      = [%.6f  %.6f  %.6f]\n', mu(1), mu(2), mu(3));

    fprintf('  azimuth = %.2f deg\n', rad2deg(az));
    fprintf('  elev.   = %.2f deg\n', rad2deg(el));

    fprintf('  theta50 = %.2f deg\n\n', rad2deg(theta50));
end

fprintf('Sum(pi) = %.6f\n', sum(model.pi));

% ========= print results end


%  ======= plot
% point rendering
% figure; 
% scatter3(dirs(:,1), dirs(:,2), dirs(:,3), 20, M_L, 'filled');
% axis equal;
% hold on;
% colormap turbo;
% colorbar;
% view(3);


tri = convhull(dirs(:,1), dirs(:,2), dirs(:,3));
figure;
trisurf(tri, dirs(:,1), dirs(:,2), dirs(:,3), M_L,'EdgeColor','none');
shading interp;
axis equal;
hold on;
colormap turbo;
colorbar;
view(3);

for ki = 1:K

    mu = model.mu(ki,:);
    kap = model.kappa(ki);

    % center
    plot3(mu(1), mu(2), mu(3), 'ro', ...
        'MarkerFaceColor','r', 'MarkerSize',8);

    % component number
    text(1.08*mu(1), 1.08*mu(2), 1.08*mu(3), ...
    sprintf('%d', ki), ...
    'FontSize', 14, ...
    'FontWeight', 'bold', ...
    'Color', 'k');

    % 50% concentration angle
    theta = acos(max(-1,min(1, 1 + log(0.5)/kap)));

    % basis perpendicular to mu
    a = [0 0 1];
    if abs(dot(mu,a)) > 0.9
        a = [0 1 0];
    end

    u = cross(mu,a);
    u = u/norm(u);

    v = cross(mu,u);
    v = v/norm(v);

    % circle
    t = linspace(0,2*pi,200);

    C = cos(theta)*mu + ...
        sin(theta)*(cos(t(:))*u + sin(t(:))*v);

    % slightly outside sphere so it is visible
    C = 1.03*C;

    plot3(C(:,1), C(:,2), C(:,3), 'k-', 'LineWidth',2);

end

hold off;
%%
pRec = reconstructVMFMixturePDF(dirs, model);
Mrec = result.scale * pRec;


% Point rendering
% figure; 
% scatter3(dirs(:,1), dirs(:,2), dirs(:,3), ...
%     25, pRec, 'filled');
% 
% axis equal;
% colormap turbo;
% colorbar;
% xlabel('x');
% ylabel('y');
% zlabel('z');
% grid on;
% view(3);
% title('Reconstructed spatial map from vMF mixture');



figure;

trisurf(tri, dirs(:,1), dirs(:,2), dirs(:,3), Mrec, 'EdgeColor','none');
shading interp;
axis equal;
colormap turbo;
colorbar;

xlabel('x');
ylabel('y');
zlabel('z');

view(3);
title(sprintf('reconstructed |HRTF| Left ear, %.0f Hz', f(k)));

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

    T = table(score(:), az(:), el(:), ...
        (1:numel(score)).', ...
        'VariableNames', {'score','az','el','idx'});

    T = sortrows(T, {'score','az','el'}, {'descend','ascend','ascend'});
    ord = T.idx;

    model.pi = model.pi(ord);
    model.mu = model.mu(ord,:);
    model.kappa = model.kappa(ord);
end

%test
function result = selectVMF_K(X, M, K_list, params, rmseThresholdDB)

% X               : N x 3 directions on sphere
% M               : N x 1 original HRTF magnitude
% K_list          : tested K values, e.g. 1:10
% params          : parameters for fitWeightedVMFMixtureOnGrid
% rmseThresholdDB : target reconstruction error in dB
%
% OUTPUT:
% result.Kbest
% result.models
% result.rmseDB
% result.relError
% result.pRec
% result.Mrec

% MML->vMF density complexity​
% RMSE(dB) -> HRTF reconstruction accuracy

    M = max(M(:), 0);

    if sum(M) <= eps
        error('Input magnitude M is zero.');
    end

    % Save absolute scale
    scale = sum(M);

    % Normalized spatial distribution
    pTrue = M / scale;

    nK = numel(K_list);

    rmseDB   = zeros(nK,1);
    relError = zeros(nK,1);

    mmlScore = zeros(nK,1);
    bicScore = zeros(nK,1);
    Neff = 1 / sum(pTrue.^2);

    models = cell(nK,1);
    pRecAll = cell(nK,1);
    MrecAll = cell(nK,1);

    % Small floor for dB calculation
    dbFloor = 1e-8;

    for i = 1:nK

        K = K_list(i);

        fprintf('Fitting K = %d ...\n', K);

        % Fit vMF mixture
        model = fitWeightedVMFMixtureOnGrid( ...
            X, pTrue, K, params);

        % Reconstruct normalized distribution
        pRec = reconstructVMFMixturePDF(X, model);

        % Restore original magnitude scale
        Mrec = scale * pRec;

        % Relative linear-domain error
        relError(i) = norm(M - Mrec) / norm(M);

        % dB-domain error
        MdB    = 20*log10(M    + dbFloor);
        MrecDB = 20*log10(Mrec + dbFloor);

        rmseDB(i) = sqrt(mean((MdB - MrecDB).^2));

        % BIC
        residualDB = MdB - MrecDB;
        RSS = sum(residualDB.^2);   
        N = numel(M);      
        pK = 4*K - 1;  
        bicScore(i) = N * log(RSS / N) + pK * log(N);

        % MML-like score

        logL = model.finalObjective * Neff;
        penalty = 0.5 * pK * log(Neff);
        mmlScore(i) = -logL + penalty;

        % Save results
        models{i} = model;
        pRecAll{i} = pRec;
        MrecAll{i} = Mrec;

        fprintf(['K = %d | RMSE = %.3f dB | ' ...
         'MML = %.3f | BIC = %.3f\n'], ...
         K, rmseDB(i), mmlScore(i), bicScore(i));

    end

    % ======== Select smallest K satisfying threshold

%     idx = find(rmseDB <= rmseThresholdDB, 1, 'first');
% 
%     if isempty(idx)
% 
%         % No K reached threshold:
%         % choose K with lowest RMSE
%         [~, idx] = min(rmseDB);
% 
%         warning(['No K reached %.2f dB threshold. ', ...
%                  'Using best tested K = %d instead.'], ...
%                  rmseThresholdDB, K_list(idx));
%     end
% 
%     Kbest = K_list(idx);

    % ===== Select K by minimum BIC =====

    [~, idx] = min(bicScore);

    Kbest = K_list(idx);


    % Output structure

    result.Kbest = Kbest;

    result.K_list = K_list(:);

    result.rmseDB = rmseDB;
    result.relError = relError;

    result.models = models;

    result.modelBest = models{idx};

    result.pRec = pRecAll;
    result.Mrec = MrecAll;

    result.pRecBest = pRecAll{idx};
    result.MrecBest = MrecAll{idx};

    result.scale = scale;
    result.thresholdDB = rmseThresholdDB;

    result.mmlScore = mmlScore;
    result.Neff = Neff;

    result.bicScore = bicScore;

    % ===== Visualization =====

    figure;
    tiledlayout(1,2);

    % RMSE
    nexttile;

    plot(K_list, rmseDB, '-o', ...
        'LineWidth', 1.5, ...
        'MarkerSize', 7);

    hold on;

    plot(Kbest, rmseDB(idx), 'o', ...
        'MarkerSize', 12, ...
        'LineWidth', 2);

    xlabel('Number of vMF components K');
    ylabel('RMSE [dB]');
    title('Reconstruction error');
    grid on;

    % BIC
    nexttile;

    plot(K_list, bicScore, '-o', ...
        'LineWidth', 1.5, ...
        'MarkerSize', 7);

    hold on;

    plot(Kbest, bicScore(idx), 'o', ...
        'MarkerSize', 12, ...
        'LineWidth', 2);

    xlabel('Number of vMF components K');
    ylabel('BIC');
    title(sprintf('BIC model selection: K_{best} = %d', Kbest));
    grid on;

end
