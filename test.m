sofa = SOFAload('pp1_HRIRs_measured.sofa');
%%
disp(sofa);
fieldnames(sofa);
%% HRIR

IR = sofa.Data.IR;
fs = sofa.Data.SamplingRate;

m = 1; % position IR 440*2*256

hL = squeeze(IR(m,1,:));
hR = squeeze(IR(m,2,:));

hL_all = squeeze(IR(:,1,:));
hR_all = squeeze(IR(:,2,:));


%g = max([max(abs(hL)), max(abs(hR))]);  %  gain
%hL = hL / g;
%hR = hR / g;

t = (0:length(hL)-1)/fs;

figure;
plot(t,hL); hold on;
plot(t,hR);
xlabel('Time (s)');
ylabel('Amplitude');
legend('Left','Right');
title('HRIR (one position)');
grid on;

%% HRTF

Nfft = 4096;
HL = fft(hL, Nfft);
HR = fft(hR, Nfft);

HL_all = fft(hL_all,Nfft,2);  % along time dimension
HR_all = fft(hR_all,Nfft,2);

%HL_all = HL_all(:, 1:Nfft/2+1); %half spectrum only positive
%HR_all = HR_all(:, 1:Nfft/2+1);

f = (0:Nfft-1)*(fs/Nfft);

magL = 20*log10(abs(HL)+1e-12);
magR = 20*log10(abs(HR)+1e-12);

figure;
semilogx(f, magL); hold on;
semilogx(f, magR);
xlim([20 20000]);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
legend('Left','Right');
title('HRTF magnitude (one position)');
grid on;

%%
%SH
L = 16;

azimuth   = sofa.SourcePosition(:,1);
elevation = sofa.SourcePosition(:,2);

azimuth   = deg2rad(azimuth);
elevation = deg2rad(elevation);

% Convert spherical angles to Cartesian points on unit sphere
x = cos(elevation) .* cos(azimuth);
y = cos(elevation) .* sin(azimuth);
z = sin(elevation);

%% plot directions


% Plot
figure;
scatter3(x, y, z, 40, elevation, 'filled');
axis equal;
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title('HUTUBS source directions on the unit sphere');
colorbar;
view(135,30);

%%

%SH
phi   = azimuth;
theta = pi/2 - elevation;

Y = real_sh_matrix(L, azimuth, elevation);
CoeffLeft  = Y \ HL_all;
CoeffRight = Y \ HR_all;

% check error reconstructed one with original
HL_rec = Y * CoeffLeft;
err = norm(HL_all - HL_rec, 'fro') / norm(HL_all, 'fro');
disp(err)
%plot the error for k freq bin
k = 100;
figure;
plot(abs(HL_all(:,k)), 'LineWidth', 1.5); hold on;
plot(abs(HL_rec(:,k)), '--', 'LineWidth', 1.5);
grid on;
legend('Original','Reconstructed');
title('Check SH reconstruction at one frequency bin');
%%
%spatial map 2D

% Dense angular grid
az = linspace(-pi, pi, 73);      % ~5° step
%az(end) = [];
el = linspace(-pi/2, pi/2, 36);  % ~5° step 2701 points 

[AZ, EL] = meshgrid(az, el); 

% Convert grid to vectors
az_grid = AZ(:);
el_grid = EL(:);

% Build SH matrix on dense grid
Y_grid = real_sh_matrix(L, az_grid, el_grid);



% one frequency bin
k = 100;

% Reconstruct spatial map
H_map_L = Y_grid * CoeffLeft(:,k);
H_map_R = Y_grid * CoeffRight(:,k);

% Reshape to 2D map
H_map_L = reshape(H_map_L, size(AZ));
H_map_R = reshape(H_map_R, size(AZ));


% Plot
figure;

%subplot(1,2,1) %LEFT ear
imagesc(rad2deg(az), rad2deg(el), abs(H_map_L));
axis xy;
xlabel('Azimuth (deg)');
ylabel('Elevation (deg)');
title(sprintf('|HRTF| Left ear (bin %d)', k));
colorbar;

%subplot(1,2,2) %RIGHT ear
%imagesc(rad2deg(az), rad2deg(el), abs(H_map_R));
%axis xy;
%xlabel('Azimuth (deg)');
%ylabel('Elevation (deg)');
%title(sprintf('|HRTF| Right ear (bin %d)', k));
%colorbar;

%all freq

%Nf = size(CoeffLeft,2);
%H_map_all_vec = Y_grid * CoeffLeft;
%H_map_all = reshape(H_map_all_vec, length(el), length(az), Nf);

%% spatial map Fibonacci sphere

fib_s_size = 1000;

[dirs, az, el] = fibonacci_sphere(fib_s_size);
Y_grid = real_sh_matrix(L, az, el);

k = 100;

H_map_L_fib = Y_grid * CoeffLeft(:,k);
H_map_R_fib = Y_grid * CoeffRight(:,k);

figure;
scatter3(dirs(:,1), dirs(:,2), dirs(:,3), 25, abs(H_map_L_fib), 'filled');
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title(sprintf('|HRTF| Left ear (bin %d)', k));
colormap turbo;
colorbar;
grid on;
view(3);


%% Background suppression (thresholding test)
M_L = abs(H_map_L);

% directions
x = cos(EL(:)) .* cos(AZ(:));
y = cos(EL(:)) .* sin(AZ(:));
z = sin(EL(:));

dir_N3 = [x y z];
dir_N3 = dir_N3 ./ vecnorm(dir_N3, 2, 2);
dir_3N = dir_N3.';


% Raw weights 
area_w = cos(EL(:)); %area correction for the sphere
w_raw = M_L(:) .* area_w;

% ========= Background suppression + square to boost peaks =======
 w_proc = max(w_raw - median(w_raw), 0);
 
 w_proc = w_proc.^2;
 mask = w_proc > 0;
 
 % data sent to vMF
 w = w_proc(mask);
 dir_3N = dir_3N(:, mask);
 w = w / (sum(w) + eps);


%   ===== SIMPLE THRESHOLD =====
%  T = 0.9;                 % threshold
%  mask = w_raw >= T;       % values >= T
%  
%  w = w_raw(mask);         % weights sent to vMF
%  dir_3N = dir_3N(:, mask);    % directions sent to vMF
%  w = w / (sum(w) + eps);


% ========== MixEst input

data.data   = dir_3N;   % 3 x N
data.weight = w.';      % 1 x N

% model
K = 3;
vmf = vmffactory(3);
D   = mixturefactory(vmf, K);

theta0 = D.randparam();


options = struct();
options.theta0  = theta0;
options.maxiter = 100;
options.miniter = 50;

[theta, D, info, options] = D.estimate(data, options);

% ======= extract directions and kappas
mu = zeros(K,3);
kappa = zeros(K,1);

for k = 1:K
    m = theta.D{k}.mu(:);
    mu(k,:) = (m / (norm(m) + eps)).';
    kappa(k) = theta.D{k}.kappa;
    disp(mu(k,:));
    disp(kappa(k));
end

% ===== Minimum Message Length (MML) =====


% effective number of samples (how many points affect the model)
Neff = (sum(w)^2) / (sum(w.^2) + eps);

% weighted log-likelihood of fitted model 
% (how well did your mu and k hit the real peaks of the map)
% scale log-likelihood by Neff because weights are normalized,
% so D.ll returns an average-like value. this restores the correct magnitude
% for model selection 
logL = D.ll(theta, data)* Neff;

% number of free parameters for 3D vMF mixture
pK = 4*K - 1;

% MML-like score
MML_score = -logL + 0.5 * pK * log(Neff);

disp(['logL = ', num2str(logL)])
disp(['Neff = ', num2str(Neff)])
disp(['MML-like score = ', num2str(MML_score)])



% ===== 3D sphere with map =====


X = reshape(dir_N3(:,1), size(AZ));
Y = reshape(dir_N3(:,2), size(AZ));
Z = reshape(dir_N3(:,3), size(AZ));

%C = M_L / (max(M_L(:)) + eps);

Cvec = nan(size(w_raw)); % points removed -> NaN
Cvec(mask) = w_raw(mask);   % show raw kept values
C = reshape(Cvec, size(M_L));

%Cvec = nan(size(w_proc));   % full size
%Cvec(mask) = w;             % normalized weights 
%C = reshape(Cvec, size(M_L));

figure;
surf(X, Y, Z, C, ...
    'EdgeColor', 'none', ...
    'FaceColor', 'interp', ...
    'FaceAlpha', 0.7);
axis equal;
hold on;
colormap turbo;
colorbar;
xlabel('x'); ylabel('y'); zlabel('z');
title('HRTF map + vMF directions+concentrations');
camlight headlight;
lighting gouraud;
grid on;

% centers + concentration circles
nCirc = 200;

for k = 1:K
    mu_k = mu(k,:).';
    kap  = kappa(k);

    % mark center
    plot3(mu_k(1), mu_k(2), mu_k(3), ...
        'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 8);

    % label
    text(1.08*mu_k(1), 1.08*mu_k(2), 1.08*mu_k(3), ...
        sprintf('\\kappa = %.2f', kap), ...
        'Color', 'w', 'FontWeight', 'bold', 'FontSize', 10);

    % choose angular radius from kappa
    % e^{-1} contour of vMF peak:
    % exp(kappa*(cos(gamma)-1)) = e^{-1}
    % => cos(gamma) = 1 - 1/kappa
    if kap > 1
        gamma = acos(max(-1, min(1, 1 - 1/kap)));
    else
        gamma = pi/3;   % fallback for very broad component
    end

    % build orthonormal basis around mu_k
    if abs(mu_k(3)) < 0.9
        a = [0;0;1];
    else
        a = [1;0;0];
    end

    u = cross(mu_k, a);
    u = u / (norm(u) + eps);

    v = cross(mu_k, u);
    v = v / (norm(v) + eps);

    % circle on sphere at angular radius gamma from mu_k
    t = linspace(0, 2*pi, nCirc);
    circ = cos(gamma)*mu_k + sin(gamma)*(u*cos(t) + v*sin(t));

    plot3(circ(1,:), circ(2,:), circ(3,:), 'w-', 'LineWidth', 2);
end

view(3);
hold off;
%% vMF with Fibonacci 
% ===== Fibonacci sphere sampling =====
Nfib = 1000;
[dir_N3, az, el] = fibonacci_sphere(Nfib);   % dir_N3: N x 3
dir_3N = dir_N3.';                           % 3 x N

% ===== SH reconstruction on Fibonacci points =====
Y_grid = real_sh_matrix(L, az, el);

k = 100;   % frequency bin

H_map_L = Y_grid * CoeffLeft(:,k);
H_map_R = Y_grid * CoeffRight(:,k);

M_L = abs(H_map_L);

w_raw = M_L;
w = w_raw / (sum(w_raw) + eps);

% ==== MixEst input =====
data.data   = dir_3N;   % 3 x N
data.weight = w.';      % 1 x N

K = 3;
vmf = vmffactory(3);
D   = mixturefactory(vmf, K);

theta0 = D.randparam();

options = struct();
options.theta0  = theta0;
options.maxiter = 100;
options.miniter = 50;

[theta, D, info, options] = D.estimate(data, options);

% === directions and kappas =====
mu = zeros(K,3);
kappa = zeros(K,1);

for kcomp = 1:K
    m = theta.D{kcomp}.mu(:);
    mu(kcomp,:) = (m / (norm(m) + eps)).';
    kappa(kcomp) = theta.D{kcomp}.kappa;

    disp(mu(kcomp,:));
    disp(kappa(kcomp));
end

% ===== Minimum Message Length (MML-like) =====
Neff = (sum(w)^2) / (sum(w.^2) + eps);

logL = D.ll(theta, data) * Neff;

pK = 4*K - 1;

MML_score = -logL + 0.5 * pK * log(Neff);

disp(['logL = ', num2str(logL)])
disp(['Neff = ', num2str(Neff)])
disp(['MML-like score = ', num2str(MML_score)])

% ===== visualization: input to vMF + vMF output =====
figure;
scatter3(dir_N3(:,1), dir_N3(:,2), dir_N3(:,3), 25, w_raw, 'filled');
axis equal;
hold on;
colormap turbo;
colorbar;
xlabel('x'); ylabel('y'); zlabel('z');
title('Fibonacci HRTF map + vMF directions + concentrations');
grid on;
view(3);

% ===== centers + concentration circles =====
nCirc = 200;

for kcomp = 1:K
    mu_k = mu(kcomp,:).';
    kap  = kappa(kcomp);

    % mark center
    plot3(mu_k(1), mu_k(2), mu_k(3), ...
        'wo', 'MarkerFaceColor', 'r', 'MarkerSize', 15);

    % label
    text(1.08*mu_k(1), 1.08*mu_k(2), 1.08*mu_k(3), ...
        sprintf('\\kappa = %.2f', kap), ...
        'Color', 'w', 'FontWeight', 'bold', 'FontSize', 8);

    % e^(-1) contour radius
    if kap > 1
        gamma = acos(max(-1, min(1, 1 - 1/kap)));
    else
        gamma = pi/3;
    end

    % local basis around mu_k
    if abs(mu_k(3)) < 0.9
        a = [0;0;1];
    else
        a = [1;0;0];
    end

    u = cross(mu_k, a);
    u = u / (norm(u) + eps);

    v = cross(mu_k, u);
    v = v / (norm(v) + eps);

    t = linspace(0, 2*pi, nCirc);
    circ = cos(gamma)*mu_k + sin(gamma)*(u*cos(t) + v*sin(t));

    plot3(circ(1,:), circ(2,:), circ(3,:), 'r-', 'LineWidth', 5);
end

hold off;


