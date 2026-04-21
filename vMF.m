sofa = SOFAload('pp1_HRIRs_measured.sofa');
%% ===== HRIR =====
IR = sofa.Data.IR;
fs = sofa.Data.SamplingRate;

m = 1; % position IR 440*2*256

hL = squeeze(IR(m,1,:));
hR = squeeze(IR(m,2,:));

hL_all = squeeze(IR(:,1,:));
hR_all = squeeze(IR(:,2,:));

t = (0:length(hL)-1)/fs;

figure;
plot(t,hL); hold on;
plot(t,hR);
xlabel('Time (s)');
ylabel('Amplitude');
legend('Left','Right');
title('HRIR (one position)');
grid on;

%% ===== HRTF ======

Nfft = 4096;
HL = fft(hL, Nfft);
HR = fft(hR, Nfft);

HL_all = fft(hL_all,Nfft,2); 
HR_all = fft(hR_all,Nfft,2);

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
%% ==== SH ====
L = 16; %order

azimuth   = sofa.SourcePosition(:,1);
elevation = sofa.SourcePosition(:,2);

azimuth   = deg2rad(azimuth);
elevation = deg2rad(elevation);

Y = real_sh_matrix(L, azimuth, elevation);
CoeffLeft  = Y \ HL_all;
CoeffRight = Y \ HR_all;


%% ===== spatial map Fibonacci ====

fib_s_size = 1000;

[dirs, az, el] = fibonacci_sphere(fib_s_size);
Y_grid = real_sh_matrix(L, az, el);

k = 100;

H_map_L = Y_grid * CoeffLeft(:,k);
H_map_R = Y_grid * CoeffRight(:,k);

figure;
scatter3(dirs(:,1), dirs(:,2), dirs(:,3), 25, abs(H_map_L), 'filled');
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title(sprintf('|HRTF| Left ear (bin %d)', k));
colormap turbo;
colorbar;
grid on;
view(3);
%% vMF movmf
M_L_mo = abs(H_map_L);

w_raw_mo = M_L_mo;
w_mo = w_raw_mo / (sum(w_raw_mo) + eps);

dirs4 = [dirs, w_mo];
clust = movmf(dirs4, 3); 


%% vMF mixturefactory 

dir_3N = dirs';  % 3 x N
k = 100;   % frequency bin
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
scatter3(dir_3N(1,:), dir_3N(2,:), dir_3N(3,:), 25, w_raw, 'filled');
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