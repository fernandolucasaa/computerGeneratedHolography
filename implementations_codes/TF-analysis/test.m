clear
clc
format compact

disp('Wigner Distribution Function');
disp('Freespace Propagation - Three-Dimensional Vector Field');

%% Initialize the Parameters

tic;

% System Parameter Definitions.............................................
sigma = [1, 2];             % Reduced RMS beam width
delta = [0.1, 0.2];         % Reduced RMS coherence width
theta0 = pi/4;              % Polarizer angle
kz = [0, 0.25, 0.5, 1.0];   % Reduced propagation distances

M = 100;                    % Spatial Resolution
N = 100;                    % Angular Resolution

disp(['----> 1a) Runtime: ', num2str(toc), ' s']);



% Coordinate Vector Definitions............................................
% Reduced transverse position
kLx = 20;
dkx = kLx / M;
kx = -kLx/2: dkx: kLx/2 - dkx;
ky = kx;

% Create plenoptic fields of the two coordinate vectors (kx, ky) where kx 
% varies along dimension 2 and ky along dimension 1
[kX, kY] = meshgrid(kx, ky);
kX = repelem(kX, N, N);
kY = repelem(kY, N, N);

% Reduced separation coordinate
kLxp = kLx;
dkxp = kLxp / N;
kxp = -kLxp/2: dkxp: kLxp/2 - dkxp;
kyp = kxp;

% Create plenoptic fields of the two coordinate vectors (kx', ky') where 
% kx' varies along dimension 2 and y' along dimension 1
[kXp, kYp] = meshgrid(kxp, kyp);
kXp = repmat(kXp, [M, M]);
kYp = repmat(kYp, [M, M]);

% Optical momentum - this is defined from the Fourier conjugate variable:
%       -i*2*pi*fx*x' = i*k*px*x' = i*2*pi*(px/lambda)*x'
%               fx = -px/lambda --> px = -fx*lambda
px = -1/(2*dkxp): 1/kLxp: 1/(2*dkxp) - 1/kLxp;
px = -(2*pi) .* px;
py = px;

% Clear some room in memory
clear kLx dkx kLxp kxp kyp kLz dkz

disp(['----> 1b) Runtime: ', num2str(toc), ' s']);



% Sampling Vectors.........................................................
kX1 = zeros(M*N, M*N, 2);
kX1(:, :, 1) = kX - kXp./2;
kX1(:, :, 2) = kY - kYp./2;

kX2 = zeros(M*N, M*N, 2);
kX2(:, :, 1) = kX + kXp./2;
kX2(:, :, 2) = kY + kYp./2;

% Clear some room in memory
clear kX kY kXp kYp

disp(['----> 1c) Runtime: ', num2str(toc), ' s']);



%% Cross-Spectral Density (CSD)

% Define the components of the CSD
norm3D = @(kx) kx(:, :, 1).^2 + kx(:, :, 2).^2;
S0  = @(kx, sigma) exp(-0.5 .* norm3D(kx ./ sigma));
mu0 = @(kx1, kx2, delta) exp(-0.5 .* norm3D((kx2 - kx1) ./ delta));

% Build the CSD
W = zeros(M*N, M*N, 4);
W(:,:,1)=cos(theta0)^2.*S0(kX1,sigma(1)).*S0(kX2,sigma(1)).*...
    mu0(kX1,kX2,delta(1));
W(:,:,4)=sin(theta0)^2.*S0(kX1,sigma(2)).*S0(kX2,sigma(2)).*...
    mu0(kX1,kX2,delta(2));

% Clear some room in memory
clear kX1 kX2 S0 mu0 norm3D c

disp(['----> 2) Runtime: ', num2str(toc), ' s']);



%% Wigner Distribution Function (WDF) Matrix

% Define the WDF from the Fourier transform of the CSD
wdf = @(csd) real(ifftshift(fft2(fftshift(csd)))) .* (dkxp / (2*pi))^2;

% Preallocate space for the WDF
B1 = zeros(size(W));

% Compute each of the four components of the WDF matrix
for c = 1: 4

    % Fourier transform each of the direction micro-images
    for i = 1: M
        qi = 1 + N * (i - 1);

        for j = 1: M
            qj = 1 + N * (j - 1);

            B1(qi:qi+N-1, qj:qj+N-1, c) = wdf(W(qi:qi+N-1, qj:qj+N-1, c));
        end
    end
end

% Clear some room in memory
clear wdf W dkxp i qi j qj c

disp(['----> 3) Runtime: ', num2str(toc), ' s']);



%% Propagation and the Coherency Matrix

% Invert each component of the WDF PMPF matrix
B1inv = zeros(size(B1));
for c = 1: 4
    B1inv(:, :, c) = invertPF(B1(:, :, c), M, N);
end

% Preallocate space for the coherency matrix
S = zeros(M, M, length(kz), 4);

% Propagate each component of the WDF matrix
for c = 1: 4

    disp(['Component: ', num2str(c)]);
    
    % Propagate each component to each observation screen
    for i = 1: length(kz)

        % Propagation --> shearing the phase space --> interpolation
        B2inv = propWDF(B1inv(:, :, c), px, py, kx, ky, kz(i));

        % Project the propagated WDF to compute the SD
        S(:, :, i, c) = coherency(B2inv, M, N, px, py);
    end
end

% Compute the Stokes parameters from the coherency matrix
[a0, a1, a2, a3] = stokesParams(S, 3);

% Compute the following from the Stokes parameters...
S0 = a0;                                    % ... spectral density
P = sqrt(a1.^2 + a2.^2 + a3.^2) ./ a0;      % ... degree of polarization

% Clear some room in memory
clear i c B1inv B2inv S

disp(['----> 4) Runtime: ', num2str(toc), ' s']);



%% Display the Numerical Results

% Spectral Density Profiles................................................
figure;
for i = 1: length(kz)
    subplot(2, 2, i);
    imagesc(kx, ky, S0(:, :, i));
    xlabel('kx');
    ylabel('ky');
    title(['kz = ', num2str(kz(i))]);
    axis xy;
    axis square;
    caxis([0, 1]);
    colorbar;
end

% DOP Profiles.............................................................
figure;
for i = 1: length(kz)
    subplot(2, 2, i);
    imagesc(kx, ky, P(:, :, i));
    xlabel('kx');
    ylabel('ky');
    title(['kz = ', num2str(kz(i))]);
    axis xy;
    axis square;
    caxis([0, 1]);
    colorbar;
end

% Clear some room in memory
clear i

disp(['Final Runtime: ', num2str(toc), ' s']);