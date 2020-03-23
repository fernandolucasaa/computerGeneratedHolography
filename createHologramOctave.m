close all
clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Generation numerique d'hologramme a partir de quelques sources ponctuelles en
% sommant leurs contributions (ondes spheriques). Les occultations ne sont pas 
% prises en compte pour l'instant !
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parametres initiaux %%

% longueur de l'onde 
lambda = 500e-9; % 500nm (vert)

% plan de l'hologramme
hologramHeight = 3e-3; % 3nm
hologramWidth = 3e-3; % 3nm
hologramZ = 0;

% nombre de pixels
Delta = 10e-6;
samplesX = hologramWidth / Delta; % 300 samples with sampling distance Delta
samplesY = hologramHeight / Delta; % 300 samples with sampling distance Delta

% coin de reference du plan de l'hologramme (inferieur a gauche)
cornerX = -hologramWidth / 2;
cornerY = -hologramHeight / 2;

% amplitude initiale
a = 1;

% nombre d'onde
k = 2*pi/lambda;

% points de la scene
%points = [0, 0, -0.2;
%          -hologramWidth/8, 0, -0.2];
points = [0, 0, -0.1];

%% [1] Calcul de l'onde objet (Nuage de points) %% 
objectWave = zeros(samplesY, samplesX);

% superposition de tous les ondes spheriques
for s = 1:rows(points)
  for column = 1:samplesX
    for row = 1:samplesY
      x = (column-1) * Delta + cornerX;
      y = (row-1) * Delta + cornerY;
      % distance oblique
      r = sqrt((x - points(s, 1))^2 + (y - points(s, 2))^2 + (hologramZ - points(s, 3))^2);
      objectWave(row,column) = (a / r)*exp(1i*k*r);
    end
  end
end

%% Calcul de l'onde de reference %%

% vecteur d'onde perpendiculaire a l'ecran (radians)
alpha = pi/2; % par rapport a l'aixe x
beta = pi/2; % par rapport a l'aixe y

% normes
nX = cos(alpha); 
nY = cos(beta);
nZ = sqrt(1 - nX^2 - nY^2);

refAmplitude = max(max(abs(objectWave)));

% l'onde de reference
referenceWave = zeros(samplesY, samplesX);

for column = 1:samplesX
  for row = 1:samplesY
    x = (column-1) * Delta + cornerX;
    y = (row-1) * Delta + cornerY;
    referenceWave(row,column) = refAmplitude * exp(1i*k*(x*nX + y*nY + hologramZ*nZ));
  end
end

%% [2] Representation de l'onde objet %%

%% Calcul de l'hologramme (interference entre l'onde objet et l'onde de reference) %%

% Modulation d'amplitude
itensityTotal = (objectWave + referenceWave).*conj(objectWave + referenceWave); % hologrammme
itensity = 2*real(objectWave.*conj(referenceWave));

norm(itensity)

figure()
plot(itensity)
colorbar

figure()
imagesc(itensity)
colorbar
colormap('gray')

%figure()
%surf(itensity)
%colorbar
