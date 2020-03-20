%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Generation numerique d'hologramme a partir de quelques sources ponctuelles en
% sommant leurs contributions (ondes spheriques). Les occultations ne sont pas 
% prises en compte pour l'instant !
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parametres initiaux %%

% longueur de l'onde 
lambda = 532e-9; % 532nm

% plan de l'hologramme
hologramHeight = 2e-3; % 2mm
hologramWidth = 2e-3; % 2mm
hologramZ = 0;

% nombre de pixels
N = 200;
Delta = 10e-6;
samplesX = hologramWidth / Delta; % 200 samples with sampling distance Delta
samplesY = hologramHeight / Delta; % 200 samples with sampling distance Delta

% coin de reference du plan de l'hologramme (inferieur a gauche)
cornerX = -hologramWidth / 2;
cornerY = -hologramHeight / 2;

% amplitude initiale
a = 1;

% nombre d'ondes
k = 2*pi/lambda;

% points de la scene
points = [0, 0, -0.5];

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

% angles d'incidience (radians)
alpha = 90 * (pi/180); % par rapport a l'aixe x
beta = 90.5 * (pi/180); % par rapport a l'aixe y

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
%optField = objectWave + referenceWave;
%hologram = optField .* conj(optField);
itensity = 2*real(objectWave.*conj(referenceWave));

plot(itensity)


