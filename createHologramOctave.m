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

%% [0] Parametres initiaux - Initialisation %%

% Tous les dimensions sont en metres

% longueur de l'onde 
lambda = 500e-9; % 500nm (vert)

% coleur des images affichés
colormap('gray')

%
% Parametres de l'hologramme
%

% plan de l'hologramme
hologramHeight = 3e-3; % 3nm
hologramWidth = 3e-3; % 3nm

% localisation dans l'aixe x
hologramZ = 0;

% distance d'echantillonnage dans les axes xy
samplingDistance = 10e-6;

% nombre de lignes (y) et de colonnes (x) du plan d'hologramme
hologramSamplesX = ceil(hologramWidth / samplingDistance);
hologramSamplesY = ceil(hologramHeight / samplingDistance);

% emplacement du coin "inférieur gauche" de l'hologramme (coin de reference)
% mettre le centre de l'hologramme a x = 0, y = 0
hologramCornerX = - (hologramSamplesX - 1) * samplingDistance / 2;
hologramCornerY = - (hologramSamplesY - 1) * samplingDistance / 2;

% amplitude initiale
a = 1;

% nombre d'onde
k = 2*pi/lambda;

Delta = 10e-6;
samplesX = hologramWidth / Delta; % 300 samples with sampling distance Delta
samplesY = hologramHeight / Delta; % 300 samples with sampling distance Delta

%
% Parametres de la scene
%

% points de la scene
points = [0, 0, -0.1];

% Utiliser les positions de tous les echantillons
x = (0:(hologramSamplesX-1)) * samplingDistance + hologramCornerX;
y = (0:(hologramSamplesY-1)) * samplingDistance + hologramCornerY;
[xx, yy] = meshgrid(x, y);

% ------------------------------------------------------------------------------

%% [1] Calcul de l'onde objet (Nuage de points) %% 

fprintf('\nHologram recording:\n');
fprintf('\nThe object wave calculation...\n');

objectWave = zeros(hologramSamplesY, hologramSamplesX);

##% superposition de tous les ondes spheriques
##for s = 1:rows(points)
##  for column = 1:samplesX
##    for row = 1:samplesY
##      x = (column-1) * Delta + cornerX;
##      y = (row-1) * Delta + cornerY;
##      % distance oblique
##      r = sqrt((x - points(s, 1))^2 + (y - points(s, 2))^2 + (hologramZ - points(s, 3))^2);
##      objectWave(row,column) = (a / r)*exp(1i*k*r);
##    end
##  end
##end

% superposition de tous les ondes spheriques
for source = 1:size(points, 1)
  fprintf('\rPoint light source %d of %d    ', source, size(points, 1));
  
  % for backpropagation, flip the sign of the imaginary unit
  
  % ?????????????
  if (points(source, 3) > hologramZ)
    ii = -1i;
  else
    ii = 1i;
  end
  
  % ??????????
  
  % distance oblique
  r = sqrt((xx - points(source, 1)).^2 + (yy - points(source, 2)).^2 + (hologramZ - points(source, 3)).^2);
	objectWave = objectWave + a .* exp(ii*k*r) ./ r;
end
fprintf('\n');


%% Calcul de l'onde de reference %%

fprintf('\nThe reference wave calculation...\n');

% les angles de direction de l'onde de reference avec le axes x et y
% vecteur d'onde perpendiculaire a l'ecran (radians)
alpha = pi/2; % par rapport a l'aixe x
beta = pi/2; % par rapport a l'aixe y

% vecteur de direction
nX = cos(alpha); 
nY = cos(beta);
nZ = sqrt(1 - nX^2 - nY^2);

% ?????????????
% allow nZ < 0, just in case...
if (nZ > 0)
  ii = 1i;
else
  ii = -1i;
end
% ???????????

refAmplitude = max(max(abs(objectWave)));

% l'onde de reference
referenceWave = refAmplitude * exp(ii * k * (xx*nX + yy*nY + hologramZ*nZ));

##for column = 1:samplesX
##  for row = 1:samplesY
##    x = (column-1) * Delta + cornerX;
##    y = (row-1) * Delta + cornerY;
##    referenceWave(row,column) = refAmplitude * exp(1i*k*(x*nX + y*nY + hologramZ*nZ));
##  end
##end

% ------------------------------------------------------------------------------

%% [2] Representation de l'onde objet %%

fprintf('\nThe hologram calculation...\n');

%% Calcul de l'hologramme (interference entre l'onde objet et l'onde de reference) %%

% Modulation d'amplitude
itensityTotal = (objectWave + referenceWave).*conj(objectWave + referenceWave);

itensity = 2*real(objectWave.*conj(referenceWave));
hologram = real(itensity);

norm(itensity)

##figure()
##imagesc(x * 1e3, y * 1e3, hologram);
##set(gca, 'YDir', 'normal');
##title('Hologram (intensity)');
##xlabel('x [mm]');
##ylabel('y [mm]');
##axis('equal');


% figure()
imagesc(x * 1e3, y * 1e3, hologram);
set(gca, 'YDir', 'normal');
colorbar;
title('Hologram (itensity)');
xlabel('x [mm]');
ylabel('y [mm]');
axis('equal');

##xticks([1, samplesX])
##xticklabels({-hologramWidth/2*1e+3, hologramWidth/2*1e+3})
##yticks([1, samplesY])
##yticklabels({hologramWidth/2*1e+3, -hologramWidth/2*1e+3})

%figure()
%surf(itensity)
%colorbar
