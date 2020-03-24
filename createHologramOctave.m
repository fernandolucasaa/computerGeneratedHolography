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
% Parametres du plan l'hologramme
%

% plan de l'hologramme
hologramHeight = 3e-3; % 3mm
hologramWidth = 3e-3; % 3mm

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

%
% Parametres de la scene
%

% points de la scene
##points = [0, 0, -0.1;
##          0, 0.1, 0.1];
points = [0, 0, -0.1];

% Utiliser les positions de tous les echantillons
x = (0:(hologramSamplesX-1)) * samplingDistance + hologramCornerX;
y = (0:(hologramSamplesY-1)) * samplingDistance + hologramCornerY;
[xx, yy] = meshgrid(x, y);

% ------------------------------------------------------------------------------

%% [1] Calcul de l'onde objet (Nuage de points) %% 

fprintf('---------------------------------------\n')
fprintf('The object wave calculation...\n');

objectWave = zeros(hologramSamplesY, hologramSamplesX);

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
fprintf('\nThe object wave calculed!\n');


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
fprintf('The reference wave calculed!\n');

% ------------------------------------------------------------------------------

%% [2] Representation de l'onde objet %%

fprintf('\nThe hologram calculation... \n');

%% Calcul de l'hologramme (interference entre l'onde objet et l'onde de reference) %%

% Modulation d'amplitude
itensityTotal = (objectWave + referenceWave).*conj(objectWave + referenceWave);

itensity = 2*real(objectWave.*conj(referenceWave));
hologram = real(itensity);

%norm(itensity)

imagesc(x * 1e3, y * 1e3, hologram);
set(gca, 'YDir', 'normal'); % inverser la direction de l'axe y
colorbar;
title('Hologram');
xlabel('x [mm]');
ylabel('y [mm]');
axis('equal');

fprintf('The hologram calculated!\n');
fprintf('---------------------------------------\n')
