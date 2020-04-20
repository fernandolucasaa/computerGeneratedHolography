close all
clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Generation numerique d'hologramme et leur reconstructon
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% demarrage du chronometre
tic; 

% Enregistrer le texte de la fenetre de commande
dfile = 'commandWindow.txt';

if exist(dfile, 'file')
  delete(dfile);
end;

diary on;
diary (dfile);

% Generer un hologramme (1) ou video holographique (2)
option = 2; % 1

% Generer l'image restitue
reconstructionChoice = false; % false

%
% Parametres generaux (tous les dimensions sont en metres)
%

% longueur de l'onde 
lambda = 500e-9; % 500nm (vert)

%
% Parametres du plan de l'hologramme
%

% plan de l'hologramme
hologramHeight = 2e-3; % 2mm
hologramWidth = 2e-3; % 2mm

% localisation dans l'aixe z
hologramZ = 0;

% distance d'echantillonnage dans les axes xy
samplingDistance = 10e-6;

% points de la scene 3D
% 1 : un point source
% 2 : trois points sources
% 3 : quatre points sources
%
%     |           |           |
%     |           | *       * | *
% - - * - -   - - * - -   - - - - -  
%     |         * |         * | *
%     |           |           |
%
pointsChoice = 4; % 1

% localisation des points dans l'aixe z
pointsZ = [-0.2, -0.2, -0.2, -0.2]; % -0.2m

% fenetre pour limiter la zone de contribution (eviter le repliement du spectre)
windowFunction = true; % true

% sauvegarder les images affiches en format jpf
img_jpg = false; % false

%
% Parametres du plan de l'image reconstituee
%

% Emplacement de l'image reconstruite dans l'axe z
targetZ = -0.2; % -0.2m

%
% Calculs
%

if (option == 1)    % creer seulement une image holographique

  fprintf('---------------------------------------\n'); 

  % calcul d'hologramme
  [hologram_out, referenceWave_out] = digitalHologramGeneration(lambda, hologramHeight, ...
                                      hologramWidth, hologramZ, samplingDistance, pointsChoice, ...
                                      pointsZ, windowFunction, img_jpg);
  
  % sauvagarder
  save('hologram_image.mat', 'hologram_out'); 
 
  % Afficher l'hologramme
  colormap('gray');
  colorbar;
  imagesc(hologram_out);

  fprintf('---------------------------------------\n'); 

  if (reconstructionChoice == true)
    
    % reconstruction d'hologramme
    [reconstruction_out] = digitalHologramReconstruction(lambda, hologramHeight, ...
                           hologramWidth, hologramZ, samplingDistance, targetZ, ...
                           hologram_out, referenceWave_out, img_jpg);
  end;
  
elseif (option == 2)    % creer un video holographique
  
  % quantite de trames
  nFrames = 3;
  pointsChoiceVideo = [4 5 6];
  
  % nombre de lignes (y) et de colonnes (x) du plan d'hologramme
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);
  
  hologram_video = zeros(hologramSamplesX, hologramSamplesY, nFrames);
  
  for frame = 1:nFrames    
    
    hologram_video(:,:,frame) = digitalHologramGeneration(lambda, hologramHeight, ...
                                hologramWidth, hologramZ, samplingDistance, pointsChoiceVideo(frame), ...
                                pointsZ, windowFunction, img_jpg);
    % sauvagarder
    save('hologram_video.mat', 'hologram_video');
    
    % Affichage
    figure(frame)
    colormap('gray')
    colorbar
    imagesc(hologram_video(:,:,frame))
  
  end;
  
end;

toc;

fprintf('---------------------------------------\n')
  
diary off;