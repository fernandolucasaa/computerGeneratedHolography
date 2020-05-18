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
dfile = 'output/commandWindow.txt';

if exist(dfile, 'file')
  delete(dfile);
end;

diary on;
diary (dfile);

% Generer un hologramme (1) ou video holographique (2) ou plusieurs hologrammes
% differents pour generer une base de donnnes pour tester les algorithmes de
% separation des sources (3)
option = 1; % 1

% Generer l'image restitue
reconstructionChoice = true; % false

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
pointsChoice = 2; % 1

% localisation des points dans l'aixe z (pour l'exemple ci-dessus)
pointsZ = [-0.1, -0.2, -0.3, -0.2]; % -0.2m

% fenetre pour limiter la zone de contribution (eviter le repliement du spectre)
windowFunction = true; % true

% sauvegarder les images affiches en format jpf
img_jpg = false; % false

%
% Parametres du plan de l'image reconstituee
%

% Emplacement de l'image reconstruite dans l'axe z
targetZ = -0.3; % -0.2m

%
% Calculs
%

if (option == 1)    % creer seulement une image holographique

  fprintf('---------------------------------------\n');
  fprintf('Hologram calculation...\n'); 

  % calcul d'hologramme
  [hologram_out, referenceWave_out, x_out, y_out] = digitalHologramGeneration(lambda, hologramHeight, ...
                                      hologramWidth, hologramZ, samplingDistance, pointsChoice, ...
                                      pointsZ, windowFunction);
  
  fprintf('Hologram calculated!\n');
  hologram_out = real(hologram_out);
  
  % sauvagarder
  save('output/referenceWave_out.mat', 'referenceWave_out', '-v7');
  save('output/hologram_out.mat', 'hologram_out', '-v7');
  
  % Afficher l'hologramme
  titlePlot = 'Hologram';
  xName = 'x [mm]';
  yName = 'y [mm]';
  fileName = 'hologram_out';
  plotImage(hologram_out, x_out, y_out, img_jpg, titlePlot, xName, yName, fileName);  
  
  fprintf('---------------------------------------\n'); 

  if (reconstructionChoice == true)
    
    fprintf('Hologram reconstruction...\n');
    
    % reconstruction d'hologramme
    [reconstruction_out] = digitalHologramReconstruction(lambda, hologramHeight, ...
                           hologramWidth, hologramZ, samplingDistance, targetZ, ...
                           hologram_out, referenceWave_out);
                           
    fprintf('Hologram reconstructed!\n');
    
    % Afficher l'image restitue
    titlePlot = 'Reconstructed image (intensity)';
    xName = 'x [mm]';
    yName = 'y [mm]';
    fileName = 'reconstruction_out';
    figure()
    plotImage(abs(reconstruction_out), x_out, y_out, img_jpg, titlePlot, xName, yName, fileName);  

    fprintf('---------------------------------------\n');
    
  end;
  
elseif (option == 2)    % creer un video holographique
  
  % quantite de trames
  nFrames = 3;
  pointsChoiceVideo = [4 5 6];
  
  % nombre de lignes (y) et de colonnes (x) du plan d'hologramme
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);
  
  hologram_video = zeros(hologramSamplesX, hologramSamplesY, nFrames);
  
  fprintf('---------------------------------------\n');
  
  for frame = 1:nFrames    
    
    fprintf('Hologram calculation...\n');
    
    [hologram_out, referenceWave_out, x_out, y_out] = digitalHologramGeneration(lambda, hologramHeight, ...
                                hologramWidth, hologramZ, samplingDistance, pointsChoiceVideo(frame), ...
                                pointsZ, windowFunction);
                                
    hologram_video(:,:,frame) = real(hologram_out);
    
    fprintf('Hologram calculated!\n');
    
    % sauvagarder
    save('output/hologram_video.mat', 'hologram_video', '-v7');
    
    % Afficher l'hologramme
    figure(frame)
    titlePlot = 'Hologram';
    xName = 'x [mm]';
    yName = 'y [mm]';
    fileName = 'hologram_out';
    plotImage(hologram_video(:,:,frame), x_out, y_out, img_jpg, titlePlot, xName, yName, fileName);
  
  end;
  
  fprintf('---------------------------------------\n');

elseif (option == 3)

  % nombre d'hologrammes a creer
  nbSamples = 10;
  
  % nombre de lignes (y) et de colonnes (x) du plan d'hologramme
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);
  
  hologram_dataset = zeros(hologramSamplesX, hologramSamplesY, nbSamples);
  
  % differents combinations des interferences geres par 2 particules
  allPoints3D = [ -hologramWidth / 4, 0, -0.2;
                   hologramWidth / 4, 0, -0.2;
                   
                  -hologramWidth / 4, hologramWidth / 4, -0.2;
                   hologramWidth / 4, hologramWidth / 4, -0.2;
                   
                  -hologramWidth / 4, -hologramWidth / 4, -0.2;
                   hologramWidth / 4, -hologramWidth / 4, -0.2
                   
                   -hologramWidth / 4, hologramWidth / 4, -0.2;
                   hologramWidth / 4, -hologramWidth / 4, -0.2
                   
                   -hologramWidth / 4, -hologramWidth / 4, -0.2;
                   hologramWidth / 4, hologramWidth / 4, -0.2
                   
                   -hologramWidth / 4, 0, -0.2;
                   hologramWidth / 4, hologramWidth / 4, -0.2
                   
                   -hologramWidth / 4, -hologramWidth / 4, -0.2;
                   hologramWidth / 4, 0, -0.2;
                   
                   -hologramWidth / 4, hologramWidth / 4, -0.2;
                   hologramWidth / 4, 0, -0.2
                   
                   -hologramWidth / 4, 0, -0.2;
                   hologramWidth / 4, -hologramWidth / 4, -0.2;
                   
                   -hologramWidth / 4, 0, -0.1;
                   hologramWidth / 4, 0, -0.1;
  ];
  
  points3D = zeros(2, 3);
  
  for i = 1:nbSamples
    
    % seulement deux sources par fois
    points3D(1, :) = [allPoints3D(2*i - 1, :)];
    points3D(2, :) = [allPoints3D(2*i, :)];
    
    % pointsChoice egual a 8 pour mettre les points comme entree
    [hologram_out, referenceWave_out, x_out, y_out] = digitalHologramGeneration(lambda, hologramHeight, ...
                                hologramWidth, hologramZ, samplingDistance, 8, ...
                                pointsZ, windowFunction, points3D);
                                
    hologram_dataset(:,:,i) = hologram_out;
    
    fileName = ["output/dataset/hologram_" num2str(i) ".mat"];
    
    % sauvagarder
    save(fileName, 'hologram_out');
    
    % Afficher l'hologramme
##    figure(i)
##    plotImage(hologram_dataset(:,:,i), x_out, y_out, img_jpg);
    
  end;
  
  % sauvagarder
  fileName = ["output/dataset/hologram_dataset_" num2str(nbSamples) "_samples_2_sources.mat"];
  save(fileName, 'hologram_dataset', '-v7');
  
end;

toc;

fprintf('---------------------------------------\n')
  
diary off;