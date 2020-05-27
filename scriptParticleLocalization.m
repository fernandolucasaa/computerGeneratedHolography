close all
clear all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Localiser les positions des particules d'un hologrammme a partir de la restituition
% de l'hologramme plan par plan
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start stopwatcher timer
tic;

% Log Command Window text to file
dfile = 'output/commandWindowScriptParticleLocalization.txt';

if exist(dfile, 'file')
  delete(dfile);
end;

diary on;
diary (dfile);

% longueur de l'onde 
lambda = 500e-9; % 500nm (vert)

% plan de l'hologramme
hologramHeight = 2e-3; % 2mm
hologramWidth = 2e-3; % 2mm

% localisation dans l'aixe z
hologramZ = 0;

% distance d'echantillonnage dans les axes xy
samplingDistance = 10e-6;

% localisation des points dans l'aixe z
pointsZ = [-0.1, -0.2, -0.3, -0.2]; % -0.2m

% sauvegarder les images affiches en format jpf
img_jpg = false; % false

% Recuperer onde de reference et hologramme cree
referenceWave_out = (load('output/referenceWave_out.mat')).referenceWave_out;
hologram_out = (load('output/hologram_out.mat')).hologram_out;

% Parameters pour l'affichage
hologramSamplesX = ceil(hologramWidth / samplingDistance);
hologramSamplesY = ceil(hologramHeight / samplingDistance);
hologramCornerX = - (hologramSamplesX - 1) * samplingDistance / 2;
hologramCornerY = - (hologramSamplesY - 1) * samplingDistance / 2;
xAxis = (0:hologramSamplesX - 1)*samplingDistance + hologramCornerX;
yAxis = (0:hologramSamplesY - 1)*samplingDistance + hologramCornerY;

% Afficher les images
show = false; % false

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Emplacements des images reconstruite dans l'axe z
step = 0.1;
limit = 1;
targets = [-step:-step:-limit];

% Un valeur maximale par plan
maxValueVector = zeros(length(targets),1);
rowColumnVector = zeros(length(targets),2);

% Intensite minimale pour etre considere une source ponctuelle
threshold = 20000;

##maxValueVectorMultipleParticle = zeros(length(targets), 5);
##rowColumnVectorMultipleParticles = zeros(length(targets), 10);

% Faire la reconstruction pour differentes profondeurs
for targetZ = targets
  
  % reconstruction d'hologramme
  [reconstruction_out] = digitalHologramReconstruction(lambda, hologramHeight, ...
                         hologramWidth, hologramZ, samplingDistance, targetZ, ...
                         hologram_out, referenceWave_out, img_jpg);
  
  % valeur maximale
  maxValue = max(max(reconstruction_out)); 
  pos = find(targetZ == targets);
  maxValueVector(pos) = maxValue;
  
  % position (en pixels) de la valeur maximale
  [row, column] = find(reconstruction_out == maxValue);
  rowColumnVector(pos, 1) = row;
  rowColumnVector(pos, 2) = column;

  % Afficher les images reconstituee
  if (show == true)
    figure;
    colormap('gray');
    imagesc(xAxis * 1e3, yAxis * 1e3, abs(reconstruction_out));
    set(gca, 'YDir', 'normal');
    colorbar;
    title(['Reconstructed image (intensity) - ', num2str(pos)]);
    xlabel('x [mm]');
    ylabel('y [mm]');
    axis('image');
##    if (maxValue >= threshold)
##      figure
##      a = fft2(reconstruction_out);
##      a2 = fftshift(a);
##      colormap('gray');
##      imagesc(log(abs(a2)));
##      colorbar;  
##    end; 
  end;
  
end;

% Trier le vectuer
maxValueVectorSorted = sort(maxValueVector, 'descend');

% Calculer la quantite des sources dans la scene 3D
counter = 0;
for i = 1:length(maxValueVector)
  if (abs(maxValueVector(i)) >= threshold)
    counter = counter + 1;
  end;
end;

% Retrouver les positions des particules
pointsFound = zeros(counter, 3);

for i = 1:counter
  
  v = maxValueVectorSorted(i);
  index = find(v == maxValueVector);
  
  r = rowColumnVector(index, 1);
  c = rowColumnVector(index, 2); 
  z = targets(index);
  
  pointsFound(i, 1) = xAxis(c);
  pointsFound(i, 2) = yAxis(r);
  pointsFound(i, 3) = z;
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('----- Particles localization -----\n');
fprintf('Dimensions of the hologram: %d m vs %d m\n', hologramHeight, hologramWidth);
fprintf('Resolution of the hologram: %d pixels vs %d pixels\n', hologramSamplesX, hologramSamplesY);
fprintf('Limit of the reconstruction (z): -%.2f m\n', limit);
fprintf('Number of segmentations calculated: %d planes\n', length(targets));
fprintf('Value of the threshold: %.2f\n', threshold);
fprintf('Distance of the step: %.2f m\n', step);

% Sources ponctuelles
points = load('output/points.mat').points;

for source = 1:size(points, 1)
  fprintf('\nPoint light source %d of %d: [%d, %d, %d]', source, size(points, 1), ...
  points(source, 1), points(source, 2), points(source, 3));
end

% Points localises
fprintf('\n\n')
for source = 1:size(pointsFound, 1)
  fprintf('Detected particle in (x,y,z) = [%.3f, %.3f, %.3f]\n', points(source, 1), ...
  points(source, 2), points(source, 3));
end

scatter3(points(:, 1), points(:, 3), points(:, 2), 'filled')
xlabel('x'); ylabel('z (depth)'); zlabel('y')
title('Point light sources in the 3D scene');

figure
scatter3(pointsFound(:, 1), pointsFound(:, 3), pointsFound(:, 2), '*')
xlabel('x'); ylabel('z (depth)'); zlabel('y')
title('Points found from the reconstruction segmentation');

##figure
##scatter(points(:, 1), points(:, 2))
##xlabel('x'); ylabel('y'); grid on; 
##title('Point light sources - xy axis')
##
##figure
##scatter(pointsFound(:, 1), pointsFound(:, 2))
##xlabel('x'); ylabel('y'); grid on; 
##title('Points found - xy axis')

fprintf('\n');

% Read elapsed time from stopwatch
toc;

fprintf('------------------------------------------------------------------\n');

diary off;