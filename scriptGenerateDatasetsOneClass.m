clear all;
close all;
clc;

%
% Generate a dataset of holograms with a specific number of points in the scene.
% Note that the positions of the point sources are generated randomly.
%

% Start stopwatcher timer
tic;

% Log Command Window text to file
dfile = 'output/commandWindowScriptGenerateDasetsOneClass.txt';

if exist(dfile, 'file')
  delete(dfile);
end;

diary on;
diary (dfile);

fprintf('------------------------------------------------------------------\n');
fprintf('[Dataset generation]\n');

% Number of holograms for the class
nbHolograms = 2000;

% Number of point sources per hologram
nbSources = 1;

fprintf('Number of sources per hologram: %d\n', nbSources);
fprintf('Total number of holograms computed: %d\n', (nbHolograms));
fprintf('Hologram properties:\n');
fprintf(' - Dimensions of the hologram: 2mm x 2mm\n');
fprintf(' - Resolutions (nb of samples): 200x200\n');
fprintf(' - Random arrangement of the particles in the 3D scene\n');
fprintf('   - X-axis, Y-axis ranges: (-1mm, 1mm)\n');
fprintf('   - Z-axis range: (-0.5m, -0.05m)\n');

% Generate datasets
fprintf('Dataset generation...\n');

% Size of the hologram plane and reconstructed image plane
hologramSamplesX = 200;
hologramSamplesY = 200;

% Create database to store values
hDataset = zeros(hologramSamplesX, hologramSamplesY, nbHolograms);
rDataset = zeros(hologramSamplesX, hologramSamplesY, nbHolograms);

% Length of matrix pDataset
pDataset = zeros(nbSources*nbHolograms, 3);

% Generate
[hologramDataset, reconstructionDataset, pointsDataset] = ...
                              generateDatasetOneClass(nbSources, nbHolograms); 

hDataset(:,:,:) = hologramDataset(:,:,:);
rDataset(:,:,:) = reconstructionDataset(:,:,:);
pDateset(:,:) = pointsDataset(:,:);

% Save datasets
save('output/dataset/oneClass/hDataset.mat', 'hDataset');
% save('output/dataset/oneClass/rDataset.mat', 'rDataset');
save('output/dataset/oneClass/pDataset.mat', 'pDataset');

% Plot figures                        
% ##for index = 1:(nbHolograms*nbClasses)
% ##  colormap('gray');
% ##  imagesc(hDataset(:,:, index))
% ##  if index != (nbHolograms*nbClasses)
% ##    figure
% ##  end
% ##endfor

fprintf('Dataset generated!\n');
fprintf('Names of saved structures in the dataset folder: hDataset.mat, rDataset.mat, pDataset.mat\n\n'); 

% Read elapsed time from stopwatch
toc;

fprintf('------------------------------------------------------------------\n');

diary off;