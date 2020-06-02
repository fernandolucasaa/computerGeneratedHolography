clear all;
close all;
clc;

%
% Generate a dataset of holograms with differents number of points in the scene.
% Note that the positions of the point sources are generated randomly.
%

% Start stopwatcher timer
tic;

% Log Command Window text to file
dfile = 'output/commandWindowScriptGenerateDasets.txt';

if exist(dfile, 'file')
  delete(dfile);
end;

diary on;
diary (dfile);

fprintf('------------------------------------------------------------------\n');
fprintf('[Dataset generation]\n');

% Number of holograms per class
nbHolograms = 200;

% Number of classes (each class has a different number of point sources)
% Note that the first class will have 1 point source per hologram and the last
% class will have "nbClasses" point sources per hologram
nbClasses = 5; % Do not change

fprintf('Number of holograms per class: %d\n', nbHolograms);
fprintf('Number of classes: %d\n', nbClasses);
fprintf('Total number of holograms computed: %d\n', (nbHolograms * nbClasses));
fprintf('Hologram properties:\n');
fprintf(' - Dimensions of the hologram: 2mm x 2mm\n');
fprintf(' - Resolutions (nb of samples): 200x200\n');
fprintf(' - Random arrangement of the particles in the 3D scene\n');
fprintf('   - X-axis, Y-axis ranges: (-1mm, 1mm)\n');
fprintf('   - Z-axis range: (-0.5m, -0.05m)\n');

% Generate datasets
fprintf('Dataset generation...\n');

[hDataset, rDataset, pDataset] = generateDatasets(nbHolograms, nbClasses);

% Save datasets
%##save('output/dataset/hDataset.mat', 'hDataset', '-v7'); % Problems with Octave
%##save('output/dataset/rDataset.mat', 'rDataset', '-v7'); % to save in matlab format
%##save('output/dataset/pDataset.mat', 'pDataset', '-v7');
save('output/dataset/hDataset.mat', 'hDataset');
save('output/dataset/rDataset.mat', 'rDataset');
save('output/dataset/pDataset.mat', 'pDataset');

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