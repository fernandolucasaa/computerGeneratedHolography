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
dfile = 'output/commandWindow.txt';

if exist(dfile, 'file')
  delete(dfile);
end;

diary on;
diary (dfile);

fprintf('------------------------------------------------------------------\n');
fprintf('[Dataset generation]\n');

% Number of holograms per class
nbHolograms = 5;

% Number of classes (each class has a different number of point sources)
% Note that the first class will have 1 point source per hologram and the last
% class will have "nbClasses" point sources per hologram
nbClasses = 5;

fprintf('Number of holograms per class: %d\n', nbHolograms);
fprintf('Number of classes: %d\n', nbClasses);

% Generate datasets
fprintf('Dataset generation...\n');

[hDataset, rDataset, pDataset] = generateDatasets(nbHolograms, nbClasses);

% Save datasets
save('output/dataset/hDataset.mat', 'hDataset', '-v7');
save('output/dataset/rDataset.mat', 'rDataset', '-v7');
save('output/dataset/pDataset.mat', 'pDataset', '-v7');

% Plot figures                        
##for index = 1:(nbHolograms*nbClasses)
##  colormap('gray');
##  imagesc(hDataset(:,:, index))
##  if index != (nbHolograms*nbClasses)
##    figure
##  end
##endfor

fprintf('Dataset generated!\n');

% Read elapsed time from stopwatch
toc;

fprintf('------------------------------------------------------------------\n');

diary off;