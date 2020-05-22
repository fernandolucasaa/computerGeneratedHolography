clear all;
close all;
clc;

%
% Generate a dataset of holograms with differents number of points in the scene.
% Note that the positions of the point sources are generated randomly.
%

% Start stopwatcher timer
tic;

% Number of holograms per class
nbHolograms = 5;

% Number of classes (each class has a different number of point sources)
% Note that the first class will have 1 point source per hologram and the last
% class will have "nbClass" point sources per hologram
nbClass = 5;

% Generate datasets
[hDataset, rDataset, pDataset] = generateDatasets(nbHolograms, nbClass);

% Save datasets
save('output/dataset/hDataset.mat', 'hDataset', '-v7');
save('output/dataset/rDataset.mat', 'rDataset', '-v7');
save('output/dataset/pDataset.mat', 'pDataset', '-v7');

% Plot figures                        
##for index = 1:(nbHolograms*nbClass)
##  colormap('gray');
##  imagesc(hDataset(:,:, index))
##  if index != (nbHolograms*nbClass)
##    figure
##  end
##endfor

% Read elapsed time from stopwatch
toc;