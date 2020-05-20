clear all;
close all;
clc;

%
%
%
%
%
% ATTENTION : time calculation, save variables!! (reconstruction and points)


% Start stopwatcher timer
tic;

% Number of point sources in the hologram 
nbPoints = 3;

% Number of holograms created with a P point sources
nbHolograms = 2;

% Number of class
nbClass = 3;
% 
% Dimensions of the hologram
hologramWidth  = 2e-3; % 2mm
hologramHeight = 2e-3; % 2mm

% Size of the hologram plane and reconstructed image plane
hologramSamplesX = 200;
hologramSamplesY = 200;

% Create database to store values
hDataset = zeros(hologramSamplesX, hologramSamplesY, nbHolograms*nbClass);
rDataset = zeros(hologramSamplesX, hologramSamplesY, nbHolograms*nbClass);
pDataset = zeros(nbHolograms*nbPoints*nbClass, 3);


for index = 1:nbClass

  [hologramDataset, reconstructionDataset, pointsDataset] = ...
                            generateDatasetOneClass(nbPoints, nbHolograms); 
  % Update databases 
  hDataset(:,:,) =
 
end

                            
##for index = 1:nbHolograms
##  
##  imagesc(hologramDataset(:,:, index))
##  figure
##  
##endfor

% Read elapsed time from stopwatch
toc;