
%
% Generate a dataset of holograms with differents number of points in the scene.
% Note that the positions of the point sources are generated randomly.
%
% Inputs:
% nbHolograms
%  - Number of holograms per class
% nbClass
%  - Number of classes (each class has a different number of point sources)
%  - Note that the first class will have 1 point source per hologram and the last
%    class will have "nbClass" point sources per hologram
%
% Outputs:
% hDataset
%  - Array NxNx(nbHolograms*nbClass) with holograms calculated for all classes
% rDataset
%  - Array NxNx(nbHolograms*nbClass) with reconstructed images calculated for all 
%    classes
% pDataset
%  -  3D matrix with points' positions for all classes
%

function [hDataset, rDataset, pDataset] = generateDatasets(nbHolograms, nbClass)

  % Size of the hologram plane and reconstructed image plane
  hologramSamplesX = 200;
  hologramSamplesY = 200;

  % Create database to store values
  hDataset = zeros(hologramSamplesX, hologramSamplesY, nbHolograms*nbClass);
  rDataset = zeros(hologramSamplesX, hologramSamplesY, nbHolograms*nbClass);

  % Length of matrix pDataset
  aux = 0;
  for i = 1:nbClass
    aux = aux + i*nbHolograms; % i = number of points in the holograms
  end
  pDataset = zeros(aux, 3);

  % Auxiliary variables
  initial = 1;
  final = initial + nbHolograms - 1;
  init = 1;

  for index = 1:nbClass

    [hologramDataset, reconstructionDataset, pointsDataset] = ...
                              generateDatasetOneClass(index, nbHolograms); 
    % Update databases
    %initial
    %final  
    hDataset(:,:, initial:final) = hologramDataset;
    rDataset(:,:, initial:final) = reconstructionDataset;
    
    % Update auxiliary variables
    initial = index*nbHolograms + 1;
    final = initial + nbHolograms - 1;
   
    % Update database and auxiliary variables
    %init
    %fin
    fin = init + size(pointsDataset,1) - 1;
    pDataset(init:fin,:) = pointsDataset(:,:);
    init = fin + 1;
    
  end

end