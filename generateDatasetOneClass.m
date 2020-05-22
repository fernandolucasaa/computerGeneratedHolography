
%
% Compute a certain number of holograms (nbHolograms) with a certain number of
% point sources in the 3D scene (nbPoints)
%
% Inputs:
% nbPoints
%  - Number of point sources in the hologram 
% nbHolograms
%  - Number of holograms created with nbPoints point sources
%
% Outputs:
% hologramDataset
%  - Array NxNxH with all holograms calculated for the specific class
% reconstructionDataset
%  - Array NxNxH with all reconstructed images calculated for the specific class
% pointsDataset
%  - Matrix (H*P)x3 with the positions of all the points for the specific class
%

function [hologramDataset, reconstructionDataset, pointsDataset] = ...
                            generateDatasetOneClass(nbPoints, nbHolograms)

  % Location of the reconstructed image in the z-axis
  targetZ = -0.2;

  % Dimensions of the hologram
  hologramWidth  = 2e-3; % 2mm
  hologramHeight = 2e-3; % 2mm

  % Size of the hologram plane and reconstructed image plane
  hologramSamplesX = 200;
  hologramSamplesY = 200;

  % Range of values for particle position
  rangeX = hologramWidth/2;
  rangeY = hologramHeight/2;
  depth = 0.2;

  % Create database to store values
  hologramDataset = zeros(hologramSamplesX, hologramSamplesY, nbHolograms);
  reconstructionDataset = zeros(hologramSamplesX, hologramSamplesY, nbHolograms);

  pointsDataset = zeros(nbHolograms*nbPoints, 3);
  points3DScene = zeros(nbPoints, 3);
                                    
  % Auxiliar
  counter = 1;
      
  for index = 1:nbHolograms
    
    for j = 1:nbPoints
      
      % Generate a random 3D point
      point3D = generateRandomPoint(rangeX, rangeY, depth);
      
      % Force z value to be in accordance with the place of reconstruction (BUG)
      point3D(1,3) = targetZ; 
    
      % Add the point created in the matrix with all points 
      points3DScene(j, :) = point3D;
   
      % Database update
      pointsDataset(counter,:) = point3D;
      counter = counter + 1;
    
    end
     
    % Compute hologram
    [hologram, reconstruction] = computeHologram(points3DScene, targetZ);
    
    % Database update
    hologramDataset(:,:,index) = hologram;
    reconstructionDataset(:,:,index) = reconstruction;
      
  endfor
 
endfunction