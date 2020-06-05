
%
% Compute a digital hologram (holographic interference pattern of an reference
% wave with an object wave) and also a numerical simulation of hologram reconstruction
% for a specific location
%
% Inputs:
% points3D
%  - Matrix of 3D points containt the positions (x, y, z) of the point sources
% targetZ
%  - Location of the reconstructed image in the z-axis
%
% Outputs:
% hologram
%  - 2D matrix with the interference patterns in the hologram plane
%  - Note that the values are complex
% reconstruction
%  - 2D matrix with the reconstructed image 
%  - Note that the values are complex
%
% TODO
% - Improve 'digitalHologramGeneration' function, some input parameters are
%   not necessary
% - Insert others paramaters in the function
%

function [hologram, reconstruction] = computeHologram(points3D, targetZ)
  
  %
  % GENERAL CALCULATION PARAMETERS
  %  
  
  % Wavelength and the wave number
  lambda = 500e-9; % 500nm (green)
  
  %
  % HOLOGRAM PARAMETERS
  %

  % Dimensions of the hologram
  hologramWidth  = 2e-3; % 2mm
  hologramHeight = 2e-3; % 2mm
  
  % Hologram is located in z = hologramZ 
  hologramZ = 0;

  % Sampling distance in both xy axes
  samplingDistance = 10e-6;  % common spatial light modulators

  %
  % CALCUL PARAMETERS
  %
  
  % Window to limit the contribution area (avoid aliasing effect)
  windowFunction = true; % true
  
  % ----------------------------------------------------------------------------

  % 1. The hologram calculation
  [hologram_out, referenceWave_out] = digitalHologramGeneration( ...
                                      lambda, hologramHeight, hologramWidth, hologramZ, ...
                                      samplingDistance, windowFunction, points3D);
  
  % Ensure that the imaginary part is zero
  %hologram = real(hologram_out);
  
  % Use complex value
  hologram = hologram_out;
  
  % 2. The reconstruction calculation
  [reconstruction_out] = digitalHologramReconstruction(lambda, hologramHeight, ...
                           hologramWidth, hologramZ, samplingDistance, targetZ, ...
                           hologram_out, referenceWave_out);
  
  % Avoid complex value
  %reconstruction = abs(reconstruction_out);
  
  % Use complex value
  reconstruction = reconstruction_out;
  
end
