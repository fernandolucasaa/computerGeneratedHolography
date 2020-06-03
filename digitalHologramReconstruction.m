
%
% Digital reconstruction of the image from the hologram.
%
% Inputs:
% lambda
%  - Wavelength
% hologramHeight, hologramWidth
%  - Dimensions of the hologram (in meters)
% hologramZ
%  - Hologram is located in z = hologramZ
% samplingDistance
%  - Sampling distance in both xy axes
% targetZ
%  - Location of the reconstructed image in the z-axis
% hologram
%  - Calculated hologram (complex values)
% referenceWave
%  - Reference wave used for the hologram calculation
%
% Outputs:
% reconstruction_out
%  - 2D matrix with the reconstructed image for the given z position (targetZ)
%  - Note that this is a complex value
%

function [reconstruction_out] = digitalHologramReconstruction(lambda, hologramHeight, ...
                                hologramWidth, hologramZ, samplingDistance, targetZ, ...
                                hologram, referenceWave)
  
  %
  % Hologram reconstruction parameters
  %
  
  % Wave number
  k = 2*pi/lambda;
  
  % Dimensions de l'image reconstituee
  targetWidth = hologramWidth;
  targetHeight = hologramWidth;
  
  % Nombre de lignes (y) et de colonnes (x) du plan d'hologramme et d'image reconstituee
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);

  targetSamplesX = ceil(targetWidth / samplingDistance);
  targetSamplesY = ceil(targetHeight / samplingDistance);
  
  % Location of the "lower left" corner of the hologram
  % put center of the hologram to x = 0, y = 0
  hologramCornerX = - (hologramSamplesX - 1) * samplingDistance / 2;
  hologramCornerY = - (hologramSamplesY - 1) * samplingDistance / 2;
  
  targetCornerX = - (targetSamplesX - 1) * samplingDistance / 2;
  targetCornerY = - (targetSamplesY - 1) * samplingDistance / 2;
  
  
  %% Numerical propagation of the hologram %%
    
  % The propagation kernel calculation %
  
  % Propagation by convolution is calculated using matrices of size (cY rows) x (cX columns).
  cX = hologramSamplesX + targetSamplesX - 1; 
  cY = hologramSamplesY + targetSamplesY - 1; 
  
  % Shift between corners of the hologram and the target
  px = targetCornerX - hologramCornerX;
  py = targetCornerY - hologramCornerY;
  z0 = targetZ - hologramZ; % propagation dans l'aixe z
  
  % Auxiliary x, y coordinates for the convolution calculation
  auxCX = cX - hologramSamplesX + 1;
  auxCY = cY - hologramSamplesY + 1;
  auxX = (1-hologramSamplesX: auxCX-1) * samplingDistance + px;
  auxY = (1-hologramSamplesY: auxCY-1) * samplingDistance + py;
  [auxXX, auxYY] = meshgrid(auxX, auxY);
  
  % Calculate the Rayleigh-Sommerefeld propagation kernel
  r2 = auxXX.^2 + auxYY.^2 + z0^2;
  r = sqrt(r2);
  kernel = -1/(2*pi) * (1i*k - 1./r)*z0.*exp(1i*k*r) ./ r2 * samplingDistance^2;
  
 
  %% The reconstruction calculation %%
  
  % Create the auxiliary matrix of the correct size for the convolution
  % calculation. "Correct size" means that cyclic nature of the discrete
  % convolution will be suppressed
  auxMatrix = zeros(cY, cX);
  
  % Place a hologram illuminated by the reference wave to the auxiliary matrix. 
  % The rest of the samples is 0. This step is called "zero padding"
  auxMatrix(1:hologramSamplesY, 1:hologramSamplesX) = hologram .* conj(referenceWave);
  
  % Kernel has to be folded so that the entry for x = 0, y = 0
  % is in the first row, first column of the kernel matrix
  kernel = circularShift(kernel, hologramSamplesY - cX, hologramSamplesX - cX);
  
  % Calculate the cyclic convolution using FFT
  auxMatrixFT = fft2(auxMatrix) .* fft2(kernel);
  auxMatrix = ifft2(auxMatrixFT);
  
  % Pick the values that are not damaged by the cyclic nature of the FFT convolution
  reconstruction = auxMatrix(1:hologramSamplesY, 1:hologramSamplesX);
  
  reconstruction_out = reconstruction; 
  
end
