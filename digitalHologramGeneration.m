
%
% Digital hologram generation from a few point sources by summing their contributions
% (spherical waves). The occultations are not taken into account for now!
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
% windowFunction
%  - Window to limit the contribution area (avoid the aliasing effect)
% points3D
%  - 3D point position (point source)
%
% Outputs:
% hologram_output
%  - 2D matrix with the calculated hologram
%  - Note that this is a complex value
% referenceWave_output
%  - 2D matrix with the reference wave used for the hologram calculation
%  - Note that we are going to use this variable in the hologram reconstruction
% x_out, y_out
%  - 1D vectors with the positions of the samples in the hologram plan
%  - e.g. (-hologramWidth/2, ..., hologramWidth/2)
%
% TODO
%  - Remove old input parameters: hologramZ, pointsChoice (removed),
%    pointsZ (removed)
%  - Remove pointsChoice's if (removed)
%

function [hologram_output, referenceWave_output, x_out, y_out] = digitalHologramGeneration(lambda, ...
          hologramHeight, hologramWidth, hologramZ, samplingDistance, windowFunction, points3D)
  
  %% [0] Initialization %%
  
  %
  % Hologram parameters
  %
  
  % Number of rows (y) and columns (x) of the hologram (resolution of the hologram
  % in pixels)
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);
  
  % Location of the "lower left" corner of the hologram
  % put center of the hologram to x = 0, y = 0
  hologramCornerX = - (hologramSamplesX - 1) * samplingDistance / 2;
  hologramCornerY = - (hologramSamplesY - 1) * samplingDistance / 2;
  
  % Initial amplitude
  a = 1;
  
  % Wave number
  k = 2*pi/lambda;
  
  %
  % Scene parameters
  %
  
  % 3D point position (point source)    
  points = points3D;
  
  % We will use positions of all samples in the hologram plane in 
  % the following calculation of the hologram.
  x = (0:(hologramSamplesX-1)) * samplingDistance + hologramCornerX;
  y = (0:(hologramSamplesY-1)) * samplingDistance + hologramCornerY;
  [xx, yy] = meshgrid(x, y);
  
  x_out = x;
  y_out = y;
  
  % ------------------------------------------------------------------------------
  
  %% [1] The object wave calculation (Point cloud approach) %% 
  
  objectWave = zeros(hologramSamplesY, hologramSamplesX);
  
  % Superposition of all the spherical waves
  for source = 1:size(points, 1)

    % For backpropagation, flip the sign of the imaginary unit
    if (points(source, 3) > hologramZ)
% ##      fprintf('\nAttention! The point is in the front of the hologram plan!\n');
      ii = -1i;
    else
      ii = 1i;
    end
    
    % Window function
    h = ones(hologramSamplesX, hologramSamplesY);
    
    % Limiting the contribution area
    if (windowFunction)
        
        % Region of contribution of the light point
        p = samplingDistance; % Sampling step
        Rmax = abs(points(source,3) * tan(asin(lambda/(2*p))));
        distance = sqrt((xx - points(source, 1)).^2 + (yy - points(source, 2)).^2);
        indices = find(distance > Rmax);
        h(indices) = 0;
      
    end
    
    % Distance oblique
    r = sqrt((xx - points(source, 1)).^2 + (yy - points(source, 2)).^2 + (hologramZ - points(source, 3)).^2);
    objectWave = objectWave + a .* exp(ii*k*r) ./ r .* h;
  
  end
  
  %% Calculation of the reference wave %%
  
  % Angles of direction of the reference wave with the axes x and y
  % The angle gamma between the reference light direction and the axis z can be calculated as
  % gamma = sqrt(1 - alpha^2 - beta^2), but we will not need it.
  % Wave vector perpendicular to the screen (radians)
  alpha = pi/2; % in relation to the x-axis
  beta = pi/2;  % in relation to the y-axis
  
  % Direction vector of the reference wave
  nX = cos(alpha); 
  nY = cos(beta);
  nZ = sqrt(1 - nX^2 - nY^2);
  
  % Allow nZ < 0, just in case...
  if (nZ > 0)
    ii = 1i;
  else
    ii = -1i;
% ##    fprintf('\nAttention! The direction vector is in the opposite direction!\n');
  end

  % Amplitude of the reference wave  
  refAmplitude = max(max(abs(objectWave)));
  
  % Reference wave
  referenceWave = refAmplitude * exp(ii * k * (xx*nX + yy*nY + hologramZ*nZ));
  referenceWave_output = referenceWave;
  
  % ------------------------------------------------------------------------------
  
  %% [2] Representation of the object wave %%
  
  %% Calculation of the hologram (interference between the object wave and the reference wave) %%
  
  % Modulation d'amplitude
  itensityTotal = (objectWave + referenceWave).*conj(objectWave + referenceWave);
  itensity = 2*(objectWave.*conj(referenceWave));
  
  % Ensure that the imaginary part is zero
  %itensity = 2*real(objectWave.*conj(referenceWave));
  %hologram = real(itensity);
  %hologram_output = hologram;
  
  % Use complex value
  hologram_output = itensity;
  
end  
