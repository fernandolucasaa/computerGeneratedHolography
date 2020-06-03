
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
% pointsChoice
%  - Choose between pre-determined points for the 3D scene (1-7)
%  - Note that the choice only influences in the number of points and the positions
%  - on the x and y axis
% pointsZ
%  - 1D vector with the depths of the pre-determined points
% windowFunction
%  - Window to limit the contribution area (avoid the aliasing effect)
% points3D
%  - 3D point position (point source)
%  - Note that this parameter is used when the 'pointsChoice' equals 8
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
%  - Remove old input parameters: hologramZ, pointsChoice, pointsZ
%  - Remove pointsChoice's if.
%

function [hologram_output, referenceWave_output, x_out, y_out] = digitalHologramGeneration(lambda, ...
          hologramHeight, hologramWidth, hologramZ, samplingDistance, pointsChoice, ...
          pointsZ, windowFunction, points3D)
  
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
  
  % Scene points
  if (pointsChoice == 1)
    points = [0, 0, pointsZ(1)];
  elseif(pointsChoice == 2)
    points = [0, 0, pointsZ(1);
              -hologramWidth / 4, -hologramHeight / 4, pointsZ(2);
              hologramWidth / 4, hologramHeight / 4, pointsZ(3)];
  elseif(pointsChoice == 3)
    points = [-hologramWidth / 4, hologramWidth / 4, pointsZ(1);
              hologramWidth / 4, hologramWidth / 4, pointsZ(2);
              -hologramWidth / 4, -hologramHeight / 4, pointsZ(3);
              hologramWidth / 4, -hologramHeight / 4, pointsZ(4)];
  elseif(pointsChoice == 4)
    points = [0, 0, -0.1];
  elseif(pointsChoice == 5)
    points = [0, 0, -0.2];
  elseif(pointsChoice == 6)
    points = [0, 0, -0.3];
  elseif(pointsChoice == 7)
    points = [-hologramWidth / 4, 0, -0.2; 
              hologramWidth / 4, 0, -0.2;];
  elseif(pointsChoice == 8)
    points = points3D;
  end;
  
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
