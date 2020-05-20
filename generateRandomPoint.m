%
% Generate a random point in 3D space
%
% Inputs:
% rangeX, rangeY
%  - Range of values for the x and y axes
%  - Note that the range will be (-a,a)
% depth
%  - Maximum value of the 3D scene
%  - Note that the value must be positive
%
% Outputs:
% point3D
% - Position vector in 3D space (x, y, z) 
%

function [point3D] = generateRandomPoint(rangeX, rangeY, depth)
  
  % Generate a uniformly distributed value in the interval (-rangeX, rangeX)
  x = -rangeX + (rangeX + rangeX)*rand;
  
  % Generate a uniformly distributed value in the interval (-rangeY, rangeY)  
  y = -rangeY + (rangeY + rangeY)*rand;
  
  % Generate a value in the interval (-depth, 0)
  if (depth <= 0)
    fprintf('Incorrect depth argument');
  end
  z = -depth*rand;
  
  point3D = [x, y, z];
  
endfunction
