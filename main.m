close all
clear
clc

%
% Digital hologram generation and their reconstruction
%

% Start stopwatcher timer
tic; 

% Log Command Window text to file
dfile = 'output/commandWindowMain.txt';

if exist(dfile, 'file')
  delete(dfile);
end;

diary on;
diary (dfile);

% Compute an hologram (1) or a video hologram (2) or several different holograms 
% to create a database for testing source separation algorithms (3)
option = 1; % 1

% Compute the restituted image
reconstructionChoice = true; % false

%
% General parameters (all dimensions in meters)
%

% Wavelength
lambda = 500e-9; % 500nm (green)

%
% Hologram parameters
%

% Dimensions of the hologram
hologramHeight = 2e-3; % 2mm
hologramWidth = 2e-3; % 2mm

% Hologram is located in z = hologramZ 
hologramZ = 0;

% Sampling distance in both xy axes
samplingDistance = 10e-6;

% Points of the 3D scene
% 1 : one source point
% 2 : three source points
% 3 : four source points
%
%     |           |           |
%     |           | *       * | *
% - - * - -   - - * - -   - - - - -  
%     |         * |         * | *
%     |           |           |
%
pointsChoice = 1; % 1

% Location of points in the z-axis (for the example above)
pointsZ = [-0.2, -0.2, -0.2, -0.2]; % -0.2m

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
elseif(pointsChoice == 7)
    points = [-hologramWidth / 4, 0, -0.2; 
              hologramWidth / 4, 0, -0.2;];
end;

% Window to limit the contribution area (avoid aliasing effect)
windowFunction = true; % true

% Save reconstructed images in jpg format
img_jpg = false; % false

%
% Hologram reconstruction parameters
%

% Location of the reconstructed image in the z-axis
targetZ = -0.2; % -0.2m

%
% Computation
%

if (option == 1)    % Create only an holographic image

  fprintf('---------------------------------------\n');
  fprintf('Hologram calculation...\n'); 

  % Hologram calculation
  [hologram_out, referenceWave_out, x_out, y_out] = digitalHologramGeneration(lambda, hologramHeight, ...
                                      hologramWidth, hologramZ, samplingDistance, windowFunction, ...
                                      points);
  
  fprintf('Hologram calculated!\n');
  %hologram_out = real(hologram_out);
  
  % Save
%   save('output/referenceWave_out.mat', 'referenceWave_out', '-v7');
%   save('output/hologram_out.mat', 'hologram_out', '-v7');
  save('output/main/referenceWave_out.mat', 'referenceWave_out');
  save('output/main/hologram_out.mat', 'hologram_out');
  
  % Display the hologram
  titlePlot = 'Hologram';
  xName = 'x [mm]';
  yName = 'y [mm]';
  fileName = 'hologram_out';
  plotImage(real(hologram_out), x_out, y_out, img_jpg, titlePlot, xName, yName, fileName);  
  
  fprintf('---------------------------------------\n'); 

  if (reconstructionChoice == true)
    
    fprintf('Hologram reconstruction...\n');
    
    % Hologram reconstruction
    [reconstruction_out] = digitalHologramReconstruction(lambda, hologramHeight, ...
                           hologramWidth, hologramZ, samplingDistance, targetZ, ...
                           real(hologram_out), referenceWave_out);
                           
    fprintf('Hologram reconstructed!\n');
    
    % Display the reconstructed image
    titlePlot = 'Reconstructed image (intensity)';
    xName = 'x [mm]';
    yName = 'y [mm]';
    fileName = 'reconstruction_out';
    figure()
    plotImage(abs(reconstruction_out), x_out, y_out, img_jpg, titlePlot, xName, yName, fileName);  

    fprintf('---------------------------------------\n');
    
  end;
  
elseif (option == 2)    % Create an holographic video
  
  % Number of frames
  nFrames = 3;
  
  % All points
  points3D = [0, 0, -0.1;
              0, 0, -0.2;
              0, 0, -0.3;
              ];
  
  % Number of rows (y) and columns (x) of the hologram
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);
  
  hologram_video = zeros(hologramSamplesX, hologramSamplesY, nFrames);
  
  fprintf('---------------------------------------\n');
  
  for frame = 1:nFrames    
    
    fprintf('Hologram calculation...\n');
    
    [hologram_out, referenceWave_out, x_out, y_out] = digitalHologramGeneration(lambda, hologramHeight, ...
                                hologramWidth, hologramZ, samplingDistance, windowFunction, points3D(frame, :));
                                
    hologram_video(:,:,frame) = real(hologram_out);
    
    fprintf('Hologram calculated!\n');
    
    % Save
%     save('output/hologram_video.mat', 'hologram_video', '-v7');
    save('output/main/hologram_video.mat', 'hologram_video');
    
    % Display the hologram
    figure(frame)
    titlePlot = 'Hologram';
    xName = 'x [mm]';
    yName = 'y [mm]';
    fileName = 'hologram_out';
    plotImage(hologram_video(:,:,frame), x_out, y_out, img_jpg, titlePlot, xName, yName, fileName);
  
  end;
  
  fprintf('---------------------------------------\n');

elseif (option == 3)

  % Number of holograms to be create
  nbSamples = 10;
  
  % Number of rows (y) and columns (x) of the hologram
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);
  
  hologram_dataset = zeros(hologramSamplesX, hologramSamplesY, nbSamples);
  
  % Different combinations of interferences generated by 2 particles
  allPoints3D = [ -hologramWidth / 4, 0, -0.2;
                   hologramWidth / 4, 0, -0.2;
                   
                  -hologramWidth / 4, hologramWidth / 4, -0.2;
                   hologramWidth / 4, hologramWidth / 4, -0.2;
                   
                  -hologramWidth / 4, -hologramWidth / 4, -0.2;
                   hologramWidth / 4, -hologramWidth / 4, -0.2
                   
                   -hologramWidth / 4, hologramWidth / 4, -0.2;
                   hologramWidth / 4, -hologramWidth / 4, -0.2
                   
                   -hologramWidth / 4, -hologramWidth / 4, -0.2;
                   hologramWidth / 4, hologramWidth / 4, -0.2
                   
                   -hologramWidth / 4, 0, -0.2;
                   hologramWidth / 4, hologramWidth / 4, -0.2
                   
                   -hologramWidth / 4, -hologramWidth / 4, -0.2;
                   hologramWidth / 4, 0, -0.2;
                   
                   -hologramWidth / 4, hologramWidth / 4, -0.2;
                   hologramWidth / 4, 0, -0.2
                   
                   -hologramWidth / 4, 0, -0.2;
                   hologramWidth / 4, -hologramWidth / 4, -0.2;
                   
                   -hologramWidth / 4, 0, -0.1;
                   hologramWidth / 4, 0, -0.1;
  ];
  
  points3D = zeros(2, 3);
  
  for i = 1:nbSamples
    
    % Only two sources at time
    points3D(1, :) = [allPoints3D(2*i - 1, :)];
    points3D(2, :) = [allPoints3D(2*i, :)];
    
    % Hologram computation
    [hologram_out, referenceWave_out, x_out, y_out] = digitalHologramGeneration(lambda, hologramHeight, ...
                                hologramWidth, hologramZ, samplingDistance, windowFunction, points3D);
                                
    hologram_dataset(:,:,i) = hologram_out;
    
    fileName = ['output/main/hologram_' num2str(i) '.mat'];
    
    % Save
    save(fileName, 'hologram_out');
    
    % Display the hologram
% ##    figure(i)
% ##    plotImage(hologram_dataset(:,:,i), x_out, y_out, img_jpg);
    
  end;
  
  % Save
  fileName = ['output/main/hologram_dataset_' num2str(nbSamples) '_samples_2_sources.mat'];
  save(fileName, 'hologram_dataset');
  
end;

% Read elapsed time from stopwatch
toc;

fprintf('---------------------------------------\n')
  
diary off;