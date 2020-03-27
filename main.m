close all
clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Generation numerique d'hologramme et leur reconstructon
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plan de l'hologramme
hologramHeight = 2e-3; % 2mm
hologramWidth = 2e-3; % 2mm

% poins de la scene 3D
% 1 : un point source
% 2 : trois points sources
% 3 : quatre points sources
%
%     |           |           |
%     |           | *       * | *
% - - * - -   - - * - -   - - - - -  
%     |         * |         * | *
%     |           |           |
%
pointsChoice = 3; % 1

% localisation des points dans l'aixe z
pointsZ = -0.2; % -0.2m

% fenetre pour limiter la zone de contribution (eviter le repliement du spectre)
windowFunction = true; % true

% calcul d'hologramme
hologram_out = digitalHologramGeneration(hologramHeight, hologramWidth, ...
               pointsChoice, pointsZ, windowFunction);

% reoonstruction d'hologramme
%reconstruction_out = digitalHologramReconstruction()