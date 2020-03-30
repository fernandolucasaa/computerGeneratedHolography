
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Generation numerique d'hologramme a partir de quelques sources ponctuelles en
% sommant leurs contributions (ondes spheriques). Les occultations ne sont pas 
% prises en compte pour l'instant !
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hologram_output, referenceWave_output] = digitalHologramGeneration(lambda, ...
          hologramHeight, hologramWidth, hologramZ, samplingDistance, pointsChoice, ...
          pointsZ, windowFunction, img_jpg)

  fprintf('[Hologram generation]\n\n');
  
  %% [0] Parametres initiaux - Initialisation %%
  
  % coleur des images affichés
  colormap('gray')
  
  %
  % Parametres du plan l'hologramme
  %
     
  fprintf('Dimensions of the hologram: %d m vs %d m\n', hologramHeight, hologramWidth);
  
  % nombre de lignes (y) et de colonnes (x) du plan d'hologramme
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);
  
  fprintf('Resolution of the hologram: %d pixels vs %d pixels\n', hologramSamplesX, hologramSamplesY);
  fprintf('\n');
  
  % emplacement du coin "inférieur gauche" de l'hologramme (coin de reference)
  % mettre le centre de l'hologramme a x = 0, y = 0
  hologramCornerX = - (hologramSamplesX - 1) * samplingDistance / 2;
  hologramCornerY = - (hologramSamplesY - 1) * samplingDistance / 2;
  
  % amplitude initiale
  a = 1;
  
  % nombre d'onde
  k = 2*pi/lambda;
  
  %
  % Parametres de la scene
  %
  
  % points de la scene
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
  end;
        
  fprintf('Positions of the points in the 3D scene [x,y,z]:');
  
  for source = 1:size(points, 1)
    fprintf('\nPoint light source %d of %d: [%d, %d, %d]', source, size(points, 1), ...
    points(source, 1), points(source, 2), points(source, 3));
  end
  
  % Utiliser les positions de tous les echantillons
  x = (0:(hologramSamplesX-1)) * samplingDistance + hologramCornerX;
  y = (0:(hologramSamplesY-1)) * samplingDistance + hologramCornerY;
  [xx, yy] = meshgrid(x, y);
  
  % ------------------------------------------------------------------------------
  
  %% [1] Calcul de l'onde objet (Nuage de points) %% 
  
  fprintf('\n\nThe object wave calculation...\n');
  
  if (windowFunction) 
    fprintf('Window function considered in the calculation'); 
  end;
  
  objectWave = zeros(hologramSamplesY, hologramSamplesX);
  
  % superposition de tous les ondes spheriques
  for source = 1:size(points, 1)

    % pour la retropropagation, inverser le signe de l'unité imaginaire
    if (points(source, 3) > hologramZ)
      fprintf('\nAttention! The point is in the front of the hologram plan!\n');
      ii = -1i;
    else
      ii = 1i;
    end
    
    % fonction fenetre
    h = ones(hologramSamplesX, hologramSamplesY);
    
    % Limiter la zone de contribution
    if (windowFunction)
        
        % Region de contribution du point lumineuse
        p = samplingDistance; % pas d'echantillonnage
        Rmax = abs(points(source,3) * tan(asin(lambda/(2*p))));
        distance = sqrt((xx - points(source, 1)).^2 + (yy - points(source, 2)).^2);
        indices = find(distance > Rmax);
        h(indices) = 0;
      
    end
    
    % distance oblique
    r = sqrt((xx - points(source, 1)).^2 + (yy - points(source, 2)).^2 + (hologramZ - points(source, 3)).^2);
    objectWave = objectWave + a .* exp(ii*k*r) ./ r .* h;
  
  end
  fprintf('\nThe object wave calculed!\n');
  
  
  %% Calcul de l'onde de reference %%
  
  fprintf('\nThe reference wave calculation...\n');
  
  % les angles de direction de l'onde de reference avec le axes x et y
  % vecteur d'onde perpendiculaire a l'ecran (radians)
  alpha = pi/2; % par rapport a l'aixe x
  beta = pi/2; % par rapport a l'aixe y
  
  % vecteur de direction
  nX = cos(alpha); 
  nY = cos(beta);
  nZ = sqrt(1 - nX^2 - nY^2);
  
  % autoriser nZ < 0, en cas ou
  if (nZ > 0)
    ii = 1i;
  else
    ii = -1i;
    fprintf('\nAttention! The direction vector is in the opposite direction!\n');
  end
  
  refAmplitude = max(max(abs(objectWave)));
  
  % l'onde de reference
  referenceWave = refAmplitude * exp(ii * k * (xx*nX + yy*nY + hologramZ*nZ));
  fprintf('The reference wave calculed!\n');
  
  referenceWave_output = referenceWave;
  
  % ------------------------------------------------------------------------------
  
  %% [2] Representation de l'onde objet %%
  
  fprintf('\nThe hologram calculation... \n');
  
  %% Calcul de l'hologramme (interference entre l'onde objet et l'onde de reference) %%
  
  % Modulation d'amplitude
  itensityTotal = (objectWave + referenceWave).*conj(objectWave + referenceWave);
  
  itensity = 2*real(objectWave.*conj(referenceWave));
  hologram = real(itensity);
  hologram_output = hologram;
  
  % Afficher l'hologramme
  imagesc(x * 1e3, y * 1e3, hologram);
  set(gca, 'YDir', 'normal'); % inverser la direction de l'axe y
  colorbar;
  title('Hologram');
  xlabel('x [mm]');
  ylabel('y [mm]');
  axis('equal');
  
  savefig('images/hologram');
  
  if (img_jpg == true)
    fig = openfig('images/hologram.fig');
    saveas(fig, 'images/hologram.jpg');
  end;
  
  fprintf('The hologram calculated!\n');
  
end;  
