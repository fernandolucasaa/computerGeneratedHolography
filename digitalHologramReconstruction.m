
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reconstruction numerique de l'image a partir de l'hologramme.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [reconstruction_out] = digitalHologramReconstruction(lambda, hologramHeight, ...
                                hologramWidth, hologramZ, samplingDistance, targetZ, ...
                                hologram, referenceWave, img_jpg)
  
  fprintf('[Hologram reconstruction]\n');

  
  %
  % Parametres de reconstruction des hologrammes
  %
  
  % nombre d'onde
  k = 2*pi/lambda;
  
  % dimensions de l'image reconstituee
  targetWidth = hologramWidth;
  targetHeight = hologramWidth;
  
  % nombre de lignes (y) et de colonnes (x) du plan d'hologramme et d'image reconstituee
  hologramSamplesX = ceil(hologramWidth / samplingDistance);
  hologramSamplesY = ceil(hologramHeight / samplingDistance);

  targetSamplesX = ceil(targetWidth / samplingDistance);
  targetSamplesY = ceil(targetHeight / samplingDistance);
  
  % emplacement du coin "inférieur gauche" de plans (coin de reference)
  % mettre le centre du plan a x = 0, y = 0
  hologramCornerX = - (hologramSamplesX - 1) * samplingDistance / 2;
  hologramCornerY = - (hologramSamplesY - 1) * samplingDistance / 2;
  
  targetCornerX = - (targetSamplesX - 1) * samplingDistance / 2;
  targetCornerY = - (targetSamplesY - 1) * samplingDistance / 2;
  
  
  %% Propagation numerique de l'hologramme %%
    
  % Calcul du noyau de propagation %
  
  fprintf('\nNumerical propagation of the hologram\n');
  fprintf('The propagation kernel calculation...\n');
  
  % La propagation par convolution est calculée à l'aide de matrices de taille
  % (cY lignes cY) x (cX colonnes)
  cX = hologramSamplesX + targetSamplesX - 1; 
  cY = hologramSamplesY + targetSamplesY - 1; 
  
  % Déplacement entre les coins de l'hologramme et la cible
  px = targetCornerX - hologramCornerX;
  py = targetCornerY - hologramCornerY;
  z0 = targetZ - hologramZ; % propagation dans l'aixe z
  
  fprintf('\nDistance between the hologram plan and the target plan : %d m\n', z0);
  
  % Coordonnees auxiliaires x, y pour le calcul de la convolution
  auxCX = cX - hologramSamplesX + 1;
  auxCY = cY - hologramSamplesY + 1;
  auxX = (1-hologramSamplesX: auxCX-1) * samplingDistance + px;
  auxY = (1-hologramSamplesY: auxCY-1) * samplingDistance + py;
  [auxXX, auxYY] = meshgrid(auxX, auxY);
  
  % Calculer le noyau de propagation de Rayleigh-Sommerefeld
  r2 = auxXX.^2 + auxYY.^2 + z0^2;
  r = sqrt(r2);
  kernel = -1/(2*pi) * (1i*k - 1./r)*z0.*exp(1i*k*r) ./ r2 * samplingDistance^2;
  
 
  %% Le calcul de la reconstruction %%
  
  fprintf('\nThe reconstruction calculation...\n');
  
  % Creer la matrice auxiliaire de la bonne taille pour le calcul de la convolution
  % La "taille correcte" signifie que la nature cyclique de la sera supprimée.
  auxMatrix = zeros(cY, cX);
  
  % Placez un hologramme eclaire par l'onde de reference sur la matrice auxiliaire. 
  % Le reste des echantillons est a 0. Cette étape est appelée "rembourrage zero".
  auxMatrix(1:hologramSamplesY, 1:hologramSamplesX) = hologram .* conj(referenceWave);
  
  % Le noyau doit etre plie de maniere a ce que l'entree pour x = 0, y = 0 soit 
  % dans la premiere ligne, premiere colonne de la matrice du noyau
  kernel = circularShift(kernel, hologramSamplesY - cX, hologramSamplesX - cX);
  
  % Calculer la convolution cyclique à l'aide de la FFT
  auxMatrixFT = fft2(auxMatrix) .* fft2(kernel);
  auxMatrix = ifft2(auxMatrixFT);
  
  % Choisissez les valeurs qui ne sont pas endommagees par la nature cyclique
  % de la convolution FFT.
  reconstruction = auxMatrix(1:hologramSamplesY, 1:hologramSamplesX);
  
  reconstruction_out = reconstruction;
  
  xAxis = (0:hologramSamplesX - 1)*samplingDistance + hologramCornerX;
  yAxis = (0:hologramSamplesY - 1)*samplingDistance + hologramCornerY;
  
  % Afficher l'image reconstituee
  figure();
  colormap('gray');
  imagesc(xAxis * 1e3, yAxis * 1e3, abs(reconstruction));
  set(gca, 'YDir', 'normal');
  colorbar;
  title('Reconstructed image (intensity)');
  xlabel('x [mm]');
  ylabel('y [mm]');
  axis('image');
  
  savefig('images/reconstructed_image');
  
  if (img_jpg == true)
    fig = openfig('images/reconstructed_image.fig');
    saveas(fig, 'images/reconstructed_image.jpg');
  end;
  
  fprintf('The reconstruction calculated!\n\n');
  
end;
