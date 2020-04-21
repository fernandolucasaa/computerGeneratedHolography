
function plotImage(hologram, x, y, img_jpg)
  
  % Afficher l'hologramme
  colormap('gray');
  imagesc(x * 1e3, y * 1e3, hologram);
  set(gca, 'YDir', 'normal'); % inverser la direction de l'axe y
  colorbar;
  title('Hologram');
  xlabel('x [mm]');
  ylabel('y [mm]');
  axis('equal');
  
  savefig('output/hologram');
  
  if (img_jpg == true)
    fig = openfig('output/hologram.fig');
    saveas(fig, 'output/hologram.jpg');
  end;
  
endfunction
