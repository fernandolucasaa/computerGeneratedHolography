
function plotImage(variable, x, y, img_jpg, titlePlot, xName, yName, fileName)
  
  % Afficher l'hologramme ou l'image reconstituee
  colormap('gray');
  imagesc(x * 1e3, y * 1e3, variable);
  set(gca, 'YDir', 'normal'); % inverser la direction de l'axe y
  colorbar;
  title(titlePlot);
  xlabel(xName);
  ylabel(yName);
  axis('equal');
  #axis('image'); #pour l'image reconstituee
  
  savefig(['output/' fileName]);
  
  if (img_jpg == true)
    fig = openfig(['output/' fileName '.fig']);
    saveas(fig, ['output/' fileName '.jpg']);
  end;

endfunction
