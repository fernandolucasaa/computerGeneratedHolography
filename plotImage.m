
%
% Display an hologram or a reconstructed image
% 

function plotImage(variable, x, y, img_jpg, titlePlot, xName, yName, fileName)
  
  colormap('gray');
  imagesc(x * 1e3, y * 1e3, variable);
  set(gca, 'YDir', 'normal'); % reverse the direction of the y-axis
  colorbar;
  title(titlePlot);
  xlabel(xName);
  ylabel(yName);
  axis('equal');
%   #axis('image'); # for the reconstructed image
  
  savefig(['output/main/' fileName]);
  
  if (img_jpg == true)
    fig = openfig(['output/main/' fileName '.fig']);
    saveas(fig, ['output/main/' fileName '.jpg']);
  end;

end
