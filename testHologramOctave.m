% Parameters
lambda = 532e-9; % wavelength
hologramHeight = 2e-3; % 2mm
hologramWidth = 2e-3; % 2mm
hologramZ = 0;
Delta = 10e-6; % 10?m
samplesX = hologramWidth / Delta; % 200 samples with sampling distance delta
samplesY = hologramHeight / Delta; % 200 samples with sampling distance delta
cornerX = -hologramWidth / 2;
cornerY = -hologramHeight / 2;
%points = [ 0, 0, -0.2;
%  -hologramWidth/4,-hologramHeight/4, -0.2;
%  hologramWidth/4, hologramHeight/4, -0.22];
points = [0, 0, -0.2];

% Object wave calculation
k = 2*pi/lambda;
objectWave = zeros(samplesY, samplesX);
for s = 1:rows(points) %all the poins in the scene
    for column = 1:samplesX
        for row = 1:samplesY
        x = (column-1) * Delta + cornerX;
        y = (row-1) * Delta + cornerY;
        r = sqrt((x - points(s, 1))^2 + (y - points(s, 2))^2 + (hologramZ - points(s, 3))^2);
        objectWave(row,column) += exp(1i*k*r) / r;
        %objectWave(row,column) = exp(1i*k*r);
        end
    end
end

% Reference wave calculation
alpha = 90 * pi/180;
beta = 90.5 * pi/180;
nX = cos(alpha); nY = cos(beta);
nZ = sqrt(1 - nX^2 - nY^2);
refAmplitude = max(max(abs(objectWave)));
referenceWave = zeros(samplesY, samplesX);
for column = 1:samplesX
  for row = 1:samplesY
  x = (column-1) * Delta + cornerX;
  y = (row-1) * Delta + cornerY;
  referenceWave(row,column) = refAmplitude * ...
  exp(1i*k*(x*nX + y*nY + hologramZ*nZ));
  end
end

% Hologram calculation
optField = objectWave + referenceWave;
hologram = optField .* conj(optField);

plot(hologram)
