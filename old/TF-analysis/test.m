addpath('C:/Users/ferna/Desktop/computerGeneratedHolography/tftb-0.2/mfiles')

% generate signal
N = 128;
sig = fmlin(N,0,0.5);
t = 1:N;

% make plot
figure
plot(t,real(sig)); 
axis([t(1) t(N) -1 1]);
xlabel('Time'); 
ylabel('Real part');
title('Linear frequency modulation'); 
grid;

% compute spectrum
dsp = fftshift(abs(fft(sig)).^2);
f = (-N/2:N/2-1)/N;

% make frequency plot
figure
plot(f,dsp);
xlabel('Normalized frequency'); ylabel('Squared modulus');
title('Spectrum'); grid

% compute time-frequency representation
tfr = tfrwv(sig);

%% make plot
colormap(jet);
fs = 1;
imagesc(t/fs,fs*f,flipud(tfr));


%%%%%%%%% testing %%%%%%%%%%%%%%%
samplingDistance = 10e-6;
hologramHeight = 2e-3; % 2mm
hologramWidth = 2e-3; % 2mm
hologramSamplesX = ceil(hologramWidth / samplingDistance);
hologramSamplesY = ceil(hologramHeight / samplingDistance);
hologramCornerX = - (hologramSamplesX - 1) * samplingDistance / 2;
hologramCornerY = - (hologramSamplesY - 1) * samplingDistance / 2;
x = (0:(hologramSamplesX-1)) * samplingDistance + hologramCornerX;
y = (0:(hologramSamplesY-1)) * samplingDistance + hologramCornerY;
figure
imagesc(x * 1e3, y * 1e3, hologram_out);

##% compute time-frequency representation
tfr2 = tfrwv(hologram_out(:,:));
##
##%% make plot
##colormap(jet);
##fs = 1;
##figure
imagesc(tfr2);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% note: matlab users can use a GUI to adjust the layout
%% and other parameters of the display. this utility is
%% not yet available with GNU octave.