addpath('C:/Users/ferna/Desktop/computerGeneratedHolography/tftb-0.2/mfiles')

close all;
clear all;

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
figure();
plot(f,dsp);
xlabel('Normalized frequency'); ylabel('Squared modulus');
title('Spectrum'); grid

% compute time-frequency representation
tfr = WIGNER(sig);

%%%%%% test %%%%%%%
[xrow,xcol] = size(sig);
t=1:xrow;
[trow,tcol] = size(t);



%%%%%%%%%%%%%%%%%%%%

%% make plot
colormap(jet);
fs = 1;
figure();
imagesc(t/fs,fs*f,flipud(tfr));
colorbar();

%% note: matlab users can use a GUI to adjust the layout
%% and other parameters of the display. this utility is
%% not yet available with GNU octave.