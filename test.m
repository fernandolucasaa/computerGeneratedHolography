% ondelette.m
% PROGRAMME DE TRANSFORMEE EN ONDELETTES
% Le signal a analyser doit s’appeler y

ii = sqrt(-1)
%..y=signal a analyser (en m´emoire)
t1 = 0:0.01:10;
y = cos(20*pi*t1);
n = length(y) - 1; 
p = fix(log(n)/log(2)); 
dx = 1/n; 
t = 0:dx:1;
x = -1:dx:1;

% calcul de l’ondelette
a0 = 1/2;
a0 = a0^(1/2);
p = 2*p;

for i=p-1:-1:0
  a = a0^ i; % ´echelle
  xx = x/a;
  g = exp(-xx.^ 2/2).*exp(ii*5*xx)./sqrt(a); % ondelette de morlet
  %ou g=(1-2*xx.^ 2).*exp(-2*xx.^ 2)./sqrt(a); % chapeau mexicain
  % calcul des coefficients d’ondelettes a l’echelle a
  W = zeros(p,n+1);
  wa = conv(y,g);
  W(i+1,1:n) = abs(wa(n+1:2*n));
end

% trace
subplot(3,1,1); plot(t(1:n),y(1:n)); title('fonction a analyser')
subplot(3,1,2); imagesc(W(1:p,1:n)); title('coefficients dondelettes')