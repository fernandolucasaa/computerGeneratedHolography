close;
clc;

img = imread('image_test.png');
gray = rgb2gray(img);

colormap('gray')
imagesc(gray)

N = 1;
theta = 0;
units = 'radian';
W = localwigner(gray, N, theta, units, 'periodic', 'square');