
import numpy
import math
from pylab import *

            
def matriceImage(matrice,gamma,rgb):
    s = matrice.shape
    a=1.0/gamma;
    norm=matrice.max()
    m = numpy.power(matrice/norm,a)
    im = numpy.zeros((s[0],s[1],3),dtype=float64)
    im[:,:,0] = rgb[0]*m
    im[:,:,1] = rgb[1]*m
    im[:,:,2] = rgb[2]*m
    return im
            
def matriceImageLog(matrice,rgb):
    s = matrice.shape
    m = numpy.log10(1+matrice)
    min = m.min()
    max = m.max()
    m = (m-min)/(max-min)
    im = numpy.zeros((s[0],s[1],3),dtype=float64)
    im[:,:,0] = rgb[0]*m
    im[:,:,1] = rgb[1]*m
    im[:,:,2] = rgb[2]*m
    return im
             
def plotSpectre(image,Lx,Ly):
    (Ny,Nx,p) = image.shape
    fxm = Nx*1.0/(2*Lx)
    fym = Ny*1.0/(2*Ly)
    imshow(image,extent=[-fxm,fxm,-fym,fym])
    xlabel("fx")
    ylabel("fy")
            
def matriceFiltre(matrice,transfert,p):
    s = matrice.shape
    Nx = s[1]
    Ny = s[0]
    nx = Nx/2
    ny = Ny/2
    Mat = zeros((Ny,Nx),dtype=numpy.complex)
    for n in range(Nx):
        for l in range(Ny):
            fx = float(n-nx-1)/Nx
            fy = float(l-ny-1)/Ny
            Mat[l,n] = matrice[l,n]*transfert(fx,fy,p)
    return Mat
            