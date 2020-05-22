import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fftpack import fft, ifft
import math

def product_function(sequence):
    """
    Starting from a given sequence, this function calculates the product 
    function of the 1D pseudo-Wigner distribution of such sequence.
    """
    product = np.ones(len(sequence))
    sequence[len(sequence)-1]=sequence[0]
    for i in range(len(sequence)-1):
        product[i]= sequence[i]*sequence[len(sequence)-1-i]
    product = np.delete(product,[len(sequence)-1])
    return product

def local_wigner_function(sequence):
    """
    Starting from a given sequence, this function calculates the pseudo-Wigner
    distribution of such sequence
    """
    product =product_function(sequence)
    wigner = fft(product)
    wigner = np.real(wigner)
    H = wigner[int(len(wigner)/2):len(wigner)]
    T = wigner[0:int(len(wigner)/2)] 
    wigner = np.append(H,T)
    return wigner

def oriented_pattern(seq_length,angle):
    """
    This function originates a pattern that is later used for the orientation of the operational pseudo-Wigner distribution
    computation window
    seq_length: sequence length in pixels, angle: orientation in degrees
    """
    angle = np.mod(angle,180)
    # correction to set axes in the image (x: rows, y: columns) to observer cartesian coordinates x,y
    angle = np.mod((angle+90),180)
    angle =math.radians(angle)
    pi = math.pi
    h = int((seq_length/2))
    values = np.r_[float(-h):float(h+1)]
    new_positions = np.zeros([2*h+1, 2])
    for position in range(seq_length):
        if angle >= 0 and angle < pi/4:
            new_positions[position,0] = values[position]+h
            new_positions[position,1] = values[position]*math.tan(angle)+h
        elif angle >= pi/4 and angle < pi/2:
            new_positions[position,0] = values[position]*math.tan(pi/2-angle)+h
            new_positions[position,1] = values[position]+h
        elif angle >= pi/2 and angle < 3*pi/4:
            new_positions[position,0] = values[position]*math.tan(pi/2-angle)+h
            new_positions[position,1] = values[position]+h
        elif angle >= 3*pi/4 and angle <= pi:
            new_positions[position,0] = 1*values[position]+h
            new_positions[position,1] = values[position]*math.tan(angle)+h
        new_points = np.round_(new_positions)
    return new_points.astype(int)

def wigner_distribution(test_image,seq_length,angle): # seq_length must be an odd number
    """
    This application calculates the  1D pseudo-Wigner distribution of test_image image (in gray levels) , seq_length
    is the length in pixels of the operating window and it has to be an odd number (9 is a common operative value). 
    The angle variable in degrees determines the spatial orientation of the distribution.
    """
    print("calculating ...")
    # change test image to float
    test_image = np.float64(test_image)
    # determine image shape
    rows = test_image.shape[0]
    columns = test_image.shape[1]
    # determine h pixels to frame the image
    h = int((seq_length/2))
    # determine framing background image
    frame = np.ones([rows+2*h,columns+2*h])
    # insert image into the frame
    frame[h:rows+h,h:columns+h] = test_image      
    # initial wigner distribution of test image        
    distribution = np.ones([2*h,rows,columns])
    # calculations
    for row in range(h,rows+h):
        if np.mod(row,100) == 0:
            print("calculating in row ",row," ...")
            
        for column in range(h,columns+h):
            working_frame = frame[row-h:row+h+1,column-h:column+h+1]
            local_copy = working_frame.copy()
            indices = oriented_pattern(seq_length,angle)
            sequence = np.zeros(seq_length)
            for k in range(seq_length):
                sequence[k] = local_copy[indices[k,0],indices[k,1]] 
            wigner = local_wigner_function(sequence)
            distribution[:,row-h,column-h] = wigner
    return distribution

def show_wigner_frequencies(distribution):
    """
    Starting from the pseudo-Wigner distribution (distribution) of the input test image, this function gives a
    visualization of the n frequency components of such distribution and images are saved in pdf's
    """
    rows = distribution.shape[1]
    columns = distribution.shape[2]
    layers = distribution.shape[0]
    frequencies = np.zeros([layers,rows,columns])
    for layer in range(layers):
        frequency = distribution[layer,:,:]
        min_val =np.amin(frequency)
        frequency = frequency - min_val
        max_val = np.amax(frequency)
        frequency = (1/max_val)*frequency
        plt.figure()
        frequency = np.uint8(255*frequency)
        #plt.imshow(frequency, interpolation='nearest',cmap='gray')
        plt.imshow(frequency,cmap='gray')
        name = "wigner_distribution_" + str(layer) + ".pdf"
        msg = "Wigner distribution, frequency #" + str(layer)
        plt.xlabel(msg)
        #plt.savefig(name)
        frequencies[layer,:,:]= frequency
    return frequencies 