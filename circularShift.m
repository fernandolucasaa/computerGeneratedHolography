% 2D circular shift in a matrix
%
% Although Octave/MATLAB implements a function circshift that does exactly
% that, Scilab does not have it. 
%
% Inputs:
% m
%  - 2D matrix
% rowShift, colShift
%  - circular shift in rows and columns
%  - row 1 becomes row '1+rowShift'
%  - column 1 becomes column '1+colShift' 
%
% Outputs:
% modified matrix out
function out = circularShift(m, rowShift, colShift)
  rows = size(m, 1);
  cols = size(m, 2);
  rowShift = mod(rowShift, rows); %% SCILAB %%   rowShift = pmodulo(rowShift, rows);
  colShift = mod(colShift, cols); %% SCILAB %%   colShift = pmodulo(colShift, cols);
  
  % allocate output matrix
  out = zeros(rows, cols);
  
  % split row indices to high and low parts in both source and destination matrices
  dstRL = 1:rowShift;
  dstRH = (1 + rowShift):rows;
  srcRL = 1:(rows-rowShift);
  srcRH = (rows - rowShift + 1):rows;

  % split column indices to high and low parts in both source and destination matrices
  dstCL = 1:colShift;
  dstCH = (1 + colShift):cols;
  srcCL = 1:(cols-colShift);
  srcCH = (cols - colShift + 1):cols;

  % fold the matrix: swap low and high parts
  out(dstRH, dstCH) = m(srcRL, srcCL);
  out(dstRL, dstCH) = m(srcRH, srcCL);
  out(dstRH, dstCL) = m(srcRL, srcCH);
  out(dstRL, dstCL) = m(srcRH, srcCH);
end