## Kernel

`computeKernel` : compute kernel matrix  
`multiplyKernel`: multiply kernel matrix by computing elements of kernel matrix on the fly  
`dqKernel`      : compute the first derivative of kernel matrix with respect to nodes  
`d2qKernel`     : compute the second derivative of kernel matrix with respect to nodes  

See runTest.m in each folder for the usage in MATLAB.  
The files \*ITF.\* are interfaces between C++/CUDA and MATLAB.
