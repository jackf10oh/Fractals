# Fractals
A repository for using CUDA and MPI to calculate fractals in C++.

Contains various scripts to create fractals. E.g. Mandelbrot, Julia, etc.
The goal is to use to speed up computations a lot on Bridges-2 supercomputer.

- Fractal.hpp
  
It is a template class for use with complexFunc_t template argments to define custom recursive functions.
It has constructors that take the initial coords of the image and the number of rows and columns
and how many iterations to perform at each point.

- class Mandelbrot
  
Template instantiation of Fractal. uses f(zn+1) = (zn)**2 + c 
where c=z0 is a point in the complex plane

- class Julia
  
Template instantiation of Fractal. uses f(z) = (|Re(z)| + |Im(z)|) ** 2 + c  
like Mandelbrot set, but c is set constant and used for every point z 
in the complex plane.

- class BurningShip
  
Template instantiation of Fractal. uses f(z) = (|Re(z)| + |Im(z)|) ** 2 + c  
like Mandelbrot but uses absolute values. Commonly viewed upside.





# sources...

- for makefile

https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable

- for Mandelbrot set coordinate
  
https://commons.wikimedia.org/wiki/User:Wolfgangbeyer
https://www.dhushara.com/DarkHeart/DarkHeart.htm
https://hypertextbook.com/chaos/mandelbrot/

- for function pointers in cuda
  
https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/
