# Fractals
A repository for various C++ files using CUDA and MPI to calculate fractals.

Fractals folder

contains various scripts to create fractals. E.g. mandelbrot, burning ship, etc
goal is to use mpi, cuda, openmp, etc to speed up computations a lot.

- Fractal.hpp  
it is an abstract base class for future classes to define custom recursive functions
it has constructors that take the initial coors of the image and the number of rows and columns

- Mandelbrot.hpp
derived from Fractal abc. uses f(z) = z**2 + c 

- BurningShip.hpp
derived from Fractal abc. uses f(z) = (|Re(z)| + |Im(z)|) ** 2 + c  
