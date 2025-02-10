#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include<iostream>
#include<string>
#include<fstream>
#include<tuple>
#include<vector>
#include<math.h>
#include<sstream>
#include <opencv2/opencv.hpp>
#include<opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>  // For imwrite function


using namespace std;

struct Complex
{
  public:
  double real;
  double im;
  CUDA_HOST_DEVICE Complex():real(0.0), im(0.0){};
  CUDA_HOST_DEVICE Complex(double a, double b): real(a), im(b){};
  CUDA_HOST_DEVICE Complex(int a, int b): real(double(a)), im(double(b)){}
  // Complex(const Complex& source):real(source.real), im(source.im){}; // copy constructor?
};

// Since C++ 11
using complexFunc_t = Complex (*) (Complex, Complex);
// using realFunc_t = double (*) (double, double);

template<complexFunc_t F>
class Fractal
{
  private:
    int nRows,nCols,num_iters; // integers number of rows and cols in m_array, numerber of iterations to calc recursion
    double radius=2;
    double x1,y1,x2,y2;
    vector<vector<Complex>> val_array; // array containing each pixel's value
    vector<vector<int>> iter_array; // array that counts how many iterations for a pixel to explode

  public:
    //constructors
    Fractal(); //default
    Fractal(Complex point1, Complex point2, tuple<int,int> dims, int num_iters_init); // from dims + n_iters
    Fractal(const Fractal &source); // copy 

    //destructors
    virtual ~Fractal(){}; // default

    //setters
    void SetIterArr(vector<vector<int>> source_array); // set iter_array from another array
    void Calculate(bool verbose=0); // serially loop through matrix entries
    void CalculateCuda(bool verbose=0); // same as calculate. uses cuda
    void Center(Complex p1); // move the mandelbrot set to be centered over a point p1
    void Zoom(double scale=0.95); // reset the coors (x1,y1), (x2,y2) to be smaller box around center

    //getters
    vector<vector<Complex>> GetValArr(); // get iter_array from member data.
    vector<vector<int>> GetIterArr(); // get iter_array from member data.
    int Rows(){return nRows;};
    int Cols(){return nCols;};
    void ValToCSV(string fname="values.csv"); // save value array to csv file
    void ItersToCSV(string fname="result.csv"); // write iters_array to a csv file
    void ItersToIMG(string fname="result.ppm"); // write iters_array to a ppm file
    cv::Mat ItersToFrame(); // write to a opencv cv::Mat

    //operators
    Fractal& operator = (const Fractal& source);

};

__host__ __device__ static Complex mandelbrot_func(Complex c_init, Complex z)
{
  double real = z.real*z.real - z.im*z.im + c_init.real;
  double im = 2*z.real*z.im + c_init.im;
  return Complex(real,im);
};

__host__ __device__ const complexFunc_t p_mandelbrot_func = mandelbrot_func;

__host__ __device__ static Complex julia_heart_func(Complex c_init, Complex z)
{
  c_init = Complex(-0.8, 0.156);
  double real = z.real*z.real - z.im*z.im + c_init.real;
  double im = 2*z.real*z.im + c_init.im;
  return Complex(real,im);
};

__host__ __device__ const complexFunc_t p_julia_heart_func = julia_heart_func;


#include "Fractals.cu"

// template instantiation
template class Fractal<p_mandelbrot_func>; 
template class Fractal<p_julia_func>;
template class Fractal<p_burning_ship_func>;

typedef Fractal<p_mandelbrot_func> Mandelbrot;

typedef Fractal<p_julia_func> Julia;

typedef Fractal<p_burning_ship_func> BurningShip;

#endif

