#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__
#else
    #define CUDA_HOST_DEVICE
    #define CUDA_HOST
    #define CUDA_DEVICE
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

void checkLastError();

using namespace std;

CUDA_HOST inline int sdiv(int a, int b){return (a+b-1)/b;};

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

// double modulus2(const tuple<double,double> z); // returns modulus squared
CUDA_HOST_DEVICE inline double modulus2(Complex z){return (z.real*z.real + z.im*z.im);}; // returns modulus squared

// Since C++ 11
using complexFunc_t = Complex (*) (Complex, Complex);
// using realFunc_t = double (*) (double, double);

template<complexFunc_t F>
class Fractal
{
  private:
    // integers number of rows and cols in arrays, numerber of iterations to calc recursion
    int nRows,nCols,num_iters; 
    // radius where the iteration formula explodes
    double radius=2;
    // coords for top left and bottom right of frame
    double x1,y1,x2,y2;
    // array containing each point's value
    vector<vector<Complex>> val_array; 
    // counts how many iterations for a point to explode. 0 means stable.
    vector<vector<int>> iter_array; 

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
    void CalculateCudaStreams(bool verbose=0); // same as CalculateCude, but uses streams for copy/compute overlap.
    void CalculateCudaGPUs(bool verbose=0); // same as CalculateCudaStreams, but use all GPUs available.
    void Center(Complex p1); // move the mandelbrot set to be centered over a point p1
    void Zoom(double scale=0.95); // reset the coors (x1,y1), (x2,y2) to be smaller box around center

    //getters
    vector<vector<Complex>> GetValArr() const; // get iter_array from member data.
    vector<vector<int>> GetIterArr() const; // get iter_array from member data.
    int Rows() const {return nRows;};
    int Cols() const {return nCols;};
    void ValToCSV(string fname="values.csv") const; // save value array to csv file
    void ItersToCSV(string fname="result.csv") const; // write iters_array to a csv file
    void ItersToIMG(string fname="result.ppm") const; // write iters_array to a ppm file
    cv::Mat ItersToFrame() const; // write to a opencv cv::Mat

    //operators
    Fractal& operator = (const Fractal& source);

};

CUDA_HOST_DEVICE static Complex mandelbrot_func(Complex c_init, Complex z)
{
  double real = z.real*z.real - z.im*z.im + c_init.real;
  double im = 2*z.real*z.im + c_init.im;
  return Complex(real,im);
};

CUDA_HOST_DEVICE const complexFunc_t p_mandelbrot_func = mandelbrot_func;

CUDA_HOST_DEVICE static Complex julia_func(Complex c_init, Complex z)
{
  c_init = Complex(-0.78,0.136);
  double real = z.real*z.real - z.im*z.im + c_init.real;
  double im = 2*z.real*z.im + c_init.im;
  return Complex(real,im);
};

CUDA_HOST_DEVICE const complexFunc_t p_julia_func = julia_func;

CUDA_HOST_DEVICE static Complex burning_ship_func(Complex c_init, Complex z)
{
  double real = fabs(z.real*z.real) - fabs(z.im*z.im) + c_init.real;
  double im = 2*fabs(z.real)*fabs(z.im) + c_init.im;
  return Complex(real,im);
};

CUDA_HOST_DEVICE const complexFunc_t p_burning_ship_func = burning_ship_func;

#include "Fractals.cu"

// template instantiation
template class Fractal<p_mandelbrot_func>; 
template class Fractal<p_julia_func>;
template class Fractal<p_burning_ship_func>;

typedef Fractal<p_mandelbrot_func> Mandelbrot;

typedef Fractal<p_julia_func> Julia;

typedef Fractal<p_burning_ship_func> BurningShip;

#endif


