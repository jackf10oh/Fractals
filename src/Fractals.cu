#ifndef MATRIX_CU
#define MATRIX_CU

#include "Fractals.hpp"
#include<math.h>

// double modulus2(const tuple<double,double> z); // returns modulus squared
__host__ __device__ inline double modulus2(Complex z){return z.real*z.real + z.im*z.im;}; // returns modulus squared

template<complexFunc_t F>
__global__ void iteration_kernel(int* iter_array, int max_iters, double radius, Complex* val_array, int num_elems)
{
  int stride = gridDim.x * blockDim.x;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i=idx; i<num_elems; i+=stride)
  {
    iter_array[i]=0;
    Complex c_init=val_array[i], z=val_array[i]; // complex numbers c,z = a+ib
    int iter=1; // leave iter_array as 0 if recursion doesn't explode.
    do
    {
      z = (*F)(c_init,z); // assignment handled in my_recursive
      // if recursion > radius of fractal
      if(modulus2(z) >= radius*radius) 
      {
        val_array[i]=z; // copy z back to val_array
        iter_array[i]=iter; // update iter_array to nonzero interger
      }; 
      iter++;
    }
    while(modulus2(z) < radius*radius && iter<max_iters); // loop while inside radius and iterations remain
  }
};

//constructors
template<complexFunc_t F>
Fractal<F>::Fractal() //default
{
  num_iters=9999;
  nRows=1020; nCols=1980; 

  x1=-2; y1=2;
  x2=2; y2=-2;

  // allocate memory
  val_array = vector<vector<Complex>>(nRows, vector<Complex>(nCols));
  iter_array = vector<vector<int>>(nRows, vector<int>(nCols,0));

  // fill values
  for(int i=0; i<nRows; i++)
  {
    double y = y1 + (y2-y1)*(double(i)/(nRows-1));
    for(int j=0; j<nCols; j++)
    {
      double x = x1 + (x2-x1)*(double(j)/(nCols-1));
      val_array[i][j] = Complex(x,y);
    }
  }
};

template<complexFunc_t F>
Fractal<F>::Fractal(Complex point1, Complex point2, tuple<int,int> dims, int num_iters_init) // from dims + n_iters
{
  num_iters=num_iters_init;

  nRows=get<0>(dims); nCols=get<1>(dims); 

  x1=point1.real; y1=point1.im;
  x2=point2.real; y2=point2.im;

  // allocate memory
  val_array = vector<vector<Complex>>(nRows, vector<Complex>(nCols));
  iter_array = vector<vector<int>>(nRows, vector<int>(nCols,0));

  // fill values
  for(int i=0; i<nRows; i++)
  {
    double y = y1 + (y2-y1)*(double(i)/(nRows-1));
    for(int j=0; j<nCols; j++)
    {
      double x = x1 + (x2-x1)*(double(j)/(nCols-1));
      val_array[i][j] = Complex(x,y);
    }
  }
};

template<complexFunc_t F>
Fractal<F>::Fractal(const Fractal &source) // copy 
{
  num_iters=source.num_iters;
  nRows=source.nRows; nCols=source.nCols; 

  x1=source.x1; y1=source.y1;
  x2=source.x2; y2=source.y2;

  // allocate memory
  val_array=source.val_array;
  iter_array=source.iter_array;
};



// setters
template<complexFunc_t F>
void Fractal<F>::Calculate(bool verbose) // serially loop through matrix entries
{
  for(int i=0; i<nRows; i++)
  {
    for(int j=0; j<nCols; j++)
    {
      if(verbose) printf("%d ", i*nCols+j);
      iter_array[i][j]=0;
      Complex c_init=val_array[i][j], z(0,0); // complex numbers c,z = a+ib
      int iter=1; // leave iter_array as 0 if recursion doesn't explode.
      do
      {
        z = (*F)(c_init,z); // assignment handled in my_recursive
        // if recursion > radius of fractal
        if(modulus2(z) >= radius*radius) 
        {
          val_array[i][j]=z; // copy z back to val_array
          iter_array[i][j]=iter; // update iter_array to nonzero interger
        }; 
        iter++;
      }
      while(modulus2(z) < radius*radius && iter<num_iters); // loop while inside radius and iterations remain
    }
    if(verbose) printf("\n");
  }
};

template<complexFunc_t F>
void Fractal<F>::CalculateCuda(bool verbose) // iterate on each matrix entry.
{
  // host memory 
  int num_elems = nRows*nCols;
  if(verbose) cout<<"host memory"<<endl;
  Complex* val_array_h = new Complex[num_elems];
  int* iter_array_h = new int[num_elems];
  for(int i=0; i<nRows; i++) // copy val array into host buffer
  {
    for(int j=0; j<nCols; j++)
    {
      val_array_h[i*nCols+j]=val_array[i][j];
    }
  }

  // device memory 
  if(verbose) cout<<"device memory"<<endl;
  Complex* val_array_d;
  int* iter_array_d;
  cudaMalloc(&val_array_d, num_elems*sizeof(Complex));
  cudaMalloc(&iter_array_d, num_elems*sizeof(int));
  cudaMemcpy(val_array_d, val_array_h, num_elems*sizeof(Complex), cudaMemcpyHostToDevice);

  // run kernel
  if(verbose) if(verbose)cout<<"kernel running"<<endl;
  iteration_kernel<F><<<32,256>>>(iter_array_d, num_iters, radius, val_array_d, num_elems);
  cudaDeviceSynchronize();
  // checkLastError();

  //copy back to host buffer, unpack into val-array
  if(verbose) cout<<"copy back to host"<<endl;
  cudaMemcpy(val_array_h, val_array_d, num_elems*sizeof(Complex), cudaMemcpyDeviceToHost);
  cudaMemcpy(iter_array_h, iter_array_d, num_elems*sizeof(int), cudaMemcpyDeviceToHost);
  for(int i=0; i<nRows; i++) // copy val array into host buffer
  {
    for(int j=0; j<nCols; j++)
    {
      val_array[i][j] = val_array_h[i*nCols+j];
      iter_array[i][j] = iter_array_h[i*nCols+j];
    }
  }

  // free memory
  if(verbose) cout<<"freeing memory"<<endl;
  delete[] val_array_h;
  delete[] iter_array_h;
  cudaFree(val_array_d);
  cudaFree(iter_array_d);
  
  // end of function
  if(verbose) cout<<"end of function"<<endl;
};

template<complexFunc_t F>
void Fractal<F>::SetIterArr(vector<vector<int>> source_array) // set iter_array from another array
{
  nRows=source_array.size();
  nCols=source_array[0].size();
  iter_array=source_array;
};

template<complexFunc_t F>
void Fractal<F>::Center(Complex center_point) // move the mandelbrot set to be centered over a point p1
{
  double half_width = 0.5 * fabs(x2-x1);
  double half_height = 0.5 * fabs(y2-y1);

  Complex new_p1(center_point.real-half_width, center_point.im+half_height);
  Complex new_p2(center_point.real+half_width, center_point.im-half_height);

  x1=new_p1.real;
  y1=new_p1.im;
  x2=new_p2.real;
  y2=new_p2.im;

  // fill values
  for(int i=0; i<nRows; i++)
  {
    double y = y1 + (y2-y1)*(double(i)/(nRows-1));
    for(int j=0; j<nCols; j++)
    {
      double x = x1 + (x2-x1)*(double(j)/(nCols-1));
      val_array[i][j] = Complex(x,y);
    }
  }
}

template<complexFunc_t F>
void Fractal<F>::Zoom(double scale) // reset the coors (x1,y1), (x2,y2) to be smaller box around center
{
  double half_width = 0.5 * fabs(x2-x1);
  double half_height = 0.5 * fabs(y2-y1);
  Complex center_point((x2+x1)/2, (y2+y1)/2);

  scale = scale + (0.5 * (1-scale));
  Complex new_p1(center_point.real-half_width*scale, center_point.im+half_height*scale);
  Complex new_p2(center_point.real+half_width*scale, center_point.im-half_height*scale);


  x1=new_p1.real;
  y1=new_p1.im;
  x2=new_p2.real;
  y2=new_p2.im;

  // fill values
  for(int i=0; i<nRows; i++)
  {
    double y = y1 + (y2-y1)*(double(i)/(nRows-1));
    for(int j=0; j<nCols; j++)
    {
      double x = x1 + (x2-x1)*(double(j)/(nCols-1));
      val_array[i][j] = Complex(x,y);
    }
  }
};

// getters
template<complexFunc_t F>
vector<vector<Complex>> Fractal<F>::GetValArr() // get iter_array from member data.
{
  return val_array;
};

template<complexFunc_t F>
vector<vector<int>> Fractal<F>::GetIterArr() // get iter_array from member data.
{
  return iter_array;
};

template<complexFunc_t F>
void Fractal<F>::ValToCSV(string fname) // write iters_array to a csv file
{
  ofstream out(fname);
  if(out.is_open())
  {
    for(int i=0; i<nRows; i++)
    {
      for(int j=0; j<nCols; j++)
      {
        out << "(" << val_array[i][j].real << ", " << val_array[i][j].im << "), ";
      }
      out << "\n";
    }
      out.close();
    cout << "array saved to " << fname << " successfully!" << endl;
  }
  else
  {
    cout << "failed to write file!" << endl;
  };
};

template<complexFunc_t F>
void Fractal<F>::ItersToCSV(string fname) // write iters_array to a csv file
{
  ofstream out(fname);
  if(out.is_open())
  {
    for(int i=0; i<nRows; i++)
    {
      for(int j=0; j<nCols; j++)
      {
        out << iter_array[i][j] << ", ";
      }
      out << "\n";
    }
      out.close();
    cout << "array saved to " << fname << " successfully!" << endl;
  }
  else
  {
    cout << "failed to write file!" << endl;
  };
};

template<complexFunc_t F>
void Fractal<F>::ItersToIMG(string fname) // write iters_array to a csv file
{
  ofstream out(fname);
  if(out.is_open())
  {
    // Write PPM header
    out << "P3\n";  // PPM magic number for color image (ASCII)
    out << nCols << " " << nRows << "\n";  // Image dimensions
    out << "255\n";  // Maximum color value (for grayscale, 255 is full white)
    for(int i=0; i<nRows; i++)
    {
      for(int j=0; j<nCols; j++)
      {
        if(iter_array[i][j]==1)
        {
          out << "255 190 190 "; // shade outside radius red
        }
        else if(iter_array[i][j]>1)
        {
          double val = iter_array[i][j];
          int red_cycle=21, blue_cycle=97, green_cycle=29;
          out << 255*fmod(val,red_cycle)/red_cycle << " " 
              << 255*fmod(val,blue_cycle)/blue_cycle << " " 
              << 255*fmod(val,green_cycle)/green_cycle << " ";  // RGB values are all the same for grayscale
        }
        else
        {
          out << "0 0 0 "; // color the mandelbrot set black
        }
      }
      out << "\n";
    }
      out.close();
    cout << "array saved to " << fname << " successfully!" << endl;
  }
  else
  {
    cout << "failed to write file!" << endl;
  };
};

template<complexFunc_t F>
cv::Mat Fractal<F>::ItersToFrame() // write to a opencv cv::Mat
{
  cv::Mat frame(nRows, nCols, CV_8UC3);
  for(int i=0; i<nRows; i++)
  {
    for(int j=0; j<nCols; j++)
    {
      if(iter_array[i][j]==1)
      {
        frame.at<cv::Vec3b>(i,j) = cv::Vec3b(190,190,255); // shade outside radius red
      }
      else if(iter_array[i][j]>1)
      {
        double val = iter_array[i][j];
        int red_cycle=21, blue_cycle=97, green_cycle=29;
        frame.at<cv::Vec3b>(i,j) = cv::Vec3b(int(255*fmod(val,green_cycle)/green_cycle),
                                            int(255*fmod(val,blue_cycle)/blue_cycle),
                                             int(255*fmod(val,red_cycle)/red_cycle));
      }
      else
      {
        frame.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0); // color the mandelbrot set black
      }
    }
  }
  if (frame.type() != CV_8UC3) {frame.convertTo(frame, CV_8UC3);}
  return frame;
};



// operators
template<complexFunc_t F>
Fractal<F>& Fractal<F>::operator = (const Fractal& source)
{
  if(this==&source) return *this;
  nRows=source.nRows;
  nCols=source.nCols;
  num_iters=source.num_iters;
  radius=source.radius;
  x1=source.x1;
  y1=source.y1;
  x2=source.x2;
  y2=source.y2;
  val_array=source.val_array; // array containing each pixel's value
  iter_array=source.iter_array; // array that counts how many iterations for a pixel to explode
  return *this;
}



//template instantiations
template class Fractal<p_mandelbrot_func>; 
template class Fractal<p_julia_func>;
template class Fractal<p_burning_ship_func>;

#endif


