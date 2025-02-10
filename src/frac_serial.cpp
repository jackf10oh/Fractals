#include<iostream>
#include<string>
#include<fstream>
#include<tuple>
#include<vector>
#include "Fractals.hpp"
#include <opencv2/opencv.hpp>
#include<opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>  // For imwrite function

using namespace std;

void print_arr(vector<vector<Complex>> val_array)
{
  for(int i=0; i<val_array.size(); i++)
  {
    for(int j=0; j<val_array[0].size(); j++)
    {
      cout << "("<< val_array[i][j].real << ", " << val_array[i][j].im << ") ";
    }
    cout << endl;
  }
};

void print_arr(vector<vector<int>> iter_array)
{
  for(int i=0; i<iter_array.size(); i++)
  {
    for(int j=0; j<iter_array[0].size(); j++)
    {
      cout << iter_array[i][j] << ", ";
    }
    cout << endl;
  }
};

int main(int argc, char** argv)
{
  tuple<int,int> dims = make_tuple(480,640);
  Complex p1(-2,2), p2(2,-2);
  Mandelbrot A(p1,p2,dims,9999); // intellisense error?
  // A.Center(Complex(0.25,0.0));
  A.Center(Complex(
    -0.743643887037151, 0.131825904205330
  ));

  printf("calcuating... \n");
  A.CalculateCuda();

  // printf("saving to file!\n");
  // A.ItersToIMG("julia.ppm");
  // A.ItersToCSV("julia.csv");
  
  cv::Mat frame;
  cv::VideoWriter writer;
  string filename = "MandelbrotZoom.avi";
  int fourcc = cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'); // Specify the codec and create VideoWriter object
  writer.open(filename, fourcc, 30, cv::Size(get<1>(dims),get<0>(dims)));
  
  for(int i=0; i<30*20; i++) // write iters to video and zoom in
  {
    frame = A.ItersToFrame();
    writer.write(frame);
    A.Zoom(0.95);
    A.CalculateCuda();
  };

  writer.release();

  A.ItersToIMG("Mandelbrot.ppm");

  return 0;
};