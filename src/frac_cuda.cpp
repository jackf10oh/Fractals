#include<iostream>
#include<string>
#include<fstream>
#include<tuple>
#include<vector>
#include "Fractals.hpp"

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
  tuple<int,int> dims = make_tuple(1080,1920); // rows, cols of frame
  Complex p1(-2,2), p2(2,-2);
  Mandelbrot A(p1,p2,dims,9999); // iterations to perform
  A.Center(Complex(
    -0.743643887037151, 0.131825904205330 // interesting coordinate...
  ));

  // setup cv video writer
  int video_length = 1*60; // vid length in second
  int fps=30; // vid frames per second
  double zoom_speed = 0.95; // 0 < zoom_seed < 1

  cv::VideoWriter writer;
  string filename = "out/mandelbrot_zoom_cuda.avi"; // output file video
  int fourcc = cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'); // Specify the codec and create VideoWriter object
  writer.open(filename, fourcc, fps,cv::Size(get<1>(dims),get<0>(dims))); // open video writer

  if (!writer.isOpened()) {
    cerr << "Error opening video writer!" << endl;
    return -1;
  }
  
  printf("calcuating... \n");
  A.CalculateCuda(); // single kernel
  // A.CalculateCudaStreams(); // multiple kernel copy/compute in concurrent streams
  // A.CalculateCudaGPUs(); // concurrent streams in muliplt GPUs
  // A.CalculateCudaGPUs1Stream(); // concurrent streams in muliplt GPUs
  A.ItersToIMG("out/frame_1_cuda.ppm"); // output of first frame

  cv::Mat frame;
  for(int i=0; i<video_length*fps; i++) // write iters to video and zoom in
  {
    frame = A.ItersToFrame();
    writer.write(frame);
    A.Zoom(zoom_speed);
    A.CalculateCuda();
    // A.CalculateCudaStreams();
    // A.CalculateCudaGPUs();
    // A.CalculateCudaGPUs1Stream();
  };

  writer.release();

  return 0;

};



