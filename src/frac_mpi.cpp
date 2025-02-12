#include<iostream>
#include<stdio.h>
#include<string>
#include<math.h>

#include "Fractals.hpp"

#include "mpi.h"

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
  // set up mpi 
  int my_PE_num, num_PEs;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_PE_num);
  MPI_Comm_size(MPI_COMM_WORLD,&num_PEs);
  MPI_Status status;
  MPI_Request send_request;

  // manager PE
  if(my_PE_num==0)  
  {
    cout << "working with " << num_PEs << " PEs" << endl;
  }

  int rows=1080,cols=1920; // video dimensions

  // top left and bottom right coords in complex plan
  double X1,Y1, X2,Y2; 
  X1=-2;Y1=2;
  X2=2;Y2=-2;
  Complex p1(X1,Y1), p2(X2,Y2);

  // build fractal on each PE
  Mandelbrot *my_frac = nullptr;
  my_frac = new Mandelbrot(p1,p2,make_tuple(rows,cols),9999); // iterations to perform
  my_frac->Center(Complex(
    -0.743643887037151, 0.131825904205330 // interesting coordinate...
  ));
  
  // setup cv video writer, only use on PE 0
  int video_length = 1*60; // vid length in second
  int fps=30; // vid frames per second
  double zoom_speed = 0.95; // 0 < zoom_seed < 1

  cv::VideoWriter writer;
  string filename = "out/mandelbrot_zoom_mpi.avi"; // output video file
  int fourcc = cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'); // Specify the codec and create VideoWriter object
  writer.open(filename, fourcc, fps, cv::Size(cols,rows));

  //setup 1d and 2d buffers for iters array
  vector<vector<int>> buff = my_frac->GetIterArr();
  vector<int> send_buff(buff.size()*buff[0].size());
  vector<int> recv_buff(buff.size()*buff[0].size());

  // set up each PE to be a different zoom
  my_frac->Zoom(pow(zoom_speed,my_PE_num));

  // chunk through video length...
  int num_chunks = sdiv(video_length*fps,num_PEs);
  for(int chunk=0; chunk<num_chunks; chunk++)
  {
    // all PEs > 0, calculate and send iteration array
    if(my_PE_num>0)
    {
      my_frac->CalculateCudaGPUs(); // utilizes all gpus available
      buff=my_frac->GetIterArr();
      for(int i=0; i<rows; i++) // manually load send_buffer
      {
        for(int j=0; j<cols;  j++)
        {
          send_buff[i*cols + j]=buff[i][j];
        }
      }
      // non blocking 
      if(chunk>0) MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      MPI_Isend(send_buff.data(),rows*cols,MPI_INT, 0, my_PE_num, MPI_COMM_WORLD, &send_request);
      // blocking
      // MPI_Send(send_buff.data(),rows*cols,MPI_INT, 0, my_PE_num, MPI_COMM_WORLD);
    }

    // on PE 0 write whole chunk to cv::videowriter object
    if(my_PE_num==0)
    {
      my_frac->CalculateCudaGPUs(); // utilizes all gpus available
      if(chunk==0) my_frac->ItersToIMG("out/frame_1_mpi.ppm");
      writer.write(my_frac->ItersToFrame());

      // recv from PEs > 0 and write to cv::videowriter object
      for(int PE=1; PE<num_PEs; PE++) 
      {
        MPI_Recv(recv_buff.data(),rows*cols,MPI_INT, PE, PE, MPI_COMM_WORLD, &status); // unblocks other PE
        // manually unpack into 2d array
        for(int i=0; i<rows; i++) 
        {
          for(int j=0; j<cols; j++) 
          {
            buff[i][j] = recv_buff[i*cols+j];
          }
        }
        // load buffer to fractal object
        my_frac->SetIterArr(buff);
        //write to video
        writer.write(my_frac->ItersToFrame());
      }
    }

    // all PEs zoom in to next chunk's frame sizes
    my_frac->Zoom(pow(zoom_speed,num_PEs));

  }

  //clean up data
  writer.release();

  delete my_frac;

  MPI_Finalize();

  return 0;

};



