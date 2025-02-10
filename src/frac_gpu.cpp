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
  tuple<int,int> dims = make_tuple(1080,1080);
  Complex p1(-2,2), p2(2,-2);
  Mandelbrot A(p1,p2,dims,99);
  // print_arr(A.GetValArr());


  printf("calcuating... \n");
  A.CalculateCuda();
  // print_arr(A.GetIterArr());

  printf("saving to file!\n");
  A.ItersToIMG();

  return 0;
};


