//************************************************************************
//  SEM proxy application v.0.0.1
//
//  main.cpp: this main file is simply a driver
//************************************************************************

#include "semproxy.hpp"

int main( int argc, char *argv[] )
{
  #ifdef USE_KOKKOS
  Kokkos::initialize();
  { 
  #endif

  SEMProxy semsim;

  cout << "\n+================================= "<<endl;
  cout << "| Initializing SEM Application ... "<<endl;
  cout << "+================================= "<<endl;
  semsim.init();

  cout << "\n+================================= "<<endl;
  cout << "| Running SEM Application ...      "<<endl;
  cout << "+================================= "<<endl;

  // start timer
  time_point< system_clock > startTime = system_clock::now();

  semsim.run();

  cout << "\n+================================= "<<endl;
  cout << "| SEM Application Finished.       "<<endl;
  cout << "+================================= \n"<<endl;

  // print timing information
  cout << "Elapsed Time : "<<( system_clock::now()-startTime ).count()/1E9 <<" seconds.\n"<<endl;

  #ifdef USE_KOKKOS
  }
  Kokkos::finalize();
  #endif
}
