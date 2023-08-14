//************************************************************************
//  SEM proxy application v.0.0.1
//
//  main.cpp: this main file is simply a driver
//************************************************************************

#include "semproxy.hpp"
#include <chrono>

int main( int argc, char *argv[] )
{
  #ifdef SEM_USE_KOKKOS
  Kokkos::initialize();
  #endif
  SEM_CALIPER_MARK_BEGIN( "TotalTime" );

  chrono::time_point< chrono::system_clock > startTime = chrono::system_clock::now();

  SEMProxy semsim;

  cout << "\n+================================= "<<endl;
  cout << "| Initializing SEM Application ... "<<endl;
  cout << "+================================= "<<endl;
  semsim.init();

  cout << "\n+================================= "<<endl;
  cout << "| Running SEM Application ...      "<<endl;
  cout << "+================================= "<<endl;
  semsim.run();

  cout << "\n+================================= "<<endl;
  cout << "| SEM Application Finished.       "<<endl;
  cout << "+================================= \n"<<endl;

  #ifdef SEM_USE_KOKKOS
  Kokkos::finalize();
  #endif
}
