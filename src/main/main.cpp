//************************************************************************
//  SEM proxy application v.0.0.1
//
//  main.cpp: this main file is simply a driver
//************************************************************************

#include "semproxy.hpp"
#include <chrono>

int main( int argc, char *argv[] )
{
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

  cout << "Elapsed Time : "<<chrono::duration_cast< chrono::milliseconds >( chrono::system_clock::now() - startTime ).count() / 1000.0 <<" seconds.\n"<<endl;

  SEM_CALIPER_MARK_END( "TotalTime" );
}
