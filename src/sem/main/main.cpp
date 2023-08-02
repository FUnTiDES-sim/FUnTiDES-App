//************************************************************************
//  SEM proxy application v.0.0.1
//
//  main.cpp: this main file is simply a driver
//************************************************************************

#include "semproxy.hpp"
#include <chrono>

#include "Array.hpp"
#include "MallocBuffer.hpp"
#include "ChaiBuffer.hpp"

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
  LvArray::Array< int,
                  2,
                  camp::idx_seq< 1, 0 >,
                  std::ptrdiff_t,
                  LvArray::ChaiBuffer > array( 5, 6 );

  // Move the array to the device.
  array.move( LvArray::MemorySpace::cuda );
  int * const devicePointer = array.data();

  RAJA::forall< RAJA::cuda_exec< 32 > >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, array.size() ),
    [devicePointer] __device__ ( std::ptrdiff_t const i )
  {
    devicePointer[ i ] = i;
  }
    );

  LvArray::ArrayView< int,
                      2,
                      0,
                      std::ptrdiff_t,
                      LvArray::ChaiBuffer > const & view = array;

  // Capture the view in a host kernel which moves the data back to the host.
  RAJA::forall< RAJA::loop_exec >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, view.size() ),
    [view] ( std::ptrdiff_t const i )
  {
    //EXPECT_EQ( view.data()[ i ], i );
  }
    );


}
