//************************************************************************
//   proxy application v.0.0.1
//
//  semproxy.cpp: the main interface of  proxy application
//
//************************************************************************

#include "SEMproxy.hpp"
// Initialize the simulation.
void SEMproxy::initFiniteElem()
{
  // get information from mesh
  getMeshInfo();

  // allocate arrays and vectors
  init_arrays();

  // initialize source and RHS
  init_source();

  mySolver.computeFEInit( myMeshinfo, myMesh );

}

// Run the simulation.
void SEMproxy::run()
{
  time_point< system_clock > startComputeTime, startOutputTime, totalComputeTime, totalOutputTime;

  for( int indexTimeSample=0; indexTimeSample<myNumSamples; indexTimeSample++ )
  {
    startComputeTime = system_clock::now();
    mySolver.computeOneStep( indexTimeSample, myMeshinfo.myOrderNumber, myMeshinfo.nPointsPerElement, i1, i2, 
                             myMeshinfo, myRHSTerm, pnGlobal, rhsElement );
    totalComputeTime += system_clock::now() - startComputeTime;

    startOutputTime = system_clock::now();
    mySolver.outputPnValues( myMesh, indexTimeSample, i1, myElementSource, pnGlobal );
    swap( i1, i2 );
    totalOutputTime += system_clock::now() - startOutputTime;
  }
  float kerneltime_ms = time_point_cast< microseconds >( totalComputeTime ).time_since_epoch().count();
  float outputtime_ms = time_point_cast< microseconds >( totalOutputTime ).time_since_epoch().count();

  cout << "------------------------------------------------ "<< endl;
  cout << "\n---- Elapsed Kernel Time : "<< kerneltime_ms/1E6<<" seconds."<< endl;
  cout << "---- Elapsed Output Time : "<< outputtime_ms/1E6<<" seconds."<< endl;
  cout << "------------------------------------------------ "<< endl;

}

// Initialize arrays
void SEMproxy::getMeshInfo()
{
  // get information from mesh
  myMeshinfo.numberOfNodes=myMesh.getNumberOfNodes();
  myMeshinfo.numberOfElements=myMesh.getNumberOfElements();
  myMeshinfo.numberOfPointsPerElement=myMesh.getNumberOfPointsPerElement();
  myMeshinfo.numberOfInteriorNodes=myMesh.getNumberOfInteriorNodes();
  myMeshinfo.numberOfPointsPerElement=myMesh.getNumberOfPointsPerElement();
  myMeshinfo.numberOfBoundaryNodes=myMesh.getNumberOfBoundaryNodes();
  myMeshinfo.numberOfBoundaryFaces=myMesh.getNumberOfBoundaryFaces();
}

// Initialize arrays
void SEMproxy::init_arrays()
{
  cout<<"Allocate host memory for source and pressure values ..."<< endl;
  myRHSTerm=allocateArray2D< arrayReal >( myMeshinfo.myNumberOfRHS, myNumSamples, "RHSTerm" );
  rhsElement=allocateVector< vectorInt >( myMeshinfo.myNumberOfRHS, "rhsElement" );
  pnGlobal=allocateArray2D< arrayReal >( myMeshinfo.numberOfNodes, 2, "pnGlobal" );
}

// Initialize sources
void SEMproxy::init_source()
{
  arrayReal myRHSLocation=allocateArray2D< arrayReal >( myMeshinfo.myNumberOfRHS, 3, "RHSLocation" );
  // set number of rhs and location
  myRHSLocation( 0, 0 )=1001;
  myRHSLocation( 0, 1 )=1001;
  myRHSLocation( 0, 2 )=1001;
  cout << "\nSource location: "<<myRHSLocation( 0, 0 )<<", "<<myRHSLocation( 0, 1 )<<", "<<myRHSLocation( 0, 2 )<< endl;
  for( int i=0; i<myMeshinfo.myNumberOfRHS; i++ )
  {
    //extract element number for current rhs
    rhsElement[i]=myMesh.getElementNumberFromPoints( myRHSLocation( i, 0 ), myRHSLocation( i, 1 ), myRHSLocation( i, 2 ));
    //printf("Element number for the source %d is: %d\n", i, rhsElement[i]);
  }

  // initialize source term
  vector< float > sourceTerm=myUtils.computeSourceTerm( myNumSamples, myMeshinfo.myTimeStep, f0, sourceOrder );
  for( int j=0; j<myNumSamples; j++ )
  {
    myRHSTerm( 0, j )=sourceTerm[j];
    if( j%100==0 )
      cout<<"Sample "<<j<<"\t: sourceTerm = "<<sourceTerm[j]<< endl;
  }
  // get element number of source term
  myElementSource=rhsElement[0];
  cout <<"Element number for the source location: "<<myElementSource<< endl<< endl;
}
