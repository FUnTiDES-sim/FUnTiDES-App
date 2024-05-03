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
}

// Run the simulation.
void SEMproxy::run()
{
  mySolver.computeFEInit( myMeshinfo, myMesh );

  for( int indexTimeStep=0; indexTimeStep<myNumSamples; indexTimeStep++ )
  {
      mySolver.computeOneStep( indexTimeStep, myMeshinfo, i1, i2, myRHSTerm, pnGlobal, rhsElement);

      mySolver.outputPnValues(myMesh, indexTimeStep, i1,  myElementSource, pnGlobal);

      swap( i1, i2 );
  }
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
  myRHSTerm=allocateArray2D<arrayReal>( myMeshinfo.myNumberOfRHS, myNumSamples , "RHSTerm");
  pnGlobal=allocateArray2D<arrayReal>( myMeshinfo.numberOfNodes, 2 , "pnGlobal");
  rhsElement=allocateVector<vectorInt>(myMeshinfo.myNumberOfRHS, "rhsElement");
}

// Initialize sources
void SEMproxy::init_source()
{
  arrayReal myRHSLocation=allocateArray2D<arrayReal>( myMeshinfo.myNumberOfRHS, 3, "RHSLocation" ); 
  // set number of rhs and location
  myRHSLocation(0,0)=1001;
  myRHSLocation(0,1)=1001;
  myRHSLocation(0,2)=1001;
  cout << "Source location: "<<myRHSLocation(0,0)<<", "<<myRHSLocation(0,1)<<", "<<myRHSLocation(0,2)<<endl;
  for( int i=0; i<myMeshinfo.myNumberOfRHS; i++ )
  {
    //extract element number for current rhs
    rhsElement[i]=myMesh.getElementNumberFromPoints( myRHSLocation(i,0), myRHSLocation(i,1), myRHSLocation(i,2) );
    printf(" rhsElement=%d\n",rhsElement[i]);
  }

  // initialize source term
  vector< float > sourceTerm=myUtils.computeSourceTerm( myNumSamples, myMeshinfo.myTimeStep, f0, sourceOrder );
  for( int j=0; j<myNumSamples; j++ )
  {
    myRHSTerm(0,j)=sourceTerm[j];
    if( j%100==0 ) cout<<"Sample "<<j<<"\t: sourceTerm = "<<sourceTerm[j]<<endl;
  }
  // get element number of source term
  myElementSource=myMesh.getElementNumberFromPoints( myRHSLocation(0,0), myRHSLocation(0,1),myRHSLocation(0,2) );
  cout <<"Element number for the source location: "<<myElementSource<<endl;
}


