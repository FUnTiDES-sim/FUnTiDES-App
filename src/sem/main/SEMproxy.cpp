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
      mySolver.computeOneStep( indexTimeStep, myMeshinfo, i1, i2, rhsElement, myRHSTerm, pnGlobal);

      mySolver.outputPnValues(indexTimeStep, i1,  myElementSource, nodeList, pnGlobal);

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
  #ifdef USE_RAJA
  // allocate arrays and vectors
  h_myRHSLocation=allocateArray2D<arrayReal>( myMeshinfo.myNumberOfRHS, 3 );
  h_myRHSTerm=allocateArray2D<arrayReal>( myMeshinfo.myNumberOfRHS, myNumSamples );
  h_nodeList=allocateArray2D<arrayInt>(myMeshinfo.numberOfElements,myMeshinfo.numberOfPointsPerElement);
  h_pnGlobal=allocateArray2D<arrayReal>( myMeshinfo.numberOfNodes, 2 );
  h_rhsElement=allocateVector<vectorInt>(myMeshinfo.myNumberOfRHS);

  // allocate arrays and vectors
  myRHSLocation=h_myRHSLocation.toView();
  myRHSTerm=h_myRHSTerm.toView();    
  nodeList=h_nodeList.toView();     
  pnGlobal=h_pnGlobal.toView();     
  rhsElement=h_rhsElement.toView();   
  #else
  // allocate arrays and vectors
  myRHSLocation=allocateArray2D<arrayRealView>( myMeshinfo.myNumberOfRHS, 3 );
  myRHSTerm=allocateArray2D<arrayRealView>( myMeshinfo.myNumberOfRHS, myNumSamples );
  nodeList=allocateArray2D<arrayIntView>(myMeshinfo.numberOfElements,myMeshinfo.numberOfPointsPerElement);
  pnGlobal=allocateArray2D<arrayRealView>( myMeshinfo.numberOfNodes, 2 );
  rhsElement=allocateVector<vectorIntView>(myMeshinfo.myNumberOfRHS);
  #endif

  // get nodelist 
  myMesh.globalNodesList( myMeshinfo.numberOfElements, nodeList );
}

// Initialize sources
void SEMproxy::init_source()
{
  // set number of rhs and location
  myRHSLocation(0,0)=1001;
  myRHSLocation(0,1)=0;//1001;
  myRHSLocation(0,2)=1001;
  cout << "Source location: "<<myRHSLocation(0,0)<<", "<<myRHSLocation(0,1)<<", "<<myRHSLocation(0,2)<<endl;
  for( int i=0; i<myMeshinfo.myNumberOfRHS; i++ )
  {
    //extract element number for current rhs
    float x=myRHSLocation(i,0);
    float y=myRHSLocation(i,1);
    float z=myRHSLocation(i,2);
    int rhsE=myMesh.getElementNumberFromPoints( x, y,z );
    rhsElement[i]=rhsE;
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


