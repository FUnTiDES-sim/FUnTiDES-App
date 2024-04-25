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
  mySolver.computeFEInit( myOrderNumber,myMesh, myQk);

  for( int indexTimeStep=0; indexTimeStep<myNumSamples; indexTimeStep++ )
  {
      mySolver.computeOneStep( indexTimeStep, myTimeStep, myOrderNumber, i1, i2, myNumberOfRHS, rhsElement, myRHSTerm, pnGlobal);

    //writes debugging ascii file.
    if( indexTimeStep%50==0 )
    {
      cout<<"TimeStep="<<indexTimeStep<<endl;
    }
    if( indexTimeStep%100==0 )
    {
      cout<<" pnGlobal @ elementSource location "<<myElementSource
          <<" after computeOneStep = "<< pnGlobal(nodeList(myElementSource,0),i1)<<endl;
      //myUtils.saveSnapShot( indexTimeStep, i1, pnGlobal, myMesh );
    }
    swap( i1, i2 );
  }
}

// Initialize arrays 
void SEMproxy::getMeshInfo()
{
  // get information from mesh
  numberOfNodes=myMesh.getNumberOfNodes();
  numberOfElements=myMesh.getNumberOfElements();
  numberOfPointsPerElement=myMesh.getNumberOfPointsPerElement();
}

// Initialize arrays 
void SEMproxy::init_arrays()
{
  // allocate arrays and vectors
  myRHSLocation=allocateArray2D<arrayRealView>( myNumberOfRHS, 3 );
  myRHSTerm=allocateArray2D<arrayRealView>( myNumberOfRHS, myNumSamples );
  nodeList=allocateArray2D<arrayIntView>(numberOfElements,numberOfPointsPerElement);
  pnGlobal=allocateArray2D<arrayRealView>( numberOfNodes, 2 );
  rhsElement=allocateVector<vectorIntView>(myNumberOfRHS);

  // get nodelist 
  myMesh.globalNodesList( numberOfElements, nodeList );
}

// Initialize sources
void SEMproxy::init_source()
{
  // set number of rhs and location
  myRHSLocation(0,0)=1001;
  myRHSLocation(0,1)=0;//1001;
  myRHSLocation(0,2)=1001;
  cout << "Source location: "<<myRHSLocation(0,0)<<", "<<myRHSLocation(0,1)<<", "<<myRHSLocation(0,2)<<endl;
  for( int i=0; i<myNumberOfRHS; i++ )
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
  vector< float > sourceTerm=myUtils.computeSourceTerm( myNumSamples, myTimeStep, f0, sourceOrder );
  for( int j=0; j<myNumSamples; j++ )
  {
    myRHSTerm(0,j)=sourceTerm[j];
    if( j%100==0 ) cout<<"Sample "<<j<<"\t: sourceTerm = "<<sourceTerm[j]<<endl;
  }
  // get element number of source term
  myElementSource=myMesh.getElementNumberFromPoints( myRHSLocation(0,0), myRHSLocation(0,1),myRHSLocation(0,2) );
  cout <<"Element number for the source location: "<<myElementSource<<endl;
}


