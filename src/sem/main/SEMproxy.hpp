//************************************************************************
//   proxy application v.0.0.1
//
//  semproxy.hpp: the main interface of SEM proxy application
//
//************************************************************************

#ifndef SEMPROXY_HPP_
#define SEMPROXY_HPP_

#include "SEMsolver.hpp"
#include "utils.hpp"

/**
 * @class SEMproxy
 */
class SEMproxy
{
public:

  /**
   * @brief Constructor of the SEMproxy class
   */
  SEMproxy(){};

  /**
   * @brief Destructor of the SEMproxy class
   */
  ~SEMproxy(){};

  /**
   * @brief Initialize the simulation.
   * @post run()
   */
  void initFiniteElem();

  /**
   * @brief Run the simulation.
   * @pre This must be called after init()
   * @post Optional printout performance resutls
   */
  void run();

  // initialize source and RHS
  void init_source();
 
  // allocate arrays and vectors
  void init_arrays();

  // initialize models
  //void init_models();

  // get information from mesh
  void getMeshInfo();

  // get elements and nodes information
  //void getElements();

  // get basis function and corresponding derivatives
  //void getBasisDeriv();

protected:
  int i1=0;
  int i2=1;

  const float f0=10.;
  const float myTimeMax=1.;
  const float myTimeStep=0.001;

  const int sourceOrder=1;
  const int myOrderNumber=3;
  const int myNumberOfRHS=1;
  const int myNumSamples=myTimeMax/myTimeStep;


  int numberOfNodes;
  int numberOfElements;
  int numberOfPointsPerElement;
  int myElementSource;

   // initialize mesh
//  SEMmesh  myMesh {50, 50 ,50, 2000, 2000, 2000, myOrderNumber};
//  SEMmesh  myMesh {65, 65 ,65, 1950, 1950, 1950, myOrderNumber};
  SEMmesh  myMesh {65, 0 ,65, 1950, 0, 1950, myOrderNumber};
 
   // arrays
  arrayRealView myRHSLocation;
  arrayRealView myRHSTerm;
  arrayIntView nodeList;
  arrayRealView pnGlobal;
  vectorIntView rhsElement;

  SEMsolver mySolver;
  SolverUtils myUtils;

};

#endif /* SEMPROXY_HPP_ */
