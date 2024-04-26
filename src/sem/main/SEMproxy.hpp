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

  // get information from mesh
  void getMeshInfo();


private:

  SEMsolver mySolver;
  SolverUtils myUtils;

  int i1=0;
  int i2=1;

  const float f0=10.;
  const float myTimeMax=1.;
  const int sourceOrder=1;

  SEMmeshinfo myMeshinfo;
  int myNumSamples=myTimeMax/myMeshinfo.myTimeStep;
  int myElementSource;


//  SEMmesh  myMesh {50, 50 ,50, 2000, 2000, 2000, myMeshinfo.myOrderNumber};
  SEMmesh  myMesh {65, 65 ,65, 1950, 1950, 1950, myMeshinfo.myOrderNumber};
//  SEMmesh myMesh {65, 0 ,65, 1950, 0, 1950, myMeshinfo.myOrderNumber};
 
   // arrays
  arrayRealView myRHSLocation;
  arrayRealView myRHSTerm;
  arrayIntView nodeList;
  arrayRealView pnGlobal;
  vectorIntView rhsElement;

  #ifdef USE_RAJA
  arrayReal h_myRHSLocation;
  arrayReal h_myRHSTerm;
  arrayInt h_nodeList;
  arrayReal h_pnGlobal;
  vectorInt h_rhsElement;
  #endif

};

#endif /* SEMPROXY_HPP_ */
