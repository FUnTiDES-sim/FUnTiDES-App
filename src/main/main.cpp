#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include "QkGL.hpp"
#include "simpleMesh.hpp"
#include "utils.hpp"
#include "solver.hpp"

using namespace std;

int main()
{
    int ex;
    int ey;
    int nx,ny;
    int order;
    float lx;
    float ly;
    float hx,hy;
    
    QkGL Qk;
    simpleMesh mesh {ex=100,ey=100,lx=1000,ly=1000,order=1};
    solver solve; 
    solverUtils utils;

    float timeMax=1;
    float timeStep=0.001;
    int nSamples=timeMax/timeStep;
    int indexTimeStepSource=nSamples;
    // iniatialize source term
    float f0=4.;
    int sourceOrder=1;
    vector<float>sourceTerm=utils.computeSourceTerm(nSamples, timeStep, f0,sourceOrder );
    for ( int i=0; i<nSamples;i++)
    {
        if( i%100==0)cout<<"sample "<<i<<" sourceTerm="<<sourceTerm[i]<<endl;
    }
    // set number of rhs and location
    int numberOfRHS=1;
    vector<vector<float>>rhsLocation(numberOfRHS,vector<float>(2));
    vector<vector<float>>rhsTerm(numberOfRHS,vector<float>(nSamples,0));
    rhsLocation[0][0]=101;
    rhsLocation[0][1]=501;
    cout << "source location "<<rhsLocation[0][0]<<", "<<rhsLocation[0][1]<<endl;
    // get element number of source term
    float x=rhsLocation[0][0];
    float y=rhsLocation[0][1];
    int elementSource=mesh.getElementNumberFromPoints(x,y);
    cout <<"element number for the source location "<<elementSource<<endl;
    for (int j=0;j<nSamples;j++)
    {
        rhsTerm[0][j]=sourceTerm[j];
    }

    //open ascii file for debugging
    ofstream fichier("test.txt", ios::out | ios::trunc);

    // loop over time
    int i1=0;
    int i2=1;
    int numberOfNodes=mesh.getNumberOfNodes();
    int numberOfElements=mesh.getNumberOfElements();
    vector<vector<int>> nodeList=mesh.globalNodesList(numberOfElements);
    vector<vector<float>> pnGlobal(numberOfNodes,vector<float> (2));
    for (int indexTimeStep=0; indexTimeStep<nSamples;indexTimeStep++)
    {       
        solve.addRightAndSides(indexTimeStep,numberOfRHS,i2,timeStep,pnGlobal,rhsTerm,rhsLocation,mesh);
        solve.computeOneStep(timeStep,order,i1,i2,pnGlobal,mesh,Qk);
        //writes debugging ascii file.
        int elementReceiver=mesh.getElementNumberFromPoints(501,201);
        int elementReceiver1=mesh.getElementNumberFromPoints(601,201);
        fichier<<pnGlobal[nodeList[elementReceiver][0]][i2]<<" "<<pnGlobal[nodeList[elementReceiver1][0]][i2]<<endl;
        if (indexTimeStep%10==0)
        {
           cout<<indexTimeStep<<" i1="<<i1<<" i2="<<i2<<endl;
           cout<<"pnGlobal @ elementSource location "<<elementSource<<" after computeOneStep ="<<pnGlobal[nodeList[elementSource][0]][i2]<<endl;
           cout<<"pnGlobal @ elementReceiver location "<<elementReceiver<<" after computeOneStep ="<<pnGlobal[nodeList[elementReceiver][0]][i2]<<" "
                                                                                                   <<pnGlobal[nodeList[elementReceiver1][0]][i2]<<endl;
        }
        int tmp;
        tmp=i1;
        i1=i2;
        i2=tmp;
    }
}

