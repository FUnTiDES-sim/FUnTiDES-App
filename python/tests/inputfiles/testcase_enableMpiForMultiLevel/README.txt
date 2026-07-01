#################################################################################

This folder contains the test cases for  feature/enableMpiForMultiLevel (#2046). 
These test cases are with small number of elements for fine-grained debugging.

#################################################################################

1. For testing on low-order Q1 with internal meshes (for comparison purpose)

   How to run: for example testing on different partitions: 

   $ mpirun -n 8 ./bin/geosx  -i acous3D_Q1_abc_10x10x10.xml -x2 -y2 -z2 
   $ mpirun -n 10 ./bin/geosx  -i acous3D_Q1_abc_10x10x10.xml -x2 -y5 -z1 
   $ mpirun -n 10 ./bin/geosx  -i acous3D_Q1_abc_10x10x10.xml -x10 

2. For testing on high-order Q3 with internal meshes

   How to run: for example testing on different partitions: 

   $ mpirun -n 8 ./bin/geosx  -i acous3D_Q3_abc_2x2x2.xml -x2 -y2 -z2 
   $ mpirun -n 8 ./bin/geosx  -i acous3D_Q3_abc_10x10x10.xml -x2 -y2 -z2 
   $ mpirun -n 8 ./bin/geosx  -i acous3D_Q3_abc_100x100x100.xml -x2 -y2 -z2 

   # the following used to be a corner case sloved by https://github.com/GEOSX/GEOSX/pull/2248
   $ mpirun -n 10 ./bin/geosx  -i acous3D_Q3_abc_10x10x10.xml -x10 


3. For testing on high-order Q3 with vtk meshes (TBD)

