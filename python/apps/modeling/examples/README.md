# Modeling examples

This folder contains a few scripts that are examples of how to build a workflow, the results are not relevant physically due to the simplification of the problems.

## Acoustic modeling
Script : `acoustic_modeling.py`
    This script is an example of **acoustic modeling workflow**. The acquisition is defined by reading a decision file. 
    A forward problem is ran, then the pressure at receivers is saved in a SEP file.

Command line:
```bash
mpirun -np 4 python3 acoustic_modeling.py --xml inputfiles/bicoucheAcousticInternal.xml --decision acquisition/decision.mapped -z 4 --scalco=-10 --scalel=-10 --swords 22 23 15 --zup
```
Note that this script can also be used in the case of an acquisition directly defined in an xml file.

## Elastic modeling
Script : `elastic_modeling.py`
    This script is an example of **elastic modeling workflow**. The acquisition is read in the xml.

Command line:
```bash
mpirun -np 4 python3 elastic_modeling.py --xml inputfiles/bicoucheElasticVTK.xml
```
Note that this script can also be used for an acquisition read from a decision file. See the command line above for the options or execute the following line to check the options:
```bash
python3 elastic_modeling.py --help
```

## Acoustic modeling with limited aperture
Script : `acquisition_limitedAperture.py`
    This script is an example of modeling workflow for a **limited aperture acquisition in client-server mode**. Here the acquisition is defined "by hand", i.e. by defining directly in the script lines of sources and receivers. 
    It is explained in the ```userguide/Limited Aperture``` section of the docs.
    The script can also run full aperture acquisitions by omitting the argument `--LA`.

Command line:
```bash
Launcher --np=12 --ppn=6 --server_queue=Q_SER --queue=Q_CLI --nb_clients=3 --gpn=4 --flag_cliser python3 acquisition_limitedAperture.py -i inputfiles/bicoucheAcousticVTK.xml -z 12 flag_cli_ser=1 --LA --l1=2000 --l2=300 --l3=500
```