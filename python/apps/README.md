# Workflows folder

This folder contains the workflows for modeling, inversion and model conversion.

## Modeling scripts

### Client-server
- `modeling/acoustic_modelingCS.py`
    This script is a forward, acoustic problem for which a decision file maps the acquisition. It uses a client-server configuration.

- `modeling/elastic_modelingCS.py`
   This is the elastic equivalent of the previous script.

The acquisition for these two workflows can be either set in the xml directly, or by giving a decision file (and associated parameters) in the command line.

With a decision.mapped file:
```bash
Launcher --np <np> --server_queue=<Q_SERVER> --queue=<Q_CLI> --nb_client=<nb_CLI> --gpn=<n_GPU> --flag_cliser python3 acoustic_modelingCS.py --xml <xmlfile> flag_cli_ser=1 --decision decision --scalco=10 --scalel=-10
```
Or if the sources and receiver coordinates are defined in the xml, it simplifies as:
```bash
Launcher --np <np> --server_queue=<Q_SERVER> --queue=<Q_CLI> --nb_client=<nb_CLI> --gpn=<n_GPU> --flag_cliser python3 acoustic_modelingCS.py --xml <xmlfile> flag_cli_ser=1
```

### Standalone
- `modeling/acoustic_modeling.py`
```bash
Launcher --np <np> --queue=<Q_CLI> --gpn=<n_GPU> python3 acoustic_modeling.py --xml <xmlfile> flag_cli_ser=1 --decision decision --scalco=10 --scalel=-10
```

- `modeling/elastic_modeling.py`

Standalone equivalent of acoustic case


## Full Waveform Inversion (FWI)

- `fullWaveformInversion/fwi.py` is a Full-Waveform Inversion workflow. The acquisition is defined from a folder containing .sgy files.
Parameters should be given *via* a txt file of the following form:

```bash
sourceType=ricker2   #parameter, need value
exportSeismicTraces  #flag, only need argument
```

See `examples/scripts/fwi/fwi.py` for an example run

Form of the command line:
```bash
Launcher --np=12 --ppn=6 --server_queue=<Q_SER> --queue=<Q_CLI> --nb_clients=3 --gpn=4 --flag_cliser python3 fwi.py --xml <xml file> -x 2 -z 6 flag_cli_ser=1 --param_file <parameter file> --segdir <segy directory>
```

## Coupled workflow - Reservoir/Seismic

- `coupled/res+seis.py`

This script is an example of coupled reservoir/seismic workflow. You need to define the PEM properties in a dedicated class PEM in the likes of the one available in the subfolder `coupled`

First, launch the seismic job in CS mode with the launcher. Get the solver job ID to launch the reservoir in client-only mode, such as:
```bash
    cd coupled
    Launcher --np=12 --ppn=12 --gpn=6 --queue=<QCLI> --server_queue=<QSER> --nb_clients=3 --jobname=SEIS --flag_cliser python3 res+seis.py --rxml <reservoir_xml> --sxml <seismic_xml> --decision selection_decision.mapped --scalco -10 --scalel -10 --phystype 0 flag_cli_ser=1
```
Then:
```bash
   Launcher --np=12 --ppn=12 --gpn=6 --queue=<QCLI> --nb_clients=1 --jobname=RES --CS_client_only_job --CS_server_jobid=<SERVER_JOBID> --CS_key=CSKEY python3 res+seis.py --rxml <seismic_xml> --sxml $SXML --decision selection_decision.mapped --scalco -10 --scalel -10 --phystype 1 flag_cli_ser=1
```


## Reservoir workflow

- `reservoir/reservoir.py`

This is a reservoir simulation workflow. You can find an executable example in the `examples/reservoir` folder (see examples' README).
```bash
    cd reservoir
    Launcher --np 6 --gpn=6 --queue=<QUEUE> --jobname=RES python3 reservoir.py --xml <XMLFILE>
```


## Model processing

- `modelProcessing/SepToVtk.py`

This script `SeptoVtk.py` allows to convert SEP models to VTK or PVTK format

For a single dataset of name `mediumVelocity`, you can convert the `inputmodel.H` to a vtu format with the following command line:

```bash
    cd modelProcessing
    python3 SeptoVtk.py --infile=/path/to/inputmodel.H --out vtu --pn=mediumVelocity
```

More information and examples in the docs (`userguide/Model`)
