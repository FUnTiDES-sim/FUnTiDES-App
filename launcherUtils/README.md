# Launcher utils

The `Launcher` script allows to launch jobs on large computing facilities with job scheduler such as SLURM or IBM LSF thanks to a command line containing all the options.

To facilitate your utilization of the launcher, we encourage you to add the path to `Launcher` to your `$PATH`.

This launcher handles both standalone and client-server job.

**Note:** The client-server configuration is set-up for the use of the RPC library. You will need to modify the code if you are using another library.


A few command line examples:

- Standalone job
```bash
Launcher --np=24 --ppn=12 --gpn=4 --queue=PR --jobname=test python3 mypythonscript.py -x 10 -y 20
```

- Client-server job
```bash
Launcher --np=24 --ppn=12 --server_queue=<server_queue> --nb_clients=2 --queue=<client_queue> --gpn=4 --jobname=CS --flag_cliser python3 <myscript.py> -i xmlfile.xml -z 24 flag_cli_ser=1
```

- Server only job
```bash
Launcher --server_queue=<server_queue> --jobname=<jobname> --CS_server_only_job python3 <myscript.py> -i xmlfile.xml -z 24 flag_cli_ser=1
```

- Client only job
```bash
Launcher --np=24 --ppn=12 --queue=<queue> --gpn=4 --nb_clients=2 --jobname=<jobname> --CS_client_only_job --CS_server_jobid=<server_jobID> --CS_key=CSKEY python3 <myscript.py> -i xmlfile.xml -z 24 flag_cli_ser=1


More explanation in the user guide !
