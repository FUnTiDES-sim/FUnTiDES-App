Job launcher
##############

The :code:`Launcher` script allows to launch jobs on large computing facilities with job scheduler such as SLURM or IBM LSF thanks to a command line containing all the options.
It is independant from the makutu :code:`utilities` codes and can be used on its own as it is just an help to launch jobs.


1. The command line is parsed
2. The corresponding job(s) command lines submission(s) are set up
3. The job(s) is(are) launched

We suggest that you add the path to :code:`Launcher` to your :code:`$PATH` by adding the following command to your :code:`.bashrc`. This prevent you from writing the full launcher path every time you need to use it

.. code-block:: bash

    PATH=/full/path/to/repo/makutu/launcherUtils/:$PATH

.. Note:: You can check all the launcher options by executing

    .. code-block:: bash

        Launcher --help


There are four main configurations when you launch a job: **standalone**, **client-server**, **client-only**, **server-only**.

1. Standalone job
********************
Standalone job is the default job case. 

Example of standalone job command line:

.. code-block:: bash

    Launcher --np=24 --ppn=12 --gpn=4 --queue=PR --jobname=test python3 mypythonscript.py -x 10 -y 20

In this example, 1 job will be launched, with 24 MPI process split such that there are 12 processes per node. Each node will be associated to 4 GPU. The job launch will be name "test" and will be submitted to the queue called "PR". 

The end of line arguments :code:`-x 10 -y 20` are arguments for :code:`mypythonscript.py`, they will be unaltered by the launcher.

2. Client-server job
***********************
In the case of client-server jobs, three configurations are possible. 

1. Launching at the same time the server and the clients (flag :code:`--flag_cliser`)
2. Launching only the server (flag :code:`--CS_server-only_job`)
3. If a server is already running, add clients (flag :code:`--CS_client-only_job`)


.. Note:: The client-server configuration of the scripts of this repository are adapted to :program:`RPC` library (property of TOTALEnergies). Hence, the explanation below will be focused on this context.

Here are three examples of launching command corresponding to the three cases evoked earlier, when using RPC.


2.1 Client and server launched at the same time
=================================================

.. code-block:: bash

    Launcher --np=24 --ppn=12 --queue=Q_CS --nb_clients=2 --gpn=4 --jobname=CS --flag_cliser python3 myscript.py -i xmlfile.xml -z 24 flag_cli_ser=1

In this example, both the client and the server run on the same queue :code:`Q_CS`, it is possible to differentiate the queues by using the flag :code:`--serveur_queue` for the server, in addition to :code:`--queue` for the clients.

.. important:: Don't forget the option :code:`flag_cli_ser=1` in the python options such that :program:`RPC` recognizes the client-server mode.


2.2 Only the server is launched
=================================

.. code-block:: bash
    
    Launcher --server_queue=Q_SER --jobname=SO --CS_server_only_job python3 myscript.py -i xmlfile.xml -z 24 flag_cli_ser=1

By default, the server is set with 1 MPI process and no GPU. It could be modified with the launcher options (see :code:`Launcher --help` for the available options) but it is not advised to do so as the server is not supposed to execute large computation.

2.3 Add clients to a server
=============================

.. code-block:: bash
    
    Launcher --np=24 --ppn=12 --queue=Q_CLI --gpn=4 --nb_clients=2 --jobname=CO --CS_client_only_job --CS_server_jobid=337062 --CS_key=CSKEY python3 myscript.py -i xmlfile.xml -z 24 flag_cli_ser=1

In this case, you can note new options :code:`--CS_server_jobid` and :code:`--CS_key` that correspond respectively to the **server Job ID** given by the job scheduler and to a file containing information about the associated server : **name, port, http port**. This file is generated automatically by RPC when you launch the server, but you need to indicate the correct name.

.. note:: You cannot launch a client only job without an associated server. 

.. important:: When using :code:`PYGEOSX` environment, it should be activated before lauching the jobs!