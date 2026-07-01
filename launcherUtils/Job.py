import subprocess

class JobScheduler:
    def __init__(self, name):
        """
        Job Scheduler class.
        Defines the main tag used for job submission with LSF and SLURM
        
        Parameters
        -----------
            name : str
                Name of the job scheduler (LSF/slurm/(other?) )

        Attributes
        -----------
            name : str
                Name of the job scheduler
            sub : str
                Script submission command
            queue : str
                Queue or partition requested
            time : str
                Maxtime requested for the job
            arraylog : str
                Option for logging job arrays
        """
        self.name = name.lower()

        if "lsf" == self.name:
            self.sub = "bsub"
            self.queue = "-q"
            self.time = "-W"
            self.arraylog = "_%J_%I"
        elif "slurm" == self.name:
            self.sub = "sbatch"
            self.queue = "-p"
            self.time = "-t"
            self.arraylog = "_%A_%a"            

    def ppn(self, ppn):
        """
        Dynamic option for the number of Processes per node (ppn).

        Parameters
        ----------- 
            ppn : int
                Number of processes per node requested
        
        Returns
        --------
            str : Command option for the job scheduler 
        """
        if self.name == "lsf":
            return f"-R \"span[ptile={ppn}]\""
        elif self.name == "slurm":
            return f"--ntasks-per-node={ppn}"

    def gpn(self, gpn):
        """ 
        Dynamic option for the number of Processes per node (ppn).

        Parameters
        -----------
            gpn : int
                Number of gpus per node requested

        Returns
        --------
            str : Command option for the job scheduler
        """
        if self.name == "lsf":
            return f"-gpu \"num={gpn}:mode=exclusive_process\""
        elif self.name == "slurm":
            return f"--gpu-per-node={gpn} --exclusive"

    def serverdep(self,jobid):
        """
        Set server dependency for client job

        Parameters
        ----------
            jobid : int
                Server Job ID
        
        Returns
        --------
            str : Command option for the job scheduler
        """
        if self.name == "lsf":
            return f"-w \"started({jobid})\""
        elif self.name == "slurm":
            return f"--dependency=after:{jobid}"
        
    def subarray(self, njob):
        """
        Command option to launch a Job Array.

        Parameters
        ----------
            njob : int
                Number of jobs constituting the job array

        Returns
        -------
            str : Command option for the job scheduler
        """
        if self.name == "lsf":
            return f"[1-{njob}]"
        elif self.name == "slurm":
            return f" --array=1-{njob}"


class Job:
    """
    Generic Job class

    Attributes
    ----------
        jobID : int 
            Job ID. Default is None
        jobtype : str
            Type of job. Client/server/job (standalone)
        js : JobScheduler
            Job scheduler
        subcmd :str
            Job submission command line
    """
    def __init__(self, param={}, jobscheduler=None):
        """
        Parameters
        ----------
            param : dict
                Dict containing job parameters (queue, jobname, ...)
            jobscheduler : str
                Job scheduler
        """
        self.jobID = None
        self.jobtype = "job" 
        self.param = param
        self.js = JobScheduler(jobscheduler)
        self.subcmd = ""

    def createsubcmd(self):
        """Generate the command line to execute for the job submission."""
        self.subcmd += ""
        if self.js.name in ["lsf", "slurm"]:
            self.subcmd += self.js.sub
            for k in self.param.keys():
                if self.param[k] != None:
                    if k == "np":
                        self.subcmd += f" -n {self.param[k]}"
                    elif k == "ppn":
                        self.subcmd += f" {self.js.ppn(self.param[k])}"
                    elif k == "queue":
                        self.subcmd += f" {self.js.queue} {self.param[k]}"
                    elif k == "gpn":
                        self.subcmd += f" {self.js.gpn(self.param[k])}"
                    elif k == "jobname":
                        self.subcmd += f" -J {self.param[k]}"
                    elif k == "max_time":
                        self.subcmd += f" {self.js.time} {self.param[k]}"
                    elif k == "sub_options":
                        self.subcmd += f" {self.param[k]}"
            self.subcmd += f" -o Job_%J.log"
            if not self.param.get("merge_logs", False):
                self.subcmd += f" -e Job_%J.err"

        else:
            print(f"Job Scheduler \"{self.js.name}\" unknown. The \"sub_options\" parameters will be considered as the submission command.")
            self.subcmd += self.param["sub_options"]

        if ("srun" in self.param['mpi_exe']) or ("mpiexec" in self.param['mpi_exe']):
            self.subcmd += f" {self.param['mpi_exe']} -n {self.param['np']} {self.param['binary_args']}"
        elif "mpirun" in self.param['mpi_exe']:
            self.subcmd += f" {self.param['mpi_exe']} -np {self.param['np']} {self.param['binary_args']}"
    
    def runJob(self):
        """Execute the submission command line in the shell."""
        print("="*70)
        print(f"Equivalent submission command for the {self.jobtype}")
        print("-"*70)
        print(self.subcmd)
        print("="*70)

        p = subprocess.Popen(self.subcmd,
                             stdout = subprocess.PIPE,
                             shell = True)
        strout = p.stdout.read().decode('utf8')

        if strout == "":
            self.jobID = -1
        else:
            if self.js.name == "lsf":
                self.jobID = strout.split("<")[1].split(">")[0]
            elif self.js.name == "slurm":
                self.jobID = strout.split()[3]


class JobServer(Job):
    """
    Class defining a server job. 
    Inheritance from Job class

    Attributes
    -----------
        jobID : int
            Job ID. Default is None
        jobtype : str
            Type of job. Client/server/Generic
        js : JobScheduler
            Job scheduler
        subcmd : str
            Job submission command line
        serveronly : bool
            Boolean indicating a server-only job or not
    """
    def __init__(self, param, jobscheduler=None):
        """Server Job initializer. Job class inheritance.
        
        Parameters
        ----------
            param : dict 
                Dict containing job parameters (queue, jobname, ...)
            jobscheduler : str
                Job scheduler
        """
        super().__init__(param, jobscheduler)
        self.jobtype = "server"
        self.serveronly = False
        if self.param["serveronly"]:
            self.serveronly = True

    def createsubcmd(self):
        """Function generating the command line to execute for the job submission.
        """
        if self.js.name in ["lsf", "slurm"]:
            self.subcmd += self.js.sub
            for k in self.param.keys():
                if self.param[k] != None:
                    if k == "np":
                        self.subcmd += f" -n {self.param[k]}"
                    elif k == "ppn":
                        self.subcmd += f" {self.js.ppn(self.param[k])}" 
                    elif k == "queue":
                        self.subcmd += f" {self.js.queue} {self.param[k]}"
                    elif k == "gpn":
                        self.subcmd += f" {self.js.gpn(self.param[k])}"
                    elif k == "jobname":
                        self.subcmd += f" -J {self.param[k]}"
                    elif k == "max_time":
                        self.subcmd += f" {self.js.time} {self.param[k]}"
                    elif k == "sub_options":
                        self.subcmd += f" {self.param[k]}"
            self.subcmd += f" -o Job_Ser_%J.log"
            if not self.param.get("merge_logs", False):
                self.subcmd += f" -e Job_Ser_%J.err"

        else:
            print(f"Job Scheduler \"{self.js.name}\" unknown. The \"sub_options\" parameters will be considered as the submission command.")
            self.subcmd += self.param["sub_options"]

        if self.js.name == "lsf":
            # Which mpi command to run
            if "mpiexec" in self.param['mpi_exe']:
                self.subcmd += f" {self.param['mpi_exe']} -n {self.param['np']} {self.param['binary_args']}"
            elif "mpirun" in self.param['mpi_exe']:
                self.subcmd += f" {self.param['mpi_exe']} -np {self.param['np']} {self.param['binary_args']}"
                
            if not "CSkey" in self.param:
                    self.param["CSkey"] = "CSKEY"
            self.subcmd += f" CSkey={self.param['CSkey']}"
            
        elif self.js.name == "slurm":
            # Which mpi command to run
            if ("srun" in self.param['mpi_exe']) or ("mpiexec" in self.param['mpi_exe']):
                self.subcmd += f" --wrap \"{self.param['mpi_exe']} -n {self.param['np']} {self.param['binary_args']}"
            elif "mpirun" in self.param['mpi_exe']:
                self.subcmd += f" {self.param['mpi_exe']} -np {self.param['np']} {self.param['binary_args']}"

            if not "CSkey" in self.param:
                    self.param["CSkey"] = "CSKEY"
            self.subcmd += f" CSkey={self.param['CSkey']}\""


class JobClient(Job):
    """ 
    Class defining a server job. 
    Inheritance from Job class

    Attributes
    -----------
        jobID : int
            Job ID. Default is None
        jobtype : str
            Type of job. Client/server/Generic
        js : JobScheduler
            Job scheduler
        subcmd : str
            Job submission command line
        clientonly : bool
            Boolean indicating a server-only job or not
        associated_server : int
            Job ID of the server associated to this client job
    """
    def __init__(self, param,jobscheduler=None):
        """
        Client Job initializer. Job class inheritance.
        
        Parameters
        ----------
            param : dict
                Dict containing job parameters (queue, jobname, ...)
            jobscheduler : str
                Job scheduler
        """
        super().__init__(param,jobscheduler)
        self.jobtype = "client"
        self.clientonly = False
        if self.param["clientonly"]:
            self.clientonly = True
        self.jobtype = "client"
        self.associated_server = None


    def createsubcmd(self):
        """Generate the command line to execute for the job submission."""
        if self.js.name in ["lsf", "slurm"]:
            self.subcmd += self.js.sub
            for k in self.param.keys():
                if self.param[k] != None:
                    if k == "np":
                        self.subcmd += f" -n {self.param[k]}"
                    elif k == "ppn":
                        self.subcmd += f" {self.js.ppn(self.param[k])}"
                    elif k == "queue":
                        self.subcmd += f" {self.js.queue} {self.param[k]}"
                    elif k == "gpn":
                        self.subcmd += f" {self.js.gpn(self.param[k])}"
                    elif k == "jobname":
                        self.subcmd += f" -J {self.param[k]}C{self.js.subarray(self.param['nb_clients'])}"
                    elif k == "max_time":
                        self.subcmd += f" {self.js.time} {self.param[k]}"
                    elif k == "sub_options":
                        self.subcmd += f" {self.param[k]}"

            self.subcmd += f" -o Job_Cli{self.js.arraylog}.log"
            if not self.param.get("merge_logs", False):
                self.subcmd += f" -e Job_Cli{self.js.arraylog}.err"
            self.subcmd += f" {self.js.serverdep(self.associated_server)}"

        else:
            print(f"Job Scheduler \"{self.js.name}\" unknown. The \"sub_options\" parameters will be considered as the submission command.")
            self.subcmd += self.param["sub_options"]

        if self.js.name == "lsf":
            # Which mpi command to run
            if "mpiexec" in self.param['mpi_exe']:
                self.subcmd += f" {self.param['mpi_exe']} -n {self.param['np']} {self.param['binary_args']}"
            elif "mpirun" in self.param['mpi_exe']:
                self.subcmd += f" {self.param['mpi_exe']} -np {self.param['np']} {self.param['binary_args']}"

            if not "CSkey" in self.param:
                    self.param["CSkey"] = "CSKEY"
            self.subcmd += f" CSkey={self.param['CSkey']}"
            self.subcmd += f" --server=server" #default. Required for client mode in RPC 
        elif self.js.name == "slurm":
            # Which mpi command to run
            if ("srun" in self.param['mpi_exe']) or ("mpiexec" in self.param['mpi_exe']):
                self.subcmd += f" --wrap \"{self.param['mpi_exe']} -n {self.param['np']} {self.param['binary_args']}"
            elif "mpirun" in self.param['mpi_exe']:
                self.subcmd += f" {self.param['mpi_exe']} -np {self.param['np']} {self.param['binary_args']}"

            if not "CSkey" in self.param:
                    self.param["CSkey"] = "CSKEY"
            self.subcmd += f" CSkey={self.param['CSkey']}"
            self.subcmd += f" --server=server \"" #default. Required for client mode in RPC 

