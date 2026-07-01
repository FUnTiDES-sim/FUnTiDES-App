import numpy as np
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py.util import pkl5

from scipy import optimize

comm = pkl5.Intracomm(MPI.COMM_WORLD)


class Minimizer:
    """
    Class containing methods for the minimization of scalar function of one or more variables\
    This class is compatible with scipy.optimize methods

    See scipy.optimize documentation for an exhaustive list of parameters:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    """
    def __init__(self, func, method="steepestDescent", jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            func : callable
                Method returning the cost function value
                When jac is set to None, func can also be a method returning both the functionnal value and the gradient value (J, gradJ). Note that this option does not work with 'steepestDescent' due to
                optimizations call in the method
            method : str
                Optimization method to be used \
                Default is 'steepest descent'
            jac : callable, optional
                Method returning the gradient of the functional
            hess : callable, optional
                Method returning the hessian of the functionnal
            hessp : callable, optional
                Hessian of objective function times an arbitrary vector p. Only for Newton-CG, trust-ncg, trust-krylov, trust-constr. Only one of hessp or hess needs to be given. If hess is provided,
                then hessp will be ignored. hessp must compute the Hessian times an arbitrary vector.
            bounds : list, optional
                Bounds on variable. Sequence of (min, max) pairs for each element in x.
            constraints : {Constraint, dict} or list of {Constraint, dict}
                Constraints definition. Only for COBYLA, SLSQP and trust-constr.
                Constraints for `trust-constr` are defined as a single object or a list of objects specifying constraints to the optimization problem.
            tol : float, optional
                Tolerance for termination. When tol is specified, the selected minimization algorithm sets some relevant solver-specific tolerance(s) equal to tol.
                For detailed control, use solver-specific options.
            callback : callable, optional
                Called after each iteration. The signature is: 'callback(xk)', where xk is the current parameter vector. For `trust-constr` method, see the Scipy.optimize documentation
            options : dict, optional
                A dictionary of methods options. See the scipy.optimize documentation for all the different options for the different methods. Or see below for 'steepestDescent' options
        """
        self.func = func
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options


    def optimize(self, x, args):
        """
        Call the optimization method

        Parameters:
        -----------
            x : array
                Current value of the variable to optimize
            args: tuple
                List of func and jac callables arguments
        """
        if self.method == "steepestDescent":
            result = SteepestDescentOptimizer( self.func, self.jac, self.hess, self.hessp, self.bounds, self.constraints, self.tol, self.callback, self.options ).optimize( x, args )

        elif self.method == "SD2":
            # Steepest Descent with scipy + Wolf conditions

            result = SteepestDescentOptimizer2( self.func, self.jac, self.hess, self.hessp, self.bounds, self.constraints, self.tol, self.callback, self.options ).optimize( x, args )            
        elif self.method == "CGPR":
            # Conjugated gradient with scipy + Wolf conditions

            result = CGPROptimizer( self.func, self.jac, self.hess, self.hessp, self.bounds, self.constraints, self.tol, self.callback, self.options ).optimize( x, args )            
        elif self.method == "CGPR2":
            # Conjugated gradient + Modified (heuristic) line search
            result = CGPROptimizer2( self.func, self.jac, self.hess, self.hessp, self.bounds, self.constraints, self.tol, self.callback, self.options ).optimize( x, args )

        else:
            result = optimize.minimize(self.func,
                                       x,
                                       args=args,
                                       method=self.method,
                                       jac=self.jac,
                                       hess=self.hess,
                                       hessp=self.hessp,
                                       bounds=self.bounds,
                                       constraints=self.constraints,
                                       tol=self.tol,
                                       callback=self.callback,
                                       options=self.options)
	        
            return result


class CGPROptimizer(Minimizer):
    """
    Class dedicated to the Polak-Ribiere algorithm of the conjugated gradient method. Close to the one in scipy.optimize.
    """
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            options:
                alpha: float, optional
                    Starting line search step
                    Default is 1e-8
                maxiter : int, optional
                    Maximum number of iterations
                    Default is 10
        """
        super().__init__(func, "CGPR", jac, hess, hessp, bounds, constraints, tol, callback, options)

        if options is None:
            options = {}
        
        self.alpha = options.get("alpha", 1e-8)
        self.maxiter = options.get("maxiter", 10)
        self.tol = options.get("tol", 0.0001)
        self.gtol = options.get("gtol", 0.0001)
        
    def optimize(self, x, args):
        """
        Optimize with conjugated gradient method

        Parameters
        -----------
            x : array
                Current value of the variable to optimize
            tol: float
                Tolerance controlling the linesearch. It corresponds to the
                relative tolerance for an acceptable step.
            gtol: float
                Tolerance controlling the convergence. Evaluates the relative
                change of the cost function.
            args: tuple
                List of func and jac callables arguments
        """
        iter = 0
        Jup = 0

        # Counter to inform how many times the linesearch was called
        lineSearchIteration = 0

        gradJ, J = self.jac( x, *args, computeFullCF=True )

        # Make one extra step to provide a starting point for the linesearch

        xup = x + self.alpha * (-gradJ[:])
        xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

        Jpast = J

        gradJ, J = self.jac( xup, *args, computeFullCF=True )

        print("Jpast: {}, J: {}".format(Jpast,J))
        print("Tolerance: {}, J(i) - J(i-1): {} ".format(self.gtol,np.abs(J - Jpast)))

        x = xup

        # Best to normalize, use gDn=1 in the options to achieve the
        # the normalization by the maximum absolute value.
        delta = -gradJ # Descent direction
        
        while iter < self.maxiter:

            # Save current model
            if self.callback is not None:
                self.callback( x )

            with open("fullCostFunction.txt", "a") as f:
                f.write(f"{iter} \t {str(J)}\n")

            print(f"\nFWI iter {iter} - linesearch run {lineSearchIteration} - Full cost function : {J}\n")

            # Max num. iter is 100
            beta = optimize.linesearch.line_search_wolfe1(f=self.func,fprime=self.jac,xk=x,pk=delta,gfk=-delta,old_fval=J,old_old_fval=Jpast,args=args, c1=1e-4,c2=0.9, amax=50, amin=1e-12,xtol=self.tol)
            
            # Fallback to line_search_wolfe2 if a bad step length is found.
            # Default max num. iter is 10.
            beta = beta[0]
            if beta is None:
                print("WARNING: Fallback to line_search_wolfe2")
                beta = optimize.linesearch.line_search_wolfe2(f=self.func,myfprime=self.jac,xk=x,pk=delta,gfk=-delta,old_fval=J,old_old_fval=Jpast,args=args,c1=1e-4,c2=0.9,amax=50)
                beta = beta[0]
                lineSearchIteration += 1
                
            
            lineSearchIteration += 1
            
            if beta is None:
                print("ERROR: step length is bad")
                break

            print("Calculated step length: ",beta)

            # Update model
            xup = x + beta * delta[:]
            # Constraining model to bounds
            xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

            Jup = self.func( xup, *args )
            print("Jup: {}, J: {}".format(Jup,J))
            print("Tolerance: {}, J(i) - J(i-1): {} ".format(self.gtol,np.abs(Jup - J)))
            if np.abs(Jup - J) < self.gtol:
                x = xup
                print("Tolerance criteria satisfied.")
                break


            # Check if model perturbation is significant.
            # Using 1e7. This controls the accuracy, 1e12 provides low accuracy,
            # 1e7 moderate, and 10 extremely high (scipy Notes).
            epsmch = np.finfo(float).eps
            factr = 1e7*epsmch
            if (((x-xup)/max(np.max(abs(x)),np.max(abs(xup)),1)) <= factr).all():
                print("WARNING: model perturbation is too small, stopping.")
                break
            
            x = xup
            d = gradJ
            gradJ = self.jac( x, *args )
            chi = (gradJ - d).dot(gradJ)/(np.linalg.norm(d)**2)
            chi = max(0,chi)
            delta = -gradJ + chi*delta
            Jpast = J
            J = Jup
                
            iter +=1

        return x            
        
        
class SteepestDescentOptimizer2(Minimizer):
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            options:
                alpha: float, optional
                    Starting line search step
                    Default is 1e-8
                maxiter : int, optional
                    Maximum number of iterations
                    Default is 10
        """
        super().__init__(func, "SD2", jac, hess, hessp, bounds, constraints, tol, callback, options)

        if options is None:
            options = {}
        
        self.alpha = options.get("alpha", 1e-8)
        self.maxiter = options.get("maxiter", 10)
        self.tol = options.get("tol", 0.0001)
        self.nbCFcheck = options.get("nbCFcheck", 5)
        self.Jtol = options.get("Jtol", 0.0001)
        
    def optimize(self, x, args):
        """
        Optimize with steepest descent method

        Parameters
        -----------
            x : array
                Current value of the variable to optimize
            tol: float
                Tolerance controlling the linesearch. It corresponds to the
                relative tolerance for an acceptable step.
            Jtol: float
                Tolerance controlling the convergence. Evaluates the relative
                change of the cost function.
            nbCfcheck : int, optional
                Number of previous cost function evaluation to check if the functionnal decreases enough
                Default is 5
            args: tuple
                List of func and jac callables arguments
        """
        iter = 0
        Jup = 0
        checkJ = np.zeros(self.nbCFcheck+1) # First value is the full J

        while iter < self.maxiter:

            # Counter to inform how many times the linesearch was called
            lineSearchIteration = 0
            
            if iter == 0:
                gradJ, J = self.jac( x, *args, computeFullCF=True )

                # Make one extra step to provide a starting point for the linesearch

                xup = x + self.alpha * (-gradJ[:])
                xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

                Jpast = J

                gradJ, J = self.jac( xup, *args, computeFullCF=True )
                
                checkJ[iter] = J # Assign 1st value

                print("Jpast: {}, J: {}".format(Jpast,J))
                print("Tolerance: {}, J(i) - J(i-1): {} ".format(self.Jtol,np.abs(J - Jpast)))

                x = xup

            else:
                gradJ = self.jac( x, *args )
                J = Jup

                
            if self.callback is not None:
                self.callback( x )

            # Descent direction
            # Best to normalize, use gDn=1 in the options to achieve the
            # the normalization illustrated below.
            d = - gradJ#/max( abs( gradJ ))
 
            with open("fullCostFunction.txt", "a") as f:
                if iter == 0:
                    f.write(f"-1 \t {str(Jpast)}\n")
                f.write(f"{iter} \t {str(J)}\n")

            print(f"\nFWI iter {iter} - linesearch run {lineSearchIteration} - Full cost function : {J}\n")

            # Max num. iter is 100
            alpha = optimize.linesearch.line_search_wolfe1(f=self.func,fprime=self.jac,xk=x,pk=d,gfk=-d,old_fval=J,old_old_fval=Jpast,args=args, c1=1e-4,c2=0.9, amax=50, amin=1e-12,xtol=self.tol)
            
            # Fallback to line_search_wolfe2 if a bad step length is found.
            # Default max num. iter is 10.
            alpha = alpha[0]
            if alpha is None:
                print("WARNING: Fallback to line_search_wolfe2")
                alpha = optimize.linesearch.line_search_wolfe2(f=self.func,myfprime=self.jac,xk=x,pk=d,gfk=-d,old_fval=J,old_old_fval=Jpast,args=args,c1=1e-4,c2=0.9,amax=50)
                lineSearchIteration += 1
                alpha = alpha[0]
                
            
            lineSearchIteration += 1
            
            if alpha is None:
                print("ERROR: alpha is bad")
                break

            print("Calculated step length: ",alpha)

            # Update model
            xup = x + alpha * d[:]
            # Constraining model to bounds
            xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

            Jup = self.func( xup, *args )
            print("Jup: {}, J: {}".format(Jup,J))
            print("Tolerance: {}, J(i) - J(i-1): {} ".format(self.Jtol,np.abs(Jup - J)))
            if np.abs(Jup - J) < self.Jtol:
                x = xup
                print("Tolerance criteria satisfied.")
                break

            # Check if model perturbation is significant.
            # Using 1e7. This controls the accuracy, 1e12 provides low accuracy,
            # 1e7 moderate, and 10 extremely high (scipy Notes).
            epsmch = np.finfo(float).eps
            factr = 1e7*epsmch
            if (((x-xup)/max(np.max(abs(x)),np.max(abs(xup)),1)) <= factr).all():
                print("WARNING: model perturbation is too small, stopping.")
                break

            
            x = xup
            Jpast = J
            
            # Avoid dwelling for too long near a minima. SD is an expert in
            # doing that. It may go up and down the lobes of the objective
            # function, converging very slowly to the minimum.
            meanJ = np.mean( checkJ )

            # This second criteria will be a safeguard for when Jtol is too
            # small if the new value can only create a 5% variation in the mean of
            # all previous cost function values, we stop the iterations
            if Jup >= 0.95 * meanJ and Jup <= 1.05 * meanJ:
                print("Safeguard tolerance criteria satisfied.")
                break

            iter +=1
            checkJ[iter] = Jup

            if all(checkJ):
                print(f"\nWARNING: Maximum number of cost function evaluations reached. Stopping iterations. \n Last {self.nbCFcheck} cost function values: {checkJ}.")


        return x            


class CGPROptimizer2(Minimizer):
    """
    Class dedicated to the Polak-Ribiere algorithm of the conjugated gradient method. Uses a modified line search.
    """
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            options:
                alpha: float, optional
                    Starting line search step
                    Default is 1e-8
                maxiter : int, optional
                    Maximum number of iterations
                    Default is 10
        """
        super().__init__(func, "CGPR2", jac, hess, hessp, bounds, constraints, tol, callback, options)

        if options is None:
            options = {}
        
        self.alpha = options.get("alpha", 1e-8)
        self.p1 = options.get("p1", 1.5)
        self.p2 = options.get("p2", 0.5)
        self.minls = options.get("minls", 3)
        self.maxls = options.get("maxls", self.minls)
        self.miniter = options.get("miniter", 1)
        self.maxiter = options.get("maxiter", 10)
        self.nbCFcheck = options.get("nbCFcheck", 5)
        self.wolfeConditions = options.get("wolfeConditions", False)
        self.k1 = options.get("k1", 0.2)
        self.k2 = options.get("k2", 0.8)
        
        if self.maxls < self.minls:
            self.maxls = self.minls

        if self.maxiter < self.miniter:
            self.maxiter = self.miniter
        
    def optimize(self, x, args):
        """
        Optimize with conjugated gradient method

        Parameters
        -----------
            x : array
                Current value of the variable to optimize
            tol: float
                Tolerance controlling the linesearch. It corresponds to the
                relative tolerance for an acceptable step.
            gtol: float
                Tolerance controlling the convergence. Evaluates the relative
                change of the cost function.
            args: tuple
                List of func and jac callables arguments
        """
        iter = 0
        Jup = 0
        success = 0
        checkJ = np.zeros(self.nbCFcheck)
        beta = 0

        while iter < self.maxiter and not success:

            lineSearchIteration = 1
          
            if iter == 0:
                gradJ, J = self.jac( x, *args, computeFullCF=True )
                checkJ[0] = J
                d = - gradJ / max( abs( gradJ ))

            else:
                gradJ = self.jac( x, *args )
                beta = (gradJ + d).dot(gradJ)/(np.linalg.norm(-d)**2) # d = -gradJ_{i-1}
                beta = max(0,beta)
                d = -gradJ / max( abs( gradJ ) ) + beta*d
                J = Jup

 
            with open("fullCostFunction.txt", "a") as f:
                f.write(f"{iter} \t {str(J)}\n")

            print(f"\nFWI iter {iter} - linesearch iteration {lineSearchIteration} - Full cost function : {J}\n")

            while True:
                print("Current alpha: {}".format(self.alpha))
                xup = x + self.alpha * d[:]
                # Constraining model to bounds
                xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])
                
                Jup = self.func( xup, *args )

                if lineSearchIteration < self.minls and lineSearchIteration < self.maxls:
                    # Adapt alpha depending on Jup and J values
                    if Jup >= J:
                        # Increase alpha
                        self.alpha *= self.p2

                    else:
                        # Lower alpha
                        self.alpha *= self.p1

                    lineSearchIteration += 1

                # Min and max number of linesearch iter reached
                elif lineSearchIteration == self.minls and lineSearchIteration == self.maxls:
                    lineSearchIteration += 1

                    print( "*"*50 )
                    print( f"FWI iteration {iter} - Max number of linesearch iterations reached \n Last {self.nbCFcheck} full cost functions computed : {checkJ}\n" )
              
                    if not all( checkJ ):
                        if Jup >= J:
                            self.alpha *= 0.1 * self.p2**self.maxls
                            xup = x + self.alpha * d
                            
                            # Constraining model to bounds
                            xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

                        x = xup

                        if self.callback is not None:
                            self.callback( x )

                        iter += 1
                        checkJ[iter % self.nbCFcheck] = Jup

                        break

                    # If checkJ full of computed values
                    else:
                        mean = np.mean( checkJ )
                        # new value satisfying, changing the frequency
                        if Jup >= 0.95 * mean:
                            success = 1

                            break

                        else:
                            x = xup

                            if self.callback is not None:
                                self.callback( x )

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                # Min number of linesearch iter reached
                elif lineSearchIteration >= self.minls and lineSearchIteration < self.maxls:
                    lineSearchIteration += 1

                    if not all( checkJ ):
                        if Jup < J:
                            x = xup

                            if self.callback is not None:
                                self.callback(x)

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                        else:
                            self.alpha *= self.p2

                    else:
                        mean = np.mean( checkJ )
                        # It may be useful to take the last criteria out to avoid additional
                        # iterations. 
                        # Perhaps it should actually be an input threshold parameter, like how distant from 
                        # the mean can a value be.
                        if Jup >= 0.95 * mean: 
                            #and Jup <= 1.05 * mean:
                            success = 1

                            break

                        elif Jup < 0.95 * mean:
                            x = xup

                            if self.callback is not None:
                                self.callback(x)

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                        else:
                            self.alpha *= self.p2

                elif lineSearchIteration >= self.maxls:
                    self.alpha *= 0.1 * self.p2**self.maxls

                    xup = x + self.alpha * d
                    
                    # Constraining to bounds
                    xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

                    x = xup

                    if self.callback is not None:
                        self.callback( x )

                    iter += 1
                    checkJ[iter % self.nbCFcheck] = Jup

                    break
            
            # Ensure a minimum amount of iterations
            if iter < self.miniter:
                if success:
                    success = 0
                    iter += 1
                
        return x


class SteepestDescentOptimizer(Minimizer):
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            options:
                alpha: float, optional
                    Starting line search step
                    Default is 1e-8
                p1 : float, optional
                    Coefficient for line search step re-evaluation if the step is to low
                    Default is 1.5
                p2 : float, optional
                    Coefficient for line search step re-evaluation if the step is to large
                    Default is 0.5
                minls : int, optional
                    Minimum number of evaluations in line search
                    Default is 3
                maxls : int, optional
                    Maximum number of evaluations in line search
                    Default is the value of `minls` parameter
                miniter : int, optional
                    Minimum number of iterations
                    Default is 1
                maxiter : int, optional
                    Maximum number of iterations
                    Default is 10
                nbCfcheck : int, optional
                    Number of previous cost function evaluation to check if the functionnal decreases enough
                    Default is 5
                wolfeConditions: bool, optional
                    Activate Wolfe conditions
                    Default is False
                k1 : float, optional
                    Coefficient for lower bound for wolfe conditions
                    Default is 0.2
                k2 : float, optional
                    Coefficient for upper bound for wolfe conditions
                    Default is 0.8
        """
        super().__init__(func, "steepestDescent", jac, hess, hessp, bounds, constraints, tol, callback, options)

        if options is None:
            options = {}
        
        self.alpha = options.get("alpha", 1e-8)
        self.p1 = options.get("p1", 1.5)
        self.p2 = options.get("p2", 0.5)
        self.minls = options.get("minls", 3)
        self.maxls = options.get("maxls", self.minls)
        self.miniter = options.get("miniter", 1)
        self.maxiter = options.get("maxiter", 10)
        self.nbCFcheck = options.get("nbCFcheck", 5)
        self.wolfeConditions = options.get("wolfeConditions", False)
        self.k1 = options.get("k1", 0.2)
        self.k2 = options.get("k2", 0.8)
        
        if self.maxls < self.minls:
            self.maxls = self.minls

        if self.maxiter < self.miniter:
            self.maxiter = self.miniter

    def optimize(self, x, args):
        """
        Optimize with steepest descent method

        Parameters
        -----------
            x : array
                Current value of the variable to optimize
            args: tuple
                List of func and jac callables arguments
        """
        iter = 0
        Jup = 0
        success = 0
        checkJ = np.zeros(self.nbCFcheck)

        while iter < self.maxiter and not success:

            lineSearchIteration = 1
          
            if iter == 0:
                gradJ, J = self.jac( x, *args, computeFullCF=True )
                checkJ[0] = J

            else:
                gradJ = self.jac( x, *args )
                J = Jup

            d = - gradJ / max( abs( gradJ ))
 
            with open("fullCostFunction.txt", "a") as f:
                f.write(f"{iter} \t {str(J)}\n")

            print(f"\nFWI iter {iter} - linesearch iteration {lineSearchIteration} - Full cost function : {J}\n")

            while True:
                print("Current alpha: {}".format(self.alpha))
                xup = x + self.alpha * d[:]
                # Constraining model to bounds
                imin = np.where( xup < self.bounds[0][0] )[0]
                imax = np.where( xup > self.bounds[0][1] )[0]
                xup[imin] = self.bounds[0][0]
                xup[imax] = self.bounds[0][1]
                
                Jup = self.func( xup, *args )

                if lineSearchIteration < self.minls and lineSearchIteration < self.maxls:
                    # Adapt alpha depending on Jup and J values
                    if Jup >= J:
                        # Increase alpha
                        self.alpha *= self.p2

                    else:
                        # Lower alpha
                        self.alpha *= self.p1

                    lineSearchIteration += 1

                # Min and max number of linesearch iter reached
                elif lineSearchIteration == self.minls and lineSearchIteration == self.maxls:
                    lineSearchIteration += 1

                    print( "*"*50 )
                    print( f"FWI iteration {iter} - Max number of linesearch iterations reached \n Last {self.nbCFcheck} full cost functions computed : {checkJ}\n" )
              
                    if not all( checkJ ):
                        if Jup >= J:
                            self.alpha *= 0.1 * self.p2**self.maxls
                            xup = x + self.alpha * d
                            
                            # Constraining model to bounds
                            imin = np.where( xup < self.bounds[0][0] )[0]
                            imax = np.where( xup > self.bounds[0][1] )[0]
                            xup[imin] = self.bounds[0][0]
                            xup[imax] = self.bounds[0][1]

                        x = xup

                        if self.callback is not None:
                            self.callback( x )

                        iter += 1
                        checkJ[iter % self.nbCFcheck] = Jup

                        break

                    # If checkJ full of computed values
                    else:
                        mean = np.mean( checkJ )
                        # new value satisfying, changing the frequency
                        if Jup >= 0.95 * mean:
                            success = 1

                            break

                        else:
                            x = xup

                            if self.callback is not None:
                                self.callback( x )

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                # Min number of linesearch iter reached
                elif lineSearchIteration >= self.minls and lineSearchIteration < self.maxls:
                    lineSearchIteration += 1

                    if not all( checkJ ):
                        if Jup < J:
                            x = xup

                            if self.callback is not None:
                                self.callback(x)

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                        else:
                            self.alpha *= self.p2

                    else:
                        mean = np.mean( checkJ )
                        # It may be useful to take the last criteria out to avoid additional
                        # iterations. 
                        # Perhaps it should actually be an input threshold parameter, like how distant from 
                        # the mean can a value be.
                        if Jup >= 0.95 * mean: 
                            #and Jup <= 1.05 * mean:
                            success = 1

                            break

                        elif Jup < 0.95 * mean:
                            x = xup

                            if self.callback is not None:
                                self.callback(x)

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                        else:
                            self.alpha *= self.p2

                elif lineSearchIteration >= self.maxls:
                    self.alpha *= 0.1 * self.p2**self.maxls

                    xup = x + self.alpha * d
                    
                    # Constraining to bounds
                    imin = np.where( xup < self.bounds[0][0] )[0]
                    imax = np.where( xup > self.bounds[0][1] )[0]
                    xup[imin] = self.bounds[0][0]
                    xup[imax] = self.bounds[0][1]

                    x = xup

                    if self.callback is not None:
                        self.callback( x )

                    iter += 1
                    checkJ[iter % self.nbCFcheck] = Jup

                    break
            
            # Ensure a minimum amount of iterations
            if iter < self.miniter:
                if success:
                    success = 0
                    iter += 1
                
        return x


    def WolfeConditions(self, J, Jup, d, gradJ):
        """
        Check Wolfe conditions
       
        Parameters
        -----------
            J : float   
                Value of the cost function
            Jup : float
                Value of the cost function for perturbed model
            d : array
                Descent direction
        
        Returns
        --------
            bool :
                0 if conditions not verified, 1 otherwise
        """
        succesArmijo = self.armijoCondition(J, Jup, d, gradJ)

        if succesArmijo:
            succesGoldstein = self.goldsteinCondition(J, Jup, d, gradJ)
        
            if succesGoldstein:
                return 1
            else:
                self.alpha *= self.p1
                return 0
        else:
            self.alpha *= self.p2
            return 0


    def armijoCondition(self, J, Jm, d, gradJ):
        """
        Check Armijo condition

        Parameters
        ----------
            J : float
                Value of the cost function
            Jm : float
                Value of the cost function for perturbed model
            d : array
                Descent direction
            gradJ : array
                Gradient of the cost function

        Returns
        --------
            success : bool
                0 if test fails, 1 if it passes
        """
        success = (Jm <= J + self.k1 * self.alpha * np.dot(d, gradJ))
        
        return success


    def goldsteinCondition(self, J, Jm, d, gradJ):
        """
        Check Goldstein condition

       Parameters
        ----------
            J : float
                Value of the cost function
            Jm : float
                Value of the cost function for perturbed model
            d : array
                Descent direction
            gradJ : array
                Gradient of the cost function

        Returns
        --------
            success : bool
                0 if test fails, 1 if it passes
        """
        success = (Jm >= J + self.k2 * self.alpha * np.dot(d, gradJ))
        
        return success
