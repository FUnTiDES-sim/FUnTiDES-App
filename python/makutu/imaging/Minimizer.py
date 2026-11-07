"""
Optimisers for the full-waveform inversion.

``func(x, *args)`` returns the cost function and ``jac(x, *args, computeFullCF=False)``
the gradient, or ``(gradient, J)`` with ``computeFullCF=True`` (``makutu.forwardBackward.func``
and ``makutu.imaging.gradf``; the gradient run gives J at no extra cost). Every method
returns the final model.
"""

import numpy as np

from scipy import optimize


class Minimizer:
    """
    Minimisation of a scalar function of the model, with the methods below or any
    scipy.optimize.minimize method:

    - "steepestDescent": steepest descent, heuristic line search (SteepestDescentOptimizer)
    - "SD2": steepest descent, Wolfe line search (SteepestDescentOptimizer2)
    - "CGPR": Polak-Ribiere conjugate gradient, Wolfe line search (CGPROptimizer)
    - "CGPR2": Polak-Ribiere conjugate gradient, heuristic line search (CGPROptimizer2)

    The Wolfe line searches and the scipy methods need the true gradient: no
    normalisation of the gradient (gDn = 0 in the gradf arguments).

    See the scipy.optimize documentation for the other parameters:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    def __init__(self, func, method="steepestDescent", jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            func : callable
                Method returning the cost function value
            method : str
                Optimization method \
                Default is 'steepestDescent'
            jac : callable, optional
                Method returning the gradient of the functional (see the module docstring)
            hess, hessp : callable, optional
                Hessian (scipy methods only)
            bounds : list, optional
                (min, max) pairs, one for each element of x, or a single pair for all
            constraints : {Constraint, dict} or list of {Constraint, dict}
                Constraints (scipy methods only)
            tol : float, optional
                Tolerance for termination (scipy methods)
            callback : callable, optional
                Called after each iteration with the current model: 'callback(xk)'
            options : dict, optional
                Options of the method (see each optimiser class, or scipy.optimize)
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
        self.options = options if options is not None else {}

    def _bounds(self):
        """Lower and upper bounds (scalars or arrays), or (None, None)."""
        if self.bounds is None or len(self.bounds) == 0:
            return None, None
        b = np.asarray(self.bounds, dtype=float).reshape(-1, 2)
        lo, hi = b[:, 0], b[:, 1]
        if np.all(lo == lo[0]) and np.all(hi == hi[0]):
            return lo[0], hi[0]
        return lo, hi

    def _clip(self, x):
        """Project the model on the bounds."""
        lo, hi = self._bounds()
        return x if lo is None else np.clip(x, lo, hi)

    def _initialStep(self, x):
        """Starting step along a direction of unit maximum: option 'alpha', or 1 % of max|x|."""
        alpha = self.options.get("alpha")
        return float(alpha) if alpha is not None else 0.01 * float(np.max(np.abs(x)))

    def _iterate(self, x):
        if self.callback is not None:
            self.callback(x)

    def optimize(self, x, args):
        """
        Call the optimization method

        Parameters:
        -----------
            x : array
                Starting model
            args: tuple
                Arguments of func and jac

        Returns:
        --------
            np.ndarray : final model
        """
        x = np.array(x, dtype=float)
        classes = {"steepestDescent": SteepestDescentOptimizer, "SD2": SteepestDescentOptimizer2,
                   "CGPR": CGPROptimizer, "CGPR2": CGPROptimizer2}
        if self.method in classes:
            return classes[self.method](self.func, self.jac, self.hess, self.hessp, self.bounds, self.constraints,
                                        self.tol, self.callback, self.options).optimize(x, args)
        result = optimize.minimize(self.func, x, args=args, method=self.method, jac=self.jac, hess=self.hess,
                                   hessp=self.hessp, bounds=self.bounds, constraints=self.constraints,
                                   tol=self.tol, callback=self.callback, options=self.options)
        return result.x


def _logCost(it, J):
    with open("fullCostFunction.txt", "a") as f:
        f.write(f"{it} \t {str(J)}\n")


class _WolfeOptimizer(Minimizer):
    """
    Common part of the optimisers with a Wolfe line search (scipy.optimize.line_search).

    The descent directions are scaled by a constant, -scale * gradient, so that a unit
    step moves the model by the starting step (option alpha, default 1 % of max|x|):
    the line search starts at step 1 and goes up to amax.
    """

    def _lineSearch(self, x, d, gradJ, J, Jpast, args):
        """Step length along d satisfying the strong Wolfe conditions, or None."""
        res = optimize.line_search(self.func, self.jac, x, d, gfk=gradJ, old_fval=J, old_old_fval=Jpast,
                                   args=args, c1=1e-4, c2=0.9, amax=self.amax, maxiter=self.lsMaxiter)
        return res[0]

    def _start(self, x, args):
        """J and gradient at x, and the direction scale; returns (None, x, J, gradient)."""
        gradJ, J = self.jac(x, *args, computeFullCF=True)
        self.scale = self._initialStep(x) / np.max(np.abs(gradJ))
        return None, x, J, gradJ

    @staticmethod
    def _tooSmall(x, xup):
        """The model perturbation is negligible (scipy L-BFGS-B criterion, factr 1e7)."""
        factr = 1e7 * np.finfo(float).eps
        return bool(np.all(np.abs(x - xup) / max(np.max(np.abs(x)), np.max(np.abs(xup)), 1) <= factr))


class CGPROptimizer(_WolfeOptimizer):
    """
    Polak-Ribiere conjugate gradient with a Wolfe line search, close to scipy.optimize's CG.
    """
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            options:
                alpha : float, optional
                    Model change of a unit step (largest element) \
                    Default is 1 % of max|x|
                maxiter : int, optional
                    Maximum number of iterations \
                    Default is 10
                gtol : float, optional
                    Stop when |J(i) - J(i-1)| < gtol \
                    Default is 1e-4
                amax : float, optional
                    Maximum step length of the line search \
                    Default is 50
                lsMaxiter : int, optional
                    Maximum number of line search iterations \
                    Default is 10
        """
        super().__init__(func, "CGPR", jac, hess, hessp, bounds, constraints, tol, callback, options)
        self.maxiter = self.options.get("maxiter", 10)
        self.gtol = self.options.get("gtol", 0.0001)
        self.amax = self.options.get("amax", 50)
        self.lsMaxiter = self.options.get("lsMaxiter", 10)

    def optimize(self, x, args):
        """Optimize with the conjugate gradient method; returns the final model."""
        Jpast, x, J, gradJ = self._start(x, args)
        d = -self.scale * gradJ
        for it in range(self.maxiter):
            self._iterate(x)
            _logCost(it, J)
            print(f"\nFWI iter {it} - Full cost function : {J}\n")

            step = self._lineSearch(x, d, gradJ, J, Jpast, args)
            if step is None:
                print("ERROR: no step satisfying the Wolfe conditions")
                break
            print("Calculated step length: ", step)

            xup = self._clip(x + step * d)
            gradUp, Jup = self.jac(xup, *args, computeFullCF=True)
            print(f"Jup: {Jup}, J: {J}, |J(i) - J(i-1)|: {abs(Jup - J)} (tolerance {self.gtol})")
            if abs(Jup - J) < self.gtol:
                x = xup
                print("Tolerance criteria satisfied.")
                break
            if self._tooSmall(x, xup):
                print("WARNING: model perturbation is too small, stopping.")
                break

            chi = max(0.0, (gradUp - gradJ).dot(gradUp) / np.linalg.norm(gradJ) ** 2)   # Polak-Ribiere+
            d = -self.scale * gradUp + chi * d
            x, gradJ, Jpast, J = xup, gradUp, J, Jup
        return x


class SteepestDescentOptimizer2(_WolfeOptimizer):
    """Steepest descent with a Wolfe line search."""

    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            options:
                alpha : float, optional
                    Model change of a unit step (largest element) \
                    Default is 1 % of max|x|
                maxiter : int, optional
                    Maximum number of iterations \
                    Default is 10
                Jtol : float, optional
                    Stop when |J(i) - J(i-1)| < Jtol \
                    Default is 1e-4
                nbCFcheck : int, optional
                    Number of previous cost functions to check if the cost decreases enough \
                    Default is 5
                amax, lsMaxiter : as CGPROptimizer
        """
        super().__init__(func, "SD2", jac, hess, hessp, bounds, constraints, tol, callback, options)
        self.maxiter = self.options.get("maxiter", 10)
        self.nbCFcheck = self.options.get("nbCFcheck", 5)
        self.Jtol = self.options.get("Jtol", 0.0001)
        self.amax = self.options.get("amax", 50)
        self.lsMaxiter = self.options.get("lsMaxiter", 10)

    def optimize(self, x, args):
        """Optimize with the steepest descent method; returns the final model."""
        Jpast, x, J, gradJ = self._start(x, args)
        checkJ = [J]
        for it in range(self.maxiter):
            self._iterate(x)
            _logCost(it, J)
            print(f"\nFWI iter {it} - Full cost function : {J}\n")

            d = -self.scale * gradJ
            step = self._lineSearch(x, d, gradJ, J, Jpast, args)
            if step is None:
                print("ERROR: no step satisfying the Wolfe conditions")
                break
            print("Calculated step length: ", step)

            xup = self._clip(x + step * d)
            gradUp, Jup = self.jac(xup, *args, computeFullCF=True)
            print(f"Jup: {Jup}, J: {J}, |J(i) - J(i-1)|: {abs(Jup - J)} (tolerance {self.Jtol})")
            if abs(Jup - J) < self.Jtol:
                x = xup
                print("Tolerance criteria satisfied.")
                break
            if self._tooSmall(x, xup):
                print("WARNING: model perturbation is too small, stopping.")
                break

            # Steepest descent may creep along a valley: stop when J stays within 5 % of the
            # mean of the last nbCFcheck values
            meanJ = np.mean(checkJ[-self.nbCFcheck:])
            x, gradJ, Jpast, J = xup, gradUp, J, Jup
            if len(checkJ) >= self.nbCFcheck and 0.95 * meanJ <= Jup <= 1.05 * meanJ:
                print("Safeguard tolerance criteria satisfied.")
                break
            checkJ.append(Jup)
        return x


class _HeuristicOptimizer(Minimizer):
    """
    Common part of the optimisers with the heuristic line search: along a direction of
    unit maximum, minls to maxls trial steps (the step grows by p1 after a decrease of J
    and shrinks by p2 after an increase); the iterations stop when J stays above 95 % of
    the mean of the last nbCFcheck accepted values.
    """
    def __init__(self, func, method, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        super().__init__(func, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
        o = self.options
        self.alpha = o.get("alpha")
        self.p1 = o.get("p1", 1.5)
        self.p2 = o.get("p2", 0.5)
        self.minls = o.get("minls", 3)
        self.maxls = max(o.get("maxls", self.minls), self.minls)
        self.miniter = o.get("miniter", 1)
        self.maxiter = max(o.get("maxiter", 10), self.miniter)
        self.nbCFcheck = o.get("nbCFcheck", 5)
        self.wolfeConditions = o.get("wolfeConditions", False)
        self.k1 = o.get("k1", 0.2)
        self.k2 = o.get("k2", 0.8)

    def _direction(self, gradJ, it):
        raise NotImplementedError

    def optimize(self, x, args):
        """Optimize; returns the final model."""
        if self.alpha is None:
            self.alpha = self._initialStep(x)
        it = 0
        success = 0
        checkJ = np.zeros(self.nbCFcheck)

        while it < self.maxiter and not success:
            lineSearchIteration = 1
            # J at the current model, with its gradient (same forward runs)
            gradJ, J = self.jac(x, *args, computeFullCF=True)
            if it == 0:
                checkJ[0] = J
            d = self._direction(gradJ, it)

            _logCost(it, J)
            print(f"\nFWI iter {it} - linesearch iteration {lineSearchIteration} - Full cost function : {J}\n")

            while True:
                print("Current alpha: {}".format(self.alpha))
                xup = self._clip(x + self.alpha * d)
                Jup = self.func(xup, *args)

                if lineSearchIteration < self.minls and lineSearchIteration < self.maxls:
                    # J increased: shorter step; J decreased: longer step
                    self.alpha *= self.p2 if Jup >= J else self.p1
                    lineSearchIteration += 1

                # Min and max number of linesearch iter reached
                elif lineSearchIteration == self.minls and lineSearchIteration == self.maxls:
                    lineSearchIteration += 1
                    print("*" * 50)
                    print(f"FWI iteration {it} - Max number of linesearch iterations reached \n "
                          f"Last {self.nbCFcheck} full cost functions computed : {checkJ}\n")
                    if not all(checkJ):
                        if Jup >= J:
                            # no decrease: a much smaller step (J is computed again next iteration)
                            self.alpha *= 0.1 * self.p2 ** self.maxls
                            xup = self._clip(x + self.alpha * d)
                        x = xup
                        self._iterate(x)
                        it += 1
                        checkJ[it % self.nbCFcheck] = Jup
                        break
                    # checkJ full of computed values
                    mean = np.mean(checkJ)
                    if Jup >= 0.95 * mean:          # no more progress at this frequency
                        success = 1
                        break
                    x = xup
                    self._iterate(x)
                    it += 1
                    checkJ[it % self.nbCFcheck] = Jup
                    break

                # Min number of linesearch iter reached
                elif self.minls <= lineSearchIteration < self.maxls:
                    lineSearchIteration += 1
                    if not all(checkJ):
                        if Jup < J:
                            x = xup
                            self._iterate(x)
                            it += 1
                            checkJ[it % self.nbCFcheck] = Jup
                            break
                        self.alpha *= self.p2
                    else:
                        mean = np.mean(checkJ)
                        if Jup >= 0.95 * mean:
                            success = 1
                            break
                        x = xup
                        self._iterate(x)
                        it += 1
                        checkJ[it % self.nbCFcheck] = Jup
                        break

                else:   # lineSearchIteration >= maxls
                    self.alpha *= 0.1 * self.p2 ** self.maxls
                    x = self._clip(x + self.alpha * d)
                    self._iterate(x)
                    it += 1
                    checkJ[it % self.nbCFcheck] = Jup
                    break

            # Ensure a minimum amount of iterations
            if it < self.miniter and success:
                success = 0
                it += 1

        return x

    def WolfeConditions(self, J, Jup, d, gradJ):
        """
        Check Wolfe conditions (Armijo and Goldstein); adapts alpha when they fail.

        Returns
        --------
            bool : 0 if conditions not verified, 1 otherwise
        """
        if self.armijoCondition(J, Jup, d, gradJ):
            if self.goldsteinCondition(J, Jup, d, gradJ):
                return 1
            self.alpha *= self.p1
            return 0
        self.alpha *= self.p2
        return 0

    def armijoCondition(self, J, Jm, d, gradJ):
        """Armijo condition: sufficient decrease."""
        return Jm <= J + self.k1 * self.alpha * np.dot(d, gradJ)

    def goldsteinCondition(self, J, Jm, d, gradJ):
        """Goldstein condition: step not too short."""
        return Jm >= J + self.k2 * self.alpha * np.dot(d, gradJ)


class SteepestDescentOptimizer(_HeuristicOptimizer):
    """
    Steepest descent with the heuristic line search.

    options:
        alpha : float, optional
            Starting line search step along -gradient / max|gradient| \
            Default is 1 % of max|x|
        p1 : float, optional
            Factor of the step after a decrease of J \
            Default is 1.5
        p2 : float, optional
            Factor of the step after an increase of J \
            Default is 0.5
        minls : int, optional
            Minimum number of evaluations in line search \
            Default is 3
        maxls : int, optional
            Maximum number of evaluations in line search \
            Default is the value of `minls` parameter
        miniter : int, optional
            Minimum number of iterations \
            Default is 1
        maxiter : int, optional
            Maximum number of iterations \
            Default is 10
        nbCFcheck : int, optional
            Number of previous cost function evaluations to check if the functional decreases enough \
            Default is 5
    """
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        super().__init__(func, "steepestDescent", jac, hess, hessp, bounds, constraints, tol, callback, options)

    def _direction(self, gradJ, it):
        return -gradJ / np.max(np.abs(gradJ))


class CGPROptimizer2(_HeuristicOptimizer):
    """
    Polak-Ribiere conjugate gradient with the heuristic line search (options of
    SteepestDescentOptimizer).
    """
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        super().__init__(func, "CGPR2", jac, hess, hessp, bounds, constraints, tol, callback, options)
        self._gradPast = None
        self._dPast = None

    def _direction(self, gradJ, it):
        g = gradJ / np.max(np.abs(gradJ))
        if self._gradPast is None:
            d = -g
        else:
            chi = max(0.0, (g - self._gradPast).dot(g) / np.linalg.norm(self._gradPast) ** 2)   # Polak-Ribiere+
            d = -g + chi * self._dPast
        self._gradPast, self._dPast = g, d
        return d
