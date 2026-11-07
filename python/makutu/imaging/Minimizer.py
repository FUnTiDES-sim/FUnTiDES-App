"""
Optimisers for the full-waveform inversion.

``func(x, *args)`` returns the cost function and ``jac(x, *args, computeFullCF=False)``
the gradient, or ``(gradient, J)`` with ``computeFullCF=True`` (``makutu.forwardBackward.func``
and ``makutu.imaging.gradf``; the gradient run gives J at no extra cost). Every method
returns the final model.
"""

import numpy as np

from scipy import optimize

from . import globals as gb


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

    ``precond`` (methods of this module only): diagonal preconditioner applied to the
    gradient to build the descent direction, ``precond(gradient) -> direction``
    (e.g. :class:`makutu.imaging.GradientConditioner`: masks and illumination). The
    cost function and the line searches still use the true gradient.

    See the scipy.optimize documentation for the other parameters:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    def __init__(self, func, method="steepestDescent", jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, precond=None):
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
            precond : callable, optional
                Preconditioner of the gradient (see the class docstring) \
                Default is None (identity)
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
        self.precond = precond

    def _precond(self, gradJ):
        """Preconditioned gradient (identity without precond)."""
        return gradJ if self.precond is None else self.precond(gradJ)

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
                                        self.tol, self.callback, self.options, self.precond).optimize(x, args)
        if self.precond is not None:
            raise ValueError(f"precond is not available with the scipy method {self.method}")
        result = optimize.minimize(self.func, x, args=args, method=self.method, jac=self.jac, hess=self.hess,
                                   hessp=self.hessp, bounds=self.bounds, constraints=self.constraints,
                                   tol=self.tol, callback=self.callback, options=self.options)
        return result.x


def _logCost(it, J):
    with open("fullCostFunction.txt", "a") as f:
        f.write(f"{it} \t {str(J)}\n")


class _WolfeOptimizer(Minimizer):
    """
    Common part of the optimisers with a Wolfe line search (scipy.optimize.line_search),
    which starts at step 1 and accepts it when it satisfies the strong Wolfe conditions.
    The descent direction is therefore scaled so that step 1 is the expected step:
    at the first iteration, a model change of the starting step (option alpha, default
    1 % of max|x|) on the largest element; then the step whose first-order decrease is
    twice the last decrease of J (Nocedal & Wright, eq. 3.60), at most 10 times the
    last model change.

    Evaluations: the line search asks for J and then for the gradient at the trial steps;
    by default (option fWithGradient) J is computed with the gradient (one forward run with
    snapshots and one adjoint run), and the gradient at the accepted step is reused for the
    next iteration: about two forward runs per iteration when the first step is accepted,
    instead of five. With fWithGradient=False, J alone is computed first (one forward run),
    which is cheaper when the first trial steps are rejected.
    """

    def _options(self, method):
        o = self.options
        self.maxiter = o.get("maxiter", 10)
        self.Jtol = o.get("Jtol", o.get("gtol", 1e-4))
        self.nbCFcheck = o.get("nbCFcheck", 5)
        self.amax = o.get("amax", 50)
        self.lsMaxiter = o.get("lsMaxiter", 10)
        self.fWithGradient = o.get("fWithGradient", True)
        self._lastChange = None
        self._cache = []                     # (x, J, gradient or None, illumination), latest last
        self.nEval = {"J": 0, "gradient": 0}

    # ------------------------------------------------------------------ evaluations
    def _lookup(self, x):
        for entry in reversed(self._cache):
            if entry[0].shape == x.shape and np.array_equal(entry[0], x):
                return entry
        return None

    def _store(self, x, J, gradJ):
        self._cache = [e for e in self._cache if not np.array_equal(e[0], x)][-3:]
        self._cache.append((np.array(x, copy=True), J, gradJ, getattr(gb, "illumination", None)))

    def _fg(self, x, *args):
        """(J, gradient) at x, from the cache or from jac."""
        entry = self._lookup(x)
        if entry is not None and entry[2] is not None:
            if entry[3] is not None:
                gb.illumination = entry[3]       # for the preconditioner
            return entry[1], entry[2]
        gradJ, J = self.jac(x, *args, computeFullCF=True)
        self.nEval["gradient"] += 1
        self._store(x, J, gradJ)
        return J, gradJ

    def _f(self, x, *args):
        """J at x (computed with the gradient if fWithGradient)."""
        entry = self._lookup(x)
        if entry is not None:
            return entry[1]
        if self.fWithGradient:
            return self._fg(x, *args)[0]
        J = self.func(x, *args)
        self.nEval["J"] += 1
        self._store(x, J, None)
        return J

    def _g(self, x, *args):
        return self._fg(x, *args)[1]

    def _scaled(self, x, d, gradJ, J, Jpast):
        """Direction d scaled so that step 1 is the expected step."""
        dmax = np.max(np.abs(d))
        if Jpast is None or not Jpast > J:
            return d * (self._initialStep(x) / dmax)
        c = 2.02 * (Jpast - J) / -np.dot(gradJ, d)
        if self._lastChange is not None:
            c = min(c, 10.0 * self._lastChange / dmax)
        return c * d

    def _lineSearch(self, x, d, gradJ, J, args):
        """Step length along d satisfying the strong Wolfe conditions, or None."""
        res = optimize.line_search(self._f, self._g, x, d, gfk=gradJ, old_fval=J, old_old_fval=None,
                                   args=args, c1=1e-4, c2=0.9, amax=self.amax, maxiter=self.lsMaxiter)
        return res[0]

    @staticmethod
    def _tooSmall(x, xup):
        """The model perturbation is negligible (scipy L-BFGS-B criterion, factr 1e7)."""
        factr = 1e7 * np.finfo(float).eps
        return bool(np.all(np.abs(x - xup) / max(np.max(np.abs(x)), np.max(np.abs(xup)), 1) <= factr))

    def _direction(self, gradJ, it):
        raise NotImplementedError

    def optimize(self, x, args):
        """Optimize; returns the final model."""
        J, gradJ = self._fg(x, *args)
        J0, Jpast = J, None
        history = [J]
        for it in range(self.maxiter):
            _logCost(it, J)
            print(f"\nFWI iter {it} - Full cost function : {J}\n")

            d = self._scaled(x, self._direction(gradJ, it), gradJ, J, Jpast)
            if np.dot(gradJ, d) >= 0:                    # not a descent direction: restart
                print("WARNING: not a descent direction, restarting along the gradient")
                self._dRaw = None
                d = self._scaled(x, self._direction(gradJ, it), gradJ, J, Jpast)
            step = self._lineSearch(x, d, gradJ, J, args)
            if step is None:
                print("ERROR: no step satisfying the Wolfe conditions")
                break
            print("Calculated step length: ", step)

            xup = self._clip(x + step * d)
            Jup, gradUp = self._fg(xup, *args)
            print(f"Jup: {Jup}, J: {J}, |J(i) - J(i-1)| / J(0): {abs(Jup - J) / J0:.3e} (tolerance {self.Jtol})")
            print(f"Evaluations so far: {self.nEval['gradient']} with gradient, {self.nEval['J']} of J alone")
            if abs(Jup - J) < self.Jtol * J0:
                x, J = xup, Jup
                self._iterate(x)
                print("Tolerance criteria satisfied.")
                break
            if self._tooSmall(x, xup):
                print("WARNING: model perturbation is too small, stopping.")
                break
            self._lastChange = float(np.max(np.abs(xup - x)))
            self._gradPast = gradJ
            x, gradJ, Jpast, J = xup, gradUp, J, Jup
            self._iterate(x)                             # callback: model after the update
            # stop when J stays within 5 % of the mean of the last nbCFcheck values
            meanJ = np.mean(history[-self.nbCFcheck:])
            history.append(J)
            if len(history) > self.nbCFcheck and 0.95 * meanJ <= J:
                print("Safeguard tolerance criteria satisfied.")
                break
        else:
            _logCost(self.maxiter, J)                    # J of the final model
        return x


class CGPROptimizer(_WolfeOptimizer):
    """
    Polak-Ribiere+ conjugate gradient (preconditioned with precond) with a Wolfe
    line search.

    options:
        alpha : float, optional
            Model change of the first step on the largest element \
            Default is 1 % of max|x|
        maxiter : int, optional
            Maximum number of iterations \
            Default is 10
        Jtol (or gtol) : float, optional
            Stop when |J(i) - J(i-1)| < Jtol J(0) \
            Default is 1e-4
        fWithGradient : bool, optional
            Compute J with the gradient in the line search (see _WolfeOptimizer) \
            Default is True
        nbCFcheck : int, optional
            Stop when J stays above 95 % of the mean of the last nbCFcheck values \
            Default is 5
        amax : float, optional
            Maximum step length of the line search (relative to the expected step) \
            Default is 50
        lsMaxiter : int, optional
            Maximum number of line search iterations \
            Default is 10
    """
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, precond=None):
        super().__init__(func, "CGPR", jac, hess, hessp, bounds, constraints, tol, callback, options, precond)
        self._options("CGPR")
        self._dRaw = None
        self._gradPast = None
        self._pgPast = None

    def _direction(self, gradJ, it):
        """d = -P g + beta d_past, beta = (g - g_past).P g / g_past.P g_past (PR+)."""
        pg = self._precond(gradJ)
        if self._dRaw is None or self._gradPast is None:
            d = -pg
        else:
            beta = max(0.0, np.dot(gradJ - self._gradPast, pg) / np.dot(self._gradPast, self._pgPast))
            d = -pg + beta * self._dRaw
        self._dRaw, self._pgPast = d, pg
        return d


class SteepestDescentOptimizer2(_WolfeOptimizer):
    """Steepest descent (preconditioned with precond) with a Wolfe line search; options
    of CGPROptimizer."""

    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, precond=None):
        super().__init__(func, "SD2", jac, hess, hessp, bounds, constraints, tol, callback, options, precond)
        self._options("SD2")

    def _direction(self, gradJ, it):
        return -self._precond(gradJ)


class _HeuristicOptimizer(Minimizer):
    """
    Common part of the optimisers with the heuristic line search, along a direction of
    unit maximum: minls to maxls trial steps (the step grows by p1 after a decrease of J
    and shrinks by p2 after an increase), and the best trial is kept if it lowers J.
    Without decrease, the step is divided by 10 up to maxShrink times. The iterations
    stop when no step lowers J, or when J stays above 95 % of the mean of the last
    nbCFcheck values (after miniter iterations).
    """
    def __init__(self, func, method, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, precond=None):
        super().__init__(func, method, jac, hess, hessp, bounds, constraints, tol, callback, options, precond)
        o = self.options
        self.alpha = o.get("alpha")
        self.p1 = o.get("p1", 1.5)
        self.p2 = o.get("p2", 0.5)
        self.minls = max(o.get("minls", 3), 1)
        self.maxls = max(o.get("maxls", self.minls), self.minls)
        self.maxShrink = o.get("maxShrink", 3)
        self.miniter = o.get("miniter", 1)
        self.maxiter = max(o.get("maxiter", 10), self.miniter)
        self.nbCFcheck = o.get("nbCFcheck", 5)
        self.wolfeConditions = o.get("wolfeConditions", False)
        self.k1 = o.get("k1", 0.2)
        self.k2 = o.get("k2", 0.8)

    def _direction(self, gradJ, it):
        raise NotImplementedError

    def _lineSearch(self, x, d, J, args):
        """Best (J, step) of the trial steps along d; J is not lowered if the J returned >= J."""
        trials = []
        alpha = self.alpha
        for ls in range(self.maxls):
            print(f"Line search {ls + 1}: alpha {alpha:g}")
            Jup = self.func(self._clip(x + alpha * d), *args)
            trials.append((Jup, alpha))
            if ls + 1 >= self.minls and min(trials)[0] < J:
                break
            alpha *= self.p2 if Jup >= J else self.p1
        best = min(trials)
        shrink = min(a for _, a in trials)
        for _ in range(self.maxShrink):
            if best[0] < J:
                break
            shrink *= 0.1
            print(f"No decrease of J: alpha {shrink:g}")
            best = min(best, (self.func(self._clip(x + shrink * d), *args), shrink))
        return best

    def optimize(self, x, args):
        """Optimize; returns the final model."""
        if self.alpha is None:
            self.alpha = self._initialStep(x)
        history = []
        for it in range(self.maxiter):
            # J at the current model, with its gradient (same forward runs)
            gradJ, J = self.jac(x, *args, computeFullCF=True)
            history.append(J)
            _logCost(it, J)
            print(f"\nFWI iter {it} - Full cost function : {J}\n")
            if it >= self.miniter and len(history) > self.nbCFcheck and \
                    J >= 0.95 * np.mean(history[-self.nbCFcheck - 1:-1]):
                print(f"J above 95 % of the mean of the last {self.nbCFcheck} values: stopping")
                break

            d = self._direction(gradJ, it)
            Jup, alpha = self._lineSearch(x, d, J, args)
            if Jup >= J:
                print("No step lowers J: stopping")
                break
            x = self._clip(x + alpha * d)
            self.alpha = alpha                       # starting step of the next line search
            self._iterate(x)
        else:
            _logCost(self.maxiter, Jup)              # J of the final model
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
        maxShrink : int, optional
            Divisions of the step by 10 when no trial lowers J \
            Default is 3
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
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, precond=None):
        super().__init__(func, "steepestDescent", jac, hess, hessp, bounds, constraints, tol, callback, options, precond)

    def _direction(self, gradJ, it):
        pg = self._precond(gradJ)
        return -pg / np.max(np.abs(pg))


class CGPROptimizer2(_HeuristicOptimizer):
    """
    Polak-Ribiere conjugate gradient with the heuristic line search (options of
    SteepestDescentOptimizer).
    """
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None, precond=None):
        super().__init__(func, "CGPR2", jac, hess, hessp, bounds, constraints, tol, callback, options, precond)
        self._gradPast = None
        self._pgPast = None
        self._dPast = None

    def _direction(self, gradJ, it):
        """PR+ on the gradients normalised by max|P g| (the line search works in that unit)."""
        pg = self._precond(gradJ)
        norm = np.max(np.abs(pg))
        g, pg = gradJ / norm, pg / norm
        if self._gradPast is None:
            d = -pg
        else:
            chi = max(0.0, np.dot(g - self._gradPast, pg) / np.dot(self._gradPast, self._pgPast))   # Polak-Ribiere+
            d = -pg + chi * self._dPast
            if np.dot(g, d) >= 0:                        # not a descent direction: restart
                d = -pg
        self._gradPast, self._pgPast, self._dPast = g, pg, d
        return d
