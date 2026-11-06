def backward(solver, residual, model="c"):
    """
    Adjoint run of the current shot and its gradient (adjoint-state method).

    Call after ``forward(solver, shot, comm, snapshotStep)`` with a snapshot step:
    the residual (observed - simulated) is injected, reversed in time, at the
    receivers, and the gradient of J = dt / 2 * sum(residual^2) is returned
    (``solver.computeGradient``). The receivers of the solver are replaced by the
    adjoint sources: read the seismograms before.

    Parameters
    ----------
        solver : AcousticSolver
            Solver of the shot
        residual : array, shape (n_samples, n_receivers)
            observed - simulated, sampled every ``solver.dtSeismoRecorded``
        model : str
            Parametrisation of the gradient: "c", "1/c" or "1/c2"

    Returns
    -------
        np.ndarray : gradient per element, flat, seismic order
    """
    return solver.computeGradient(residual, model)
