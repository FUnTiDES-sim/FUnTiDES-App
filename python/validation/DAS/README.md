# DAS validation: analytical solution

`analytical/DAS_verification.py` computes the analytical response of a homogeneous
isotropic elastic medium (P velocity 4000 m/s, S velocity 2000 m/s, density
2000 kg/m^3) to an isotropic moment-tensor point source at the origin (Ricker
wavelet, 10 Hz, dt = 1 ms, 1000 samples), at the point (100, 100, 100) m:

- displacement: `ux.txt`, `uy.txt`, `uz.txt`;
- displacement differences between the ends of the DAS gauge: `uxdiff.txt`,
  `uydiff.txt`, `uzdiff.txt`;
- DAS signal (strain along the fiber averaged over the gauge length of 8 m, 7
  quadrature points; dip 30°, azimuth 45°): `dassyn`;
- strain tensor at the last quadrature point: `strain_ij`.

Run it from `analytical/` (numpy, about 5 s) to regenerate these files:

    cd analytical && python DAS_verification.py

The files are the reference for a DAS simulation with the same parameters.
