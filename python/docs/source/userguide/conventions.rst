Conventions
###########

- **Units**: metres, seconds, m/s, kg/m\ :sup:`3`.
- **Coordinates**: :code:`[x, y, z]`, seismic convention: z = 0 at the surface (the free
  surface, top of the mesh), positive downwards. In 2-D, y is ignored (give 0). The
  FUnTiDES meshes have z upwards; the conversion is done by makutu.
- **Models and gradients**: flat arrays per element in seismic order (x fastest, then y,
  then z from the surface); :code:`reshape(ez, ex)` in 2-D, :code:`reshape(ez, ey, ex)` in
  3-D. FWI parametrisations: :code:`c`, :code:`1/c` or :code:`1/c2` (density 1 by default).
- **SEP files** (models): 2-D if :code:`n3 = 1`, axis 1 = x and axis 2 = depth; else
  x, y, depth.
- **SEG-Y files** (data): one file per shot, one trace per receiver, IEEE float; source
  x, y in :code:`sx`, :code:`sy` and depth in :code:`sdepth` (byte 49); receiver x, y in
  :code:`gx`, :code:`gy` and depth in :code:`gelev` (byte 41), with the scalars
  :code:`scalco` and :code:`scalel`.
- **Source**: Ricker wavelets (:code:`ricker0` to :code:`ricker4`, :code:`ricker2` being the
  second derivative of a Gaussian) of peak frequency f0, delayed by 1 / f0.
