Display
#######

Scripts of :file:`apps/display`. They print their options with :code:`--help`; the
figures open in a window, or are saved as PNG files with :code:`--png` (no display
needed, e.g. on a cluster).

- :file:`showSeismicTraces.py`: SEG-Y gathers written by makutu, or SEP files (axis 1 =
  time); time down, symmetric grey scale.

  .. code-block:: bash

      python3 apps/display/showSeismicTraces.py --dir obs seismoTrace_shot00001.sgy --png

- :file:`showVelocityModel.py`: models of the FWI (:code:`velocityModel<n>.hdf5`,
  :code:`finalModel.hdf5`), displayed as velocity whatever the parametrisation
  (:code:`--vmodel c`, :code:`1/c` or :code:`1/c2`); :code:`--diff` adds the second model
  minus the first.

  .. code-block:: bash

      python3 apps/display/showVelocityModel.py velocityModel0.hdf5 finalModel.hdf5 --diff

- :file:`showFullGradient.py`: gradients saved in debug mode
  (:code:`partialGradient/fullGradient<n>.hdf5`, symmetric colour scale), or the
  illumination with :code:`--fname illumination`.

The HDF5 files written by makutu carry the shape of the grid and the domain lengths as
attributes of the datasets (:func:`makutu.forwardBackward.modelio.gridAttributes`): the
2-D fields are displayed whole, in metres, z down. 3-D fields are displayed as a slice:
:code:`--axis y` (default) is the vertical x-z section at constant y, :code:`--axis x` the
y-z section, :code:`--axis z` the horizontal x-y slice; :code:`--index` is the element
index of the slice (default the middle). Files without attributes need the numbers of
elements, :code:`--n nx nz` or :code:`--n nx ny nz`, and optionally :code:`--lengths`.

The classes behind them are in :mod:`makutu.display`. The FWI examples also have their own
:file:`check_result.py` (models, profiles, cost per stage).
