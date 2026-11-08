Display
#######

Scripts of :file:`apps/display` (they print their options with :code:`--help`; the figure
opens in a window):

- :file:`showSeismicTraces.py`: SEG-Y (or SEP) gathers,

  .. code-block:: bash

      python3 apps/display/showSeismicTraces.py --dir obs seismoTrace_shot00001.sgy

- :file:`showVelocityModel.py`: models of the FWI (:code:`velocityModel<n>.hdf5`,
  :code:`finalModel.hdf5`); :code:`--n nx ny nz` is the number of elements (ny = 1 in
  2-D), :code:`--vmodel` the parametrisation;

  .. code-block:: bash

      python3 apps/display/showVelocityModel.py finalModel.hdf5 --n 200 1 70 --axis y   # 2-D: (x, z) section

- :file:`showFullGradient.py`: gradients saved in debug mode
  (:code:`partialGradient/fullGradient<n>.hdf5`), same options.

The classes behind them are in :mod:`makutu.display`. The FWI examples also have their own
:file:`check_result.py` (models, profiles, cost per stage).
