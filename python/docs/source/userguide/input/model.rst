Model
########
Information about the mesh can be found in the official GEOS `documentation <https://geosx-geosx.readthedocs-hosted.com/en/latest/coreComponents/mesh/docs/Mesh.html#>`_.

The initial model is given to GEOS thanks to the **Mesh** field in the XML input file. It is a mandatory field, but note that it is possible to update the mesh from the python side using a function such as :code:`AcousticSolver.updateVelocityModel( newModel )` for example for the acoustic case.


.. rubric:: An example for an acoustic VTK mesh

.. code-block:: xml

    <Mesh>
        <VTKMesh
            name="mesh"
            fieldsToImport="{ Vp }"
            fieldNamesInGEOSX="{ acousticVelocityVp }"
            file="bicoucheAcoustic.vtu"/>
    </Mesh>


.. rubric:: An example for an elastic VTK mesh

.. code-block:: xml

    <Mesh>
        <VTKMesh
            name="mesh"
            fieldsToImport="{Vs, Vp, density }"
            fieldNamesInGEOSX="{ elasticVelocityVs, elasticVelocityVp, elasticDensity }"
            file="bicoucheElastic.vtu"/>
    </Mesh>


.. Important:: GEOS only admit :code:`CellData` type for the VTK fields to import.