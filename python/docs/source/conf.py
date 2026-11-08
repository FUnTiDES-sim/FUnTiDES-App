# Sphinx configuration of the makutu documentation (build: `make docs` from FUnTiDES-App/python).
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # makutu

project = 'makutu'
copyright = '2022-2026, Makutu team, TotalEnergies - Inria'
author = 'Makutu team'

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon',
              'sphinx.ext.mathjax', 'sphinx-prompt']

# FUnTiDES (pyfuntides, kokkos) is not needed to build the documentation
autodoc_mock_imports = ["kokkos", "pyfuntides"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
napoleon_numpy_docstring = True
suppress_warnings = ["ref.python"]     # 'list' in the types matches several attributes

templates_path = ['_templates']
exclude_patterns = []
toc_object_entries = False
add_module_names = False

html_theme = "sphinx_rtd_theme"
html_static_path = []
