import datetime

import kirby

author = "Neuro-foundations team"
project = "torch_brain"
# version = kirby.__version__
copyright = f"{datetime.datetime.now().year}, {author}"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_inline_tabs",
    "sphinx.ext.mathjax",
]

html_theme = "sphinx_rtd_theme"
# html_logo = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
#              'master/pyg_sphinx_theme/static/img/pyg_logo.png')
# html_favicon = ('https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/'
#                 'master/pyg_sphinx_theme/static/img/favicon.png')
html_static_path = ["_static"]
templates_path = ["_templates"]

add_module_names = False
autodoc_member_order = "bysource"

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
}

myst_enable_extensions = [
    "html_admonition",
    "html_image",
    # add other MyST extensions you want to use
]
