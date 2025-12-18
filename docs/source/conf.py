# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'python-gazetteer'
copyright = '2025, Sooraj Sivadasan'
author = 'Sooraj Sivadasan'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = ["_build"]
pygments_style = "one-dark"
pygments_dark_style = "one-dark"



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ['_static']
html_theme_options = {
    "show_prev_next": True,
    "awesome_external_links": True,
}
html_css_files = [
    "custom.css",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
html_sidebars = {
  "**": ["sidebar_main_nav_links.html", "sidebar_toc.html"]
}
html_title = "Python Gazetteer"
