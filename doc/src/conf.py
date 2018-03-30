project = 'Visual SLAM'
html_short_title = 'Workspace'
copyright = '2018, Dexter Watkins'

import os,inspect,sys
sys.path.insert(0, os.path.abspath('.'))

master_doc = 'index'
source_suffix = '.rst'
exclude_patterns = ['**/.#*']
extensions = ['numfig','sphinx.ext.pngmath', 'sphinx.ext.autodoc', 'sphinxcontrib.spelling', 'sphinx.ext.todo']
templates_path = ['_templates']
autoclass_content = "both"
autodoc_member_order = "bysource"

pygments_style = 'sphinx'
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = 'Visual SLAM'
html_static_path = ['_static']
html_context = { 'css_files': ['./_static/custom.css'] }

todo_include_todos = True

add_module_names = False
show_authors = True

spelling_word_list_filename = 'dictionary.txt'

