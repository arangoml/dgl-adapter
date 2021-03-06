## 2.1.0 (2022-06-29)

### New

* Adjust DGL to ArangoDB for increased accessibility to python-arango interface (#24) [Anthony Mahanna]

  * #23: initial commit

  * Update build.yml

  * Update README.md

  * remove: unecessary dict merge using Kwargs

  The DGL HeteroNodeDataView / HeteroEdgeDataView is able to track previously inserted features, so this is not needed

  * remove: setuptools_scm from setup.py

  deprecated in favor of pyproject.toml usage

  * Update README.md

  * remove: __validate_attributes()

  Uncessary noise as we already have proper docstring examples

  * pragma: no cover

  * fix: black

  * chg: #24 (after redefining scope)

  * new: CodeQL Action v2

  * drop python 3.6 support

  mirrors https://github.com/arangoml/networkx-adapter/pull/81/commits/5930c804b03a6f5322b930a498dd68d3c5a743c9

  * Update adapter.py

  * fix: increase coverage

  * prep for 2.1.0 release: documentation tweaks

  * bump: python-arango version

### Fix

* Readme typo. [aMahanna]

### Other

* Update README.md. [aMahanna]

* Changelog: release 2.0.1 (#22) [github-actions[bot]]

  !gitchangelog


## 2.0.1 (2022-05-31)

### Fix

* Can't convert DGL heterograph with edge attributes into ArangoDB (#21) [Anthony Mahanna]

  * fix: #20

  * new: test case for #20

  * update: notebook version

  * remove: duplicate docstring

  * cleanup: test_dgl_to_adb

  * fix: notebook typo

### Other

* Changelog: release 2.0.0 (#19) [github-actions[bot]]

  !gitchangelog


## 2.0.0 (2022-05-25)

### New

* Notebook prep for 2.0.0 release (#18) [Anthony Mahanna]

  * #16: initial commit

  * nbstripout

* Verbose Logging (#17) [Anthony Mahanna]

  * #13: initial commit

  * Update ArangoDB_DGL_Adapter.ipynb

  * Revert "Update ArangoDB_DGL_Adapter.ipynb"

  This reverts commit 24059fe2ab74d5d879c990b3b10e8d094bd04518.

  * fix: black

  * new: validate_attributes test case

  * Update README.md

  * fix: mypy

  * cleanup

  * fix: black

  * Update release.yml

  * cleanup

  * fix: set default password to empty string

  * set empty password

  * #14: initial commit

  * cleanup

  * cleanup release.yml

  * new: import shortcut

  * cleanup

  * fix: switch back to manual changelog merge

  * replace: set_verbose with set_logging

* Expose ArangoClient & StandardDatabase from adapter (#15) [Anthony Mahanna]

  * #13: initial commit

  * Update ArangoDB_DGL_Adapter.ipynb

  * Revert "Update ArangoDB_DGL_Adapter.ipynb"

  This reverts commit 24059fe2ab74d5d879c990b3b10e8d094bd04518.

  * fix: black

  * new: validate_attributes test case

  * Update README.md

  * fix: mypy

  * cleanup

  * fix: black

  * Update release.yml

  * cleanup

  * fix: set default password to empty string

  * set empty password

  * new: specify requests dep

* Docker-based testing (#12) [Anthony Mahanna]

  * initial commit

  * fix: typo

  * Update conftest.py

  * fix: isort

  * temp: disable mypy

  * attempt: mypy fix

  * remove: unused import

  * cleanup

  * bump python-arango and torch versions

  * lower python-arango version

  * lower torch version

  * Update README.md

  * fix: password typo

  * Update README.md

  * Update conftest.py

  * Update conftest.py

  * Update setup.py

### Fix

* Flake8. [aMahanna]

* Var name. [aMahanna]

* Edge_collection retrieval. [aMahanna]

### Other

* Revert "update: start enumerate() at 1" [aMahanna]

  This reverts commit 7422187993933ec60c718cf06dcdbd8fcec812db.

* Update: start enumerate() at 1. [aMahanna]

* Update: build & analyze triggers. [aMahanna]

* Pragma no cover. [aMahanna]

* (more) minor cleanup. [aMahanna]

* Minor cleanup. [aMahanna]


## 1.0.2 (2021-12-31)

### New

* Blog post preparation. [Anthony Mahanna]

### Fix

* Update package version to latest release (#7) [Anthony Mahanna]


## 1.0.1 (2021-12-30)

### Fix

* Replace auto merge with echo (#5) [Anthony Mahanna]

* README colab link. [aMahanna]


## 1.0.0 (2021-12-30)

### New

* Mirror networkx-adapter changes (#3) [Anthony Mahanna]

* Notebook. [aMahanna]

* Workflow_dispatch. [aMahanna]

* __insert_adb_docs. [aMahanna]

* Readme & logos. [aMahanna]

### Changes

* Actions. [aMahanna]

### Fix

* Arangorestore. [aMahanna]

* Typo. [aMahanna]

* NotImplementedError() [aMahanna]

* Typo. [aMahanna]

* Tests. [aMahanna]

### Other

* Minor repo restructure & documentation update. [aMahanna]

* Update README.md. [aMahanna]

* Cleanup. [aMahanna]

* Update ArangoDB_DGL_Adapter.ipynb. [aMahanna]

* Update extract_version.py. [aMahanna]

* Update README.md. [Anthony Mahanna]

* Update README.md. [Anthony Mahanna]

* Update test_adbdgl_adapter.py. [aMahanna]

* Delete main.py. [Anthony Mahanna]

* Cleanup. [aMahanna]

* Update adbdgl_adapter.py. [aMahanna]

* Cleanup & doc strings. [aMahanna]

* Update setup.py. [aMahanna]

* Cleanup. [aMahanna]

* :night_with_stars: [aMahanna]

* Update conftest.py. [aMahanna]

* Update conftest.py. [aMahanna]

* Update conftest.py. [aMahanna]

* Update conftest.py. [aMahanna]

* Update conftest.py. [aMahanna]

* Update conftest.py. [aMahanna]

* Update conftest.py. [aMahanna]

* Update conftest.py. [aMahanna]

* More cleanup & batch insertion. [aMahanna]

* Cleanup. [aMahanna]

* :rocket: [aMahanna]

* Update adbdgl_adapter.py. [aMahanna]

* Update adbdgl_adapter.py. [aMahanna]

* Update adbdgl_adapter.py. [aMahanna]

* Update adbdgl_controller.py. [aMahanna]

* Checkpoint. [aMahanna]

* Remove py 3.10. [aMahanna]

* Initial commit. [aMahanna]


