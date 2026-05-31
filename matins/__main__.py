"""Enable `python -m matins ...` as an alias for the `matins` console script.

Useful when running an in-tree checkout (e.g. a git worktree) whose code should
take precedence over a globally pip-installed copy.
"""
from .cli import main

raise SystemExit(main())
