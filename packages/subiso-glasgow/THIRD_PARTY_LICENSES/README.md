This package links against the Glasgow Subgraph Solver, pinned as the
`vendor/glasgow` git submodule of the subiso repository and built into the
wheel.  Glasgow is distributed under the MIT licence; a copy is in this
directory (Glasgow-LICENCE).

GMP is linked dynamically and is not bundled; it is dual-licensed
LGPL-3.0-or-later / GPL-2.0-or-later.  When redistributing wheels, keep the
link dynamic (as this package does by linking the system library) or provide a
replacement mechanism, and state the GMP licence to users.
