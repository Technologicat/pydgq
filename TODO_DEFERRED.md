# Deferred TODOs

Items noticed during modernization that are out of scope for the current task.

<!-- New items go below this line. -->

## Add convergence tolerance setting (GitHub #4)

*Cluster: solver-features · Cost: S · Gate: none · Filed: 2026-04-17 · See also: GitHub #4*

Enhancement. Needs changes in `implicit.pyx` and `galerkin.pyx` wherever `maxit` is used.

## Add Newton-Raphson iteration (GitHub #5)

*Cluster: solver-features · Cost: ? · Gate: none · Filed: 2026-04-17 · See also: GitHub #5*

Enhancement. Would allow faster convergence for stiff problems compared to Banach/Picard.
