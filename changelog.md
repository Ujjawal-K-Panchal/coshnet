# Changes:

## Releases:
`v1.0.0`:
-> first stable version.
-> code/ has runnable experiments code.
-> libs/ has libraries: pyutils, shnetutil, tt-pytorch.

## Branch PR changes:

## Important Abbreviations:
	- `TODO`: Each `TODO` denotes something that is incomplete or in-progress. It will be improved in future when bandwidth and time permits.

	- `NOTE`: Each `NOTE` denotes an important note, either to the user who is trying to use the code, or the maintainer who is trying to find the relevance or importance of a particular suite with respect to the repo architecture. 


## TODO:
- [x] To make docker run.
- [x] To make both the sequential runs into a shell file and put as final run command.

## Note to devs:
1. Leaving concept of `colorspace in shnetutil`. It will be easy if we want to iteratively add stuff to this, but if we donot, having it in, does not hurt and saves us trouble to refactor.
2. `shnetutil.pipeline.logutils` and `shnetutil.pipeline.trainutils` misplaced? Because utilities architecturally should be in `shnetutil.utils`.
