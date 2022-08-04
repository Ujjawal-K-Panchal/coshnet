The currrent implementation of **mp** contains the following tools for multiprocessing:


1. [mppool](./mppool.py): Having asynchronous processes with load balancing using a processor pool.

2. [mpxform](./mpxform.py): To transform and precondition samples of a given dataset using any transform *(Complex Shearlet Transform in our case)*. Usually the end product is
an implementation.
compression.