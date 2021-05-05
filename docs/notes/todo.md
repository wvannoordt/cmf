# CMF To-Do List
================

This is a place where items for development can be listed. It feels great to check things off!

 - [x] Make to-do list
 - [x] Design the CMF database I/O model and implement the appropriate base classes
 - [x] Implement I/O for mesh arrays and block refinement
 - [x] Implement CmfMeshDataBuffer class to replace simple void pointer
 - [ ] Design and implement the post-refinement callback sequence
 - [x] Complete implementation of CartesianMeshBuffer
 - [x] Ensure that exchange cells are properly output for cartesian mesh
 - [x] Implement the SeekString() in the parallel file class
 - [ ] Implement allocation, partitioning, and management of GPU arrays
 - [ ] Inter-level operators for block-data exchanges
 - [ ] Restriction and prolongation operators: implement as event subscriptions
 - [x] Configuration file system: Produce a header file included by `cmf.h` so that the end user doesn't have to compile with the same compiler flags.
 - [ ] Switch build system to Python?
 - [ ] Change surface representation to accommodate 2-D surfaces
 - [ ] Change surface representation to store nodes and topology (instead of stl-ish format)
 - [ ] Move mesh array define functions to base class
 - [ ] Write config system in Python