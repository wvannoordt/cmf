# CMF To-Do List
================

This is a place where items for development can be listed. It feels great to check things off!

 - [x] Make to-do list
 - [x] Design the CMF database I/O model and implement the appropriate base classes
 - [x] Implement I/O for mesh arrays and block refinement
 - [x] Implement CmfMeshDataBuffer class to replace simple void pointer
 - [x] Determine if the Synchronize in the DataExchangePattern is strictly necessary (it is not)
 - [x] Clearer boundary in CartesianMeshArray between allocated and defined nodes
 - [ ] Design and implement the post-refinement callback sequence
 - [x] Implement single-file output for cartesian mesh array
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
 - [ ] Move general-purpose mesh array functions into base class
 - [ ] Write config system in Python
 - [ ] Modify fixed-size arrays to be always 3D with 2-D default values
 - [x] Vtk blocks in bespoke directories
 - [ ] Implement Error code handling for CreateDirectory, best bet is probably try-catch
 - [ ] Fix the issue with multiple neighbor relationship bifurcation