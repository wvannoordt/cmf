# CMF Standard

This document lays out the expected standard for CMF. This standard is provided in the hope that it makes the lives of users and the development team as easy as possible.

## General Guidelines

The general philosophy of this project is straightforward: favor ease-of-use when possible, and optimize where it matters, and allow users a great deal of control.

## Documentation

Documentation for CMF is done using Doxygen. As such, it is important that functions/variables/classes etc. are documented accordingly. Descriptions should be as comprehensive as possible
and should be able to be understood when in the context of whatever class/file they appear in. Everything should be documented.

## Naming Conventions and Namespaces

Since this is a large framework with a lot of moving parts, the names of variables and functions are very important, especially public-facing ones. The aim of the naming convention
in CMF, along with the documentation strategy, is to prevent developers from being required to go to the definition of a function or variable to understand its purpose.

Class names and function names should be written using `PascalCase`, while variables should be written using `camelCase`. `snake_case` and names with underscores should be avoided
with the exception of special functions such as `Cmf_Alloc`, which is (necessarily) a macro invoking an internal function call. Full words and complete variable names should be preferred,
as an example consider the function `UpdateNeighborsOfNeighborsToChildNodes(...)`. The function name may look long and cumbersome, but it is clear what it does within the context of its
class.

## Comments

Comments are great things! Generally, there isn't much to say about comments as the naming conventions and documentation standard should result in reasonably readable code.
If a function necessarily does a lot of individual things, then comments should accompany it. There should be no lines of code that have been commented out. If a function
is an implementation of something coming from a research paper or an online source, there should be a citation, DOI, or link to the corresponding source.

## Macros

Macros should generally be avoided for a number of well-known reasons, although macros certainly have their place within CMF. An example is the `CmfError` function,
which is a macro for `CmfError_M` including the file name and line number arguments. In short, if it can be done without a macro, it should be done that way, but if
the implementation justifies it, a macro is fine. Macro expansions should use containing parentheses generously.

## Output

CMF is a library, meaning it ultimately accompanies an application that is being developed externally, which might require a specific way of managing output. Therefore,
any output statements in CMF should use the built-in `WriteLine` functions, and the debug level parameter should be strictly greater than 0. This means that by default,
there is absolutely no output from CMF, and this is the desired behavior. In the case that CMF throws an error and `WriteLine` is not enough for the required output,
CMF provides a stream output class with a global instance `cmfout`, used the same as `std::cout`, that can be used for output. The reason to use this is because
`cmfout` can be configured to output to log files as well as the terminal, and is what `WriteLine` uses internally. There should be no use of `std::cout` at all, and
the usage of `cmfout` should be restricted to instances where errors are thrown.

Excessive output should be avoided, as it does not provide useful information to the user. Any information-rich output should be logged as a file that can be
processed separately.

## Input Options

CMF makes use of `PTL`, which is a script-like, JSON-like input file reader specifically designed for use in scientific computing. Classes such as `CartesianMesh`
are constructed using quite a lot of information, so constructor structs are provided to build these objects (e.g. `CartesianMeshInputInfo`). These can be constructed directly
in an external application, but should also inherit the `ICmfInputObject` interface. `PTL` is designed to provide a simple and easy-to-use input file interface, and
so should be used for user-provided settings when possible and appropriate.

## Error Handling

All errors should be handled through the `CmfError` function. This is because the error is handled as an exception internally, allowing external applications to handle errors
elegantly if required. `CmfError` also provides the file and line (and, if supported, the function) that the error is called from, allowing for quicker debugging of issues.
There should be absolutely no use of `abort()` as this will automatically crash any external application that uses CMF. In general, errors should not be thrown from deeply-nested
loops.

## Memory Allocation

CMF provides a garbage-collector-like class `CmfGC` and a globally-available instance `cmfGC`. This is done to keep a close eye on memory allocation for performance and to see where
resources are being allocated to. Other advantages are to prevent memory leaks by keeping track of allocated pointers, to track where large allocations are coming from, and to allow
for dynamic stack (instead of heap) allocations. In short, memory allocations for primitive arrays (for both CPU and GPU) should be done using the `Cmf_Alloc` function, and frees
should be made using the corresponding `Cmf_Free` function. There is currently no internal handling of `new` (and `new []`) allocations, so those should be handled with the corresponding
`delete` (and `delete []`) statements.

## External Dependencies

CMF has only one core dependency: `PTL`. Any other external dependency should be included in a way that allows users to compile CMF with or without that dependency. The current
approach is to use wrapper classes for each dependency that is included, for example the `ParallelGroup` class, which is like a wrapper for all `MPI` operations. The makefile should be
modified to include a preprocessor flag to enable each dependency, e.g. compiling with `-DPARRALEL=1` for MPI. Dependencies will not be packaged with CMF, and external dependencies
shuld generally be minimized.

## Practices to Avoid

- There should be no `using namespace <namespace>` directives.
- Macros should be avoided as much as possible, although there are times and places for them.
- There should be no use of `std::cout`, `std::cin`, or `abort()`.
- There should absolutely not be any pause statements of any kind.
- Any comments in the code should not contain any executable code, unless provided as a usage example.
- There should be no dupluicate functions, including variations ending in `_new`, `_v2`, `_old`, etc. separate functions for
  2D and 3D cases should be avoided as much as possible.
- The selection of methods should not be assigned to an integer. If a differentiating class is not appropriate, then any
  method selection should be implemented using an enumeration with a corresponding string function.
- There should be no files output unless explicitly enabled or as a consequence of a crash. No files should be output to the current directory explicitly.
- Avoid deallocating memory only to immediately re-allocate it, unless it is part of an isolated resizing or repartitioning operation.

## Header Files
