# Shortest delivery route

An entry for make.fun's `shortest-route-01`: a Euclidean TSP solver in plain Node 22 (standard library only).

`main.ts` reads every `/input/<id>.txt` and writes `/output/<id>.tour`:

1. K-nearest candidate lists from a uniform grid;
2. a greedy-edge start tour;
3. 2-opt + Or-opt local search with neighbour lists and a don't-look queue;
4. an iterated local search with local double-bridge kicks and exact undo.

The search is bounded by iteration counts and seeded by constants, so runs are deterministic.

Local run: `MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=<dir> node main.ts`.
