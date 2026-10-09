# GRAM changelog

No GRAM source patch is included. The controlled workflow invokes the official
checkout and records its exact commit. If seed propagation proves incomplete,
add only a minimal seed patch under `gram_patches/` and document the file,
original behavior, modified behavior, reason, and whether algorithm semantics
changed before running controlled jobs.

At audited commit `7ac4d9272a57beed9df35c27ea34221f6e4a8fb1`, the official entrypoint
calls `set_seed(args.seed)` in both single and distributed paths; that helper
seeds Python, NumPy, PyTorch, and all CUDA devices. The data samplers derive
their deterministic shuffles from the same seed. Therefore no seed patch is
needed and algorithm semantics remain unchanged.
