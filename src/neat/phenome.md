# Genome → Phenome Transformation and Activation

## Overview

This module converts a **genome** (graph-like genetic encoding) into a **phenome** (an executable neural structure), then activates that phenome on input data.

The implementation supports both:

- **Feedforward topologies**
- **Recurrent topologies** (including cycles and self-loops)

It handles this by decomposing the network into strongly connected components (SCCs), ordering those components safely, and using bounded fixed-point iteration for recurrent parts.

---

## Core Types Involved

### From genome layer (`super::genome::types`)

- `Genome`: source representation containing nodes and connection genes.
- `NodeId`: stable node identifier used in genome storage.
- `NodeKind`: node role, including:
  - `Sensor` (input)
  - `Output`
  - hidden/internal kinds
- `GenomeError`: genome-level validation/construction errors.

### Phenome-specific runtime types

- `Edge { src, dst, weight }`
  - Compact runtime edge using indexed node positions (`usize`) instead of `NodeId`.
- `PlannedComponent`
  - One SCC execution unit:
    - `nodes`: node indices in this SCC
    - `external_edges`: incoming edges from other SCCs
    - `internal_edges`: edges inside this SCC
    - `recurrent`: true if SCC has a cycle (size > 1 or self-loop)
- `ActivationConfig`
  - `recurrent_iterations`: hard cap on recurrent solve steps
  - `recurrent_epsilon`: convergence threshold
- `Phenome`
  - Runtime structure with:
    - node metadata (`node_ids`, `node_kinds`)
    - input/output index lists
    - planned SCC execution order (`components`)
    - mutable activation buffers:
      - `state`: current node outputs (the live network values)
      - `scratch`: temporary per-step accumulator buffer (explained below)
- `PhenomeError`
  - `InputArityMismatch`
  - `InvalidGenome(GenomeError)` (defined for compatibility/future use)

---

## What `scratch` means (plain language)

`scratch` is just a reusable **work buffer** with one `f64` slot per node.

Think of each activation step for a component as two phases:

1. **Accumulate all incoming weighted signals** into `scratch[node]`.
2. **Apply activation function** (ReLU) and write the result into `state[node]`.

Why this exists:

- It prevents mixing “old” and “new” values during the same step.
- It avoids reallocating temporary vectors every call.
- It makes recurrent iteration stable and explicit.

So when the code says “zero scratch values for component nodes”, it means:
“clear the temporary accumulator slots for exactly those nodes before computing their next values.”

---

## Activation Semantics

Activation is performed by `Phenome::activate(inputs)`.

### Step A) Reset dynamic state

`reset_non_sensor_state()` zeros all non-sensor nodes each call.

This makes each activation call independent, while still allowing iterative solving inside recurrent SCCs.

### Step B) Set input values

`set_inputs(inputs)` writes sensor values and validates arity.

Mismatch returns:
`PhenomeError::InputArityMismatch { expected, actual }`.

### Step C) Execute components in topological order

For each `PlannedComponent`:

- If non-recurrent: single-pass update.
- If recurrent: bounded iterative update until convergence or iteration cap.

---

## Non-recurrent component update

`activate_acyclic_component` does:

1. Zero temporary accumulator (`scratch`) for component nodes.
2. Add external weighted contributions into `scratch`.
3. Add internal weighted contributions into `scratch`.
4. Commit `state[node] = relu(scratch[node])` for non-sensor nodes.

Contribution rule per edge:
`scratch[dst] += state[src] * weight`

Node update rule:
`state[node] = relu(scratch[node])`

---

## Recurrent component update (safe handling)

`activate_recurrent_component` repeats up to `recurrent_iterations`:

1. Zero component scratch.
2. Accumulate external + internal weighted contributions from current `state` into `scratch`.
3. For each non-sensor node, compute `next = relu(scratch[node])`.
4. Commit `next` to `state` and track max delta.
5. Stop early if `max_delta <= recurrent_epsilon`.

### Why this is safe

- **No infinite loops**: hard iteration cap guarantees termination.
- **Convergence-aware**: epsilon stop avoids unnecessary iterations.
- **Sensor protection**: sensor values are never overwritten by recurrent updates.
- **Scoped writes**: only nodes in current component are touched via component-local scratch reset and commit.

---

## Performance Characteristics

### Algorithmic complexity

- SCC decomposition: **O(V + E)**
- Condensation topo sort: **O(C + Ec)** (components and inter-component edges)
- Activation:
  - Acyclic component: linear in its edges/nodes.
  - Recurrent component: linear per iteration, bounded by `recurrent_iterations`.

Overall, planning is near-linear in graph size, and runtime is bounded and predictable.

### Data locality and allocation strategy

- Uses index-based vectors (`Vec`) for node state and kinds.
- `state` and `scratch` are preallocated once in `from_genome_with_config`.
- Per-activation work reuses these buffers (no full-buffer reallocations).
- Component-level scratch zeroing limits writes to active nodes.

### Deterministic execution

- SCC DAG topological order ensures stable component sequencing.
- Recurrence handling is explicit and bounded, not implicit recursion or unbounded propagation.

---

## Error Handling and Correctness Signals

- Input size is strictly checked before activation.
- Construction assumes genome node references are valid (guarded by `expect` invariants in index conversion).
- Unit tests cover:
  - feedforward ReLU behavior (activation after sum)
  - recurrent bounded execution
  - input arity mismatch
  - mutation pipeline compatibility

---

## Practical Summary

The implementation converts a genome into a compact runtime plan by:

1. Indexing nodes and enabled edges,
2. Factoring cycles into SCC components,
3. Ordering components topologically,
4. Executing acyclic parts once and recurrent parts with bounded fixed-point iteration.

This yields a phenome activation process that is:

- **cycle-safe**
- **deterministic**
- **bounded in runtime**
- **efficient in memory/layout**
- **compatible with evolving NEAT-style topologies**