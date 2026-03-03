use super::genome::types::{Genome, GenomeError, NodeId, NodeKind};
use std::collections::{BTreeSet, HashMap, VecDeque};

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[derive(Debug, Copy, Clone)]
struct Edge {
    src: usize,
    dst: usize,
    weight: f64,
}

#[derive(Debug, Clone)]
struct PlannedComponent {
    nodes: Vec<usize>,
    external_edges: Vec<Edge>,
    internal_edges: Vec<Edge>,
    recurrent: bool,
}

#[derive(Debug, Copy, Clone)]
pub struct ActivationConfig {
    pub recurrent_iterations: usize,
    pub recurrent_epsilon: f64,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            recurrent_iterations: 12,
            recurrent_epsilon: 1e-9,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PhenomeError {
    InputArityMismatch { expected: usize, actual: usize },
    InvalidGenome(GenomeError),
}

fn build_node_index(genome: &Genome) -> (Vec<NodeId>, Vec<NodeKind>, HashMap<NodeId, usize>) {
    let node_ids = genome.nodes.keys().copied().collect::<Vec<_>>();
    let node_kinds = node_ids
        .iter()
        .map(|id| *genome.nodes.get(id).expect("node id must exist"))
        .collect::<Vec<_>>();
    let node_index = node_ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (*id, idx))
        .collect::<HashMap<_, _>>();
    (node_ids, node_kinds, node_index)
}

fn build_enabled_edges(genome: &Genome, node_index: &HashMap<NodeId, usize>) -> Vec<Edge> {
    genome
        .connections_by_innovation
        .values()
        .filter(|g| g.enabled)
        .map(|g| Edge {
            src: *node_index
                .get(&g.key.in_node)
                .expect("source node must exist"),
            dst: *node_index
                .get(&g.key.out_node)
                .expect("destination node must exist"),
            weight: g.weight,
        })
        .collect::<Vec<_>>()
}

fn tarjan_scc(node_count: usize, edges: &[Edge]) -> Vec<Vec<usize>> {
    fn strong_connect(
        v: usize,
        graph: &[Vec<usize>],
        index: &mut usize,
        indices: &mut [Option<usize>],
        lowlink: &mut [usize],
        on_stack: &mut [bool],
        stack: &mut Vec<usize>,
        sccs: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = Some(*index);
        lowlink[v] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        for &w in &graph[v] {
            if indices[w].is_none() {
                strong_connect(w, graph, index, indices, lowlink, on_stack, stack, sccs);
                lowlink[v] = lowlink[v].min(lowlink[w]);
            } else if on_stack[w] {
                lowlink[v] = lowlink[v].min(indices[w].expect("must exist"));
            }
        }

        if lowlink[v] == indices[v].expect("must exist") {
            let mut component = Vec::new();
            loop {
                let w = stack.pop().expect("stack underflow");
                on_stack[w] = false;
                component.push(w);
                if w == v {
                    break;
                }
            }
            component.sort_unstable();
            sccs.push(component);
        }
    }

    let mut graph = vec![Vec::<usize>::new(); node_count];
    for e in edges {
        graph[e.src].push(e.dst);
    }

    let mut index = 0usize;
    let mut indices = vec![None; node_count];
    let mut lowlink = vec![0usize; node_count];
    let mut on_stack = vec![false; node_count];
    let mut stack = Vec::<usize>::new();
    let mut sccs = Vec::<Vec<usize>>::new();

    for v in 0..node_count {
        if indices[v].is_none() {
            strong_connect(
                v,
                &graph,
                &mut index,
                &mut indices,
                &mut lowlink,
                &mut on_stack,
                &mut stack,
                &mut sccs,
            );
        }
    }

    sccs
}

fn build_component_index(sccs: &[Vec<usize>], node_count: usize) -> Vec<usize> {
    let mut component_of_node = vec![0usize; node_count];
    for (cid, comp) in sccs.iter().enumerate() {
        for &n in comp {
            component_of_node[n] = cid;
        }
    }
    component_of_node
}

fn build_condensation_topo(
    sccs: &[Vec<usize>],
    component_of_node: &[usize],
    edges: &[Edge],
) -> Vec<usize> {
    let c_count = sccs.len();
    let mut dag_out = vec![BTreeSet::<usize>::new(); c_count];
    let mut indegree = vec![0usize; c_count];

    for e in edges {
        let c_src = component_of_node[e.src];
        let c_dst = component_of_node[e.dst];
        if c_src != c_dst && dag_out[c_src].insert(c_dst) {
            indegree[c_dst] += 1;
        }
    }

    let mut q = VecDeque::new();
    for (cid, d) in indegree.iter().enumerate() {
        if *d == 0 {
            q.push_back(cid);
        }
    }

    let mut order = Vec::with_capacity(c_count);
    while let Some(cid) = q.pop_front() {
        order.push(cid);
        for &next in &dag_out[cid] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                q.push_back(next);
            }
        }
    }

    order
}

fn build_plan(node_count: usize, edges: &[Edge]) -> Vec<PlannedComponent> {
    let sccs = tarjan_scc(node_count, edges);
    let component_of_node = build_component_index(&sccs, node_count);
    let topo = build_condensation_topo(&sccs, &component_of_node, edges);

    topo.into_iter()
        .map(|cid| {
            let nodes = sccs[cid].clone();
            let in_component = nodes.iter().copied().collect::<BTreeSet<_>>();

            let external_edges = edges
                .iter()
                .copied()
                .filter(|e| in_component.contains(&e.dst) && !in_component.contains(&e.src))
                .collect::<Vec<_>>();

            let internal_edges = edges
                .iter()
                .copied()
                .filter(|e| in_component.contains(&e.src) && in_component.contains(&e.dst))
                .collect::<Vec<_>>();

            let has_self_loop = internal_edges.iter().any(|e| e.src == e.dst);
            let recurrent = nodes.len() > 1 || has_self_loop;

            PlannedComponent {
                nodes,
                external_edges,
                internal_edges,
                recurrent,
            }
        })
        .collect::<Vec<_>>()
}

#[derive(Debug, Clone)]
pub struct Phenome {
    node_ids: Vec<NodeId>,
    node_kinds: Vec<NodeKind>,
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
    components: Vec<PlannedComponent>,
    config: ActivationConfig,
    state: Vec<f64>,
    scratch: Vec<f64>,
}

impl Phenome {
    pub fn from_genome(genome: &Genome) -> Result<Self, PhenomeError> {
        Self::from_genome_with_config(genome, ActivationConfig::default())
    }

    pub fn from_genome_with_config(
        genome: &Genome,
        config: ActivationConfig,
    ) -> Result<Self, PhenomeError> {
        let (node_ids, node_kinds, node_index) = build_node_index(genome);
        let edges = build_enabled_edges(genome, &node_index);
        let components = build_plan(node_ids.len(), &edges);

        let input_indices = node_ids
            .iter()
            .enumerate()
            .filter_map(|(i, _)| (node_kinds[i] == NodeKind::Sensor).then_some(i))
            .collect::<Vec<_>>();

        let output_indices = node_ids
            .iter()
            .enumerate()
            .filter_map(|(i, _)| (node_kinds[i] == NodeKind::Output).then_some(i))
            .collect::<Vec<_>>();

        Ok(Self {
            node_ids,
            node_kinds,
            input_indices,
            output_indices,
            components,
            config,
            state: vec![0.0; genome.nodes.len()],
            scratch: vec![0.0; genome.nodes.len()],
        })
    }

    fn reset_non_sensor_state(&mut self) {
        for (i, kind) in self.node_kinds.iter().enumerate() {
            if *kind != NodeKind::Sensor {
                self.state[i] = 0.0;
            }
        }
    }

    fn set_inputs(&mut self, inputs: &[f64]) -> Result<(), PhenomeError> {
        if inputs.len() != self.input_indices.len() {
            return Err(PhenomeError::InputArityMismatch {
                expected: self.input_indices.len(),
                actual: inputs.len(),
            });
        }

        for (x, idx) in inputs.iter().zip(self.input_indices.iter().copied()) {
            self.state[idx] = *x;
        }

        Ok(())
    }

    fn zero_component_scratch(&mut self, comp: &PlannedComponent) {
        for &n in &comp.nodes {
            self.scratch[n] = 0.0;
        }
    }

    fn accumulate_external_weighted_sum(&mut self, comp: &PlannedComponent) {
        for e in &comp.external_edges {
            self.scratch[e.dst] += self.state[e.src] * e.weight;
        }
    }

    fn accumulate_internal_weighted_sum(&mut self, comp: &PlannedComponent) {
        for e in &comp.internal_edges {
            self.scratch[e.dst] += self.state[e.src] * e.weight;
        }
    }

    fn commit_component_values(&mut self, comp: &PlannedComponent) {
        for &n in &comp.nodes {
            if self.node_kinds[n] != NodeKind::Sensor {
                self.state[n] = relu(self.scratch[n]);
            }
        }
    }

    fn activate_acyclic_component(&mut self, comp: &PlannedComponent) {
        self.zero_component_scratch(comp);
        self.accumulate_external_weighted_sum(comp);
        self.accumulate_internal_weighted_sum(comp);
        self.commit_component_values(comp);
    }

    fn activate_recurrent_component(&mut self, comp: &PlannedComponent) {
        for _ in 0..self.config.recurrent_iterations {
            self.zero_component_scratch(comp);
            self.accumulate_external_weighted_sum(comp);
            self.accumulate_internal_weighted_sum(comp);

            let mut max_delta = 0.0f64;
            for &n in &comp.nodes {
                if self.node_kinds[n] == NodeKind::Sensor {
                    continue;
                }
                let next = relu(self.scratch[n]);
                let delta = (next - self.state[n]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                self.state[n] = next;
            }

            if max_delta <= self.config.recurrent_epsilon {
                break;
            }
        }
    }

    pub fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, PhenomeError> {
        self.reset_non_sensor_state();
        self.set_inputs(inputs)?;

        for i in 0..self.components.len() {
            let comp = self.components[i].clone();
            if comp.recurrent {
                self.activate_recurrent_component(&comp);
            } else {
                self.activate_acyclic_component(&comp);
            }
        }

        Ok(self
            .output_indices
            .iter()
            .map(|&idx| self.state[idx])
            .collect::<Vec<_>>())
    }

    pub fn node_count(&self) -> usize {
        self.node_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::genome::innovation::InnovationTracker;
    use crate::neat::genome::mutation::Mutation;
    use crate::neat::genome::types::{Genome, Innovation, NodeId};

    fn base_genome() -> (InnovationTracker, Genome) {
        let mut t = InnovationTracker::new(NodeId(2));
        let g = Genome::minimal_fully_connected(1, 1, &mut t, |_i, _o| 1.0);
        (t, g)
    }

    #[test]
    fn feedforward_activation_uses_relu_per_connection() {
        let mut t = InnovationTracker::new(NodeId(3));
        let g = Genome::minimal_fully_connected(2, 1, &mut t, |_i, _o| 1.0);

        let mut p = Phenome::from_genome(&g).unwrap();
        let out = p.activate(&[2.0, -3.0]).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 2.0).abs() < 1e-12); // relu(2*1) + relu(-3*1)
    }

    #[test]
    fn recurrent_activation_is_bounded_and_does_not_loop_forever() {
        let (mut t, g0) = base_genome();
        let split = g0.innovations().next().unwrap();

        let g1 = g0.with_added_node(&mut t, split).unwrap();
        let g2 = g1
            .with_added_connection(&mut t, NodeId(1), NodeId(2), 0.5)
            .unwrap();

        let mut p = Phenome::from_genome_with_config(
            &g2,
            ActivationConfig {
                recurrent_iterations: 4,
                recurrent_epsilon: 0.0,
            },
        )
        .unwrap();

        let out = p.activate(&[1.0]).unwrap();
        assert!((out[0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn activation_respects_input_arity() {
        let (_t, g) = base_genome();
        let mut p = Phenome::from_genome(&g).unwrap();
        let err = p.activate(&[1.0, 2.0]).err().unwrap();
        match err {
            PhenomeError::InputArityMismatch {
                expected: 1,
                actual: 2,
            } => {}
            _ => panic!("expected input arity mismatch"),
        }
    }

    #[test]
    fn apply_mutations_then_activate() {
        let mut t = InnovationTracker::new(NodeId(3));
        let g0 = Genome::minimal_fully_connected(2, 1, &mut t, |_i, _o| 1.0);
        let first = g0.innovations().next().unwrap();

        let g1 = g0
            .apply_mutations(
                &mut t,
                &[
                    Mutation::PerturbWeight {
                        innovation: first,
                        delta: 2.0,
                    },
                    Mutation::AddNode {
                        split_innovation: Innovation(1),
                    },
                ],
            )
            .unwrap();

        let mut p = Phenome::from_genome(&g1).unwrap();
        let out = p.activate(&[1.0, 3.0]).unwrap();
        assert!((out[0] - 6.0).abs() < 1e-12); // relu(1*3) + relu(3*1)
    }

    #[test]
    fn feedforward_activation_applies_relu_after_accumulation() {
        let mut t = InnovationTracker::new(NodeId(3));
        let g = Genome::minimal_fully_connected(2, 1, &mut t, |_i, _o| 1.0);

        let mut p = Phenome::from_genome(&g).unwrap();
        let out = p.activate(&[2.0, -3.0]).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.0).abs() < 1e-12); // relu((2*1) + (-3*1))
    }
}
