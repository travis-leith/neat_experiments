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

fn compute_can_reach_output(
    node_count: usize,
    output_indices: &[usize],
    edges: &[Edge],
) -> Vec<bool> {
    let mut reverse = vec![Vec::<usize>::new(); node_count];
    for e in edges {
        reverse[e.dst].push(e.src);
    }

    let mut can_reach = vec![false; node_count];
    let mut queue = VecDeque::new();

    for &out in output_indices {
        if !can_reach[out] {
            can_reach[out] = true;
            queue.push_back(out);
        }
    }

    while let Some(dst) = queue.pop_front() {
        for &src in &reverse[dst] {
            if !can_reach[src] {
                can_reach[src] = true;
                queue.push_back(src);
            }
        }
    }

    can_reach
}

fn build_active_component_indices(
    components: &[PlannedComponent],
    can_reach_output: &[bool],
) -> Vec<usize> {
    components
        .iter()
        .enumerate()
        .filter_map(|(cid, comp)| {
            comp.nodes
                .iter()
                .any(|&n| can_reach_output[n])
                .then_some(cid)
        })
        .collect::<Vec<_>>()
}

fn build_active_non_sensor_nodes(
    components: &[PlannedComponent],
    active_component_indices: &[usize],
    node_kinds: &[NodeKind],
) -> Vec<usize> {
    let mut seen = vec![false; node_kinds.len()];
    let mut active = Vec::new();

    for &cid in active_component_indices {
        for &n in &components[cid].nodes {
            if node_kinds[n] != NodeKind::Sensor && !seen[n] {
                seen[n] = true;
                active.push(n);
            }
        }
    }

    active
}

#[derive(Debug, Clone)]
pub struct Phenome {
    node_ids: Vec<NodeId>,
    node_kinds: Vec<NodeKind>,
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
    components: Vec<PlannedComponent>,
    active_component_indices: Vec<usize>,
    active_non_sensor_nodes: Vec<usize>,
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

        let can_reach_output = compute_can_reach_output(node_ids.len(), &output_indices, &edges);
        let active_component_indices =
            build_active_component_indices(&components, &can_reach_output);
        let active_non_sensor_nodes =
            build_active_non_sensor_nodes(&components, &active_component_indices, &node_kinds);

        Ok(Self {
            node_ids,
            node_kinds,
            input_indices,
            output_indices,
            components,
            active_component_indices,
            active_non_sensor_nodes,
            config,
            state: vec![0.0; genome.nodes.len()],
            scratch: vec![0.0; genome.nodes.len()],
        })
    }

    fn reset_non_sensor_state(&mut self) {
        for &i in &self.active_non_sensor_nodes {
            self.state[i] = 0.0;
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

        for i in 0..self.active_component_indices.len() {
            let cid = self.active_component_indices[i];
            let comp = self.components[cid].clone();

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
    use crate::neat::genome::types::{ConnectionKey, Genome, Innovation, NodeId};

    fn base_genome() -> (InnovationTracker, Genome) {
        let mut t = InnovationTracker::new(NodeId(2));
        let g = Genome::minimal_fully_connected(1, 1, &mut t, |_i, _o| 1.0);
        (t, g)
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

    // fn genome_sample_feed_forward_1() -> Genome {
    //     Genome::create(
    //         vec![
    //             Gene::create(0, 4, -0.1, true),
    //             Gene::create(4, 3, 0.6, true),
    //             Gene::create(1, 5, -0.8, true),
    //             Gene::create(5, 3, -0.9, true),
    //             Gene::create(0, 5, 0.6, true),
    //             Gene::create(5, 2, 0.4, true),
    //         ],
    //         2,
    //         2,
    //     )
    // }

    fn genome_sample_feed_forward_1() -> Genome {
        let mut t = InnovationTracker::new(NodeId(0));
        let mut g = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.);

        fn get_innovation(g: &Genome, in_node: NodeId, out_node: NodeId) -> Innovation {
            let key = ConnectionKey { in_node, out_node };
            g.connection_to_innovation.get(&key).copied().unwrap()
        }

        let innovation_0_3 = get_innovation(&g, NodeId(0), NodeId(3));
        let innovation_1_3 = get_innovation(&g, NodeId(1), NodeId(3));

        let new_node_mutations = vec![
            Mutation::AddNode {
                split_innovation: innovation_0_3,
            },
            Mutation::AddNode {
                split_innovation: innovation_1_3,
            },
        ];

        g = g.apply_mutations(&mut t, &new_node_mutations).unwrap();

        let innovation_0_2 = get_innovation(&g, NodeId(0), NodeId(2));

        g = g
            .apply_mutation(
                &mut t,
                &Mutation::DisableConnection {
                    innovation: innovation_0_2,
                },
            )
            .unwrap();

        let add_connection_mutations = vec![
            Mutation::AddConnection {
                in_node: NodeId(0),
                out_node: NodeId(5),
                weight: 0.6,
            },
            Mutation::AddConnection {
                in_node: NodeId(5),
                out_node: NodeId(2),
                weight: 0.4,
            },
        ];

        g = g
            .apply_mutations(&mut t, &add_connection_mutations)
            .unwrap();

        fn get_weight_adjust_mutation(
            g: &Genome,
            in_node: NodeId,
            out_node: NodeId,
            delta: f64,
        ) -> Mutation {
            let innovation = get_innovation(g, in_node, out_node);
            Mutation::PerturbWeight { innovation, delta }
        }

        let weight_adjust_mutations = vec![
            get_weight_adjust_mutation(&g, NodeId(0), NodeId(4), -0.1),
            get_weight_adjust_mutation(&g, NodeId(1), NodeId(5), -0.8),
            get_weight_adjust_mutation(&g, NodeId(4), NodeId(3), 0.6),
            get_weight_adjust_mutation(&g, NodeId(5), NodeId(3), -0.9),
        ];

        g.apply_mutations(&mut t, &weight_adjust_mutations).unwrap()
    }

    #[test]
    fn feed_forward() {
        let genome = genome_sample_feed_forward_1();
        let mut p = Phenome::from_genome(&genome).unwrap();
        let output = p.activate(&vec![0.5, -0.2]).unwrap();
        println!("{:?}", output);
        assert!((output[0] - 0.184).abs() < 1e-12);
        assert!((output[1] - 0.0).abs() < 1e-12);
    }
}
