use super::genome::types::{Genome, GenomeError, NodeId, NodeKind};
use serde::{Deserialize, Serialize};
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
    enabled: bool,
}

#[derive(Debug, Clone)]
struct PlannedComponent {
    nodes: Vec<usize>,
    external_edges: Vec<Edge>,
    internal_edges: Vec<Edge>,
    recurrent: bool,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct ActivationConfig {
    pub recurrent_iterations: usize,
    pub recurrent_epsilon: f64,
    pub logging_enabled: bool,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            recurrent_iterations: 12,
            recurrent_epsilon: 1e-9,
            logging_enabled: false,
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

fn build_all_edges(genome: &Genome, node_index: &HashMap<NodeId, usize>) -> Vec<Edge> {
    genome
        .connections_by_innovation
        .values()
        .map(|g| Edge {
            src: *node_index
                .get(&g.key.in_node)
                .expect("source node must exist"),
            dst: *node_index
                .get(&g.key.out_node)
                .expect("destination node must exist"),
            weight: g.weight,
            enabled: g.enabled,
        })
        .collect::<Vec<_>>()
}

fn build_enabled_edges(all_edges: &[Edge]) -> Vec<Edge> {
    all_edges
        .iter()
        .copied()
        .filter(|e| e.enabled)
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

fn compute_reachable_from_inputs(
    node_count: usize,
    input_indices: &[usize],
    edges: &[Edge],
) -> Vec<bool> {
    let mut forward = vec![Vec::<usize>::new(); node_count];
    for e in edges {
        forward[e.src].push(e.dst);
    }

    let mut reachable = vec![false; node_count];
    let mut queue = VecDeque::new();

    for &input in input_indices {
        if !reachable[input] {
            reachable[input] = true;
            queue.push_back(input);
        }
    }

    while let Some(src) = queue.pop_front() {
        for &dst in &forward[src] {
            if !reachable[dst] {
                reachable[dst] = true;
                queue.push_back(dst);
            }
        }
    }

    reachable
}

fn intersect_masks(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| *x && *y)
        .collect::<Vec<_>>()
}

fn build_active_component_indices(
    components: &[PlannedComponent],
    active_nodes: &[bool],
) -> Vec<usize> {
    components
        .iter()
        .enumerate()
        .filter_map(|(cid, comp)| comp.nodes.iter().any(|&n| active_nodes[n]).then_some(cid))
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
    all_edges: Vec<Edge>,
    config: ActivationConfig,
    state: Vec<f64>,
    scratch: Vec<f64>,
    audit_log: Vec<String>,
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
        let all_edges = build_all_edges(genome, &node_index);
        let edges = build_enabled_edges(&all_edges);
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
        let reachable_from_inputs =
            compute_reachable_from_inputs(node_ids.len(), &input_indices, &edges);
        let active_nodes = intersect_masks(&can_reach_output, &reachable_from_inputs);

        let active_component_indices = build_active_component_indices(&components, &active_nodes);
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
            all_edges,
            config,
            state: vec![0.0; genome.nodes.len()],
            scratch: vec![0.0; genome.nodes.len()],
            audit_log: Vec::new(),
        })
    }

    fn node_label(&self, idx: usize) -> String {
        format!("{:?}", self.node_ids[idx])
    }

    fn log_line(&mut self, line: String) {
        if self.config.logging_enabled {
            self.audit_log.push(line);
        }
    }

    pub fn set_logging_enabled(&mut self, enabled: bool) {
        self.config.logging_enabled = enabled;
        if !enabled {
            self.audit_log.clear();
        }
    }

    pub fn audit_log(&self) -> &[String] {
        &self.audit_log
    }

    pub fn take_audit_log(&mut self) -> Vec<String> {
        std::mem::take(&mut self.audit_log)
    }

    fn node_kind_class(kind: NodeKind) -> &'static str {
        match kind {
            NodeKind::Sensor => "sensor",
            NodeKind::Output => "output",
            NodeKind::Hidden => "internal",
        }
    }

    pub fn to_mermaid(&self, include_disabled: bool) -> String {
        let mut lines = vec![
            "graph TD".to_string(),
            "classDef sensor fill:#d7f0ff,stroke:#1e88e5,stroke-width:1px;".to_string(),
            "classDef output fill:#e8f5e9,stroke:#43a047,stroke-width:1px;".to_string(),
            "classDef internal fill:#fff8e1,stroke:#f9a825,stroke-width:1px;".to_string(),
        ];

        for (cid, comp) in self.components.iter().enumerate() {
            let component_kind = if comp.recurrent {
                "recurrent"
            } else {
                "acyclic"
            };
            lines.push(format!(
                "subgraph C{}[\"Component {} ({})\"]",
                cid, cid, component_kind
            ));
            for &n in &comp.nodes {
                lines.push(format!(
                    "  N{}[\"{} / {:?}\"]",
                    n,
                    self.node_label(n),
                    self.node_kinds[n]
                ));
            }
            lines.push("end".to_string());
        }

        for e in &self.all_edges {
            if !include_disabled && !e.enabled {
                continue;
            }
            let w = format!("{:.6}", e.weight);
            let link = if e.enabled {
                format!("-->|{}|", w)
            } else {
                format!("-.->")
            };
            lines.push(format!("N{} {} N{}", e.src, link, e.dst));
        }

        for (i, kind) in self.node_kinds.iter().copied().enumerate() {
            lines.push(format!("class N{} {};", i, Self::node_kind_class(kind)));
        }

        lines.join("\n")
    }

    fn reset_non_sensor_state(&mut self) {
        for &i in &self.active_non_sensor_nodes {
            self.state[i] = 0.0;
        }
        self.log_line("reset_non_sensor_state".to_string());
    }

    fn zero_component_scratch_by_index(&mut self, cid: usize) {
        for ni in 0..self.components[cid].nodes.len() {
            let n = self.components[cid].nodes[ni];
            self.scratch[n] = 0.0;
        }
    }

    fn accumulate_external_weighted_sum_by_index(&mut self, cid: usize) {
        for ei in 0..self.components[cid].external_edges.len() {
            let e = self.components[cid].external_edges[ei];
            let src_val = self.state[e.src];
            let contribution = src_val * e.weight;
            self.scratch[e.dst] += contribution;
        }
    }

    fn accumulate_internal_weighted_sum_by_index(&mut self, cid: usize) {
        for ei in 0..self.components[cid].internal_edges.len() {
            let e = self.components[cid].internal_edges[ei];
            let src_val = self.state[e.src];
            let contribution = src_val * e.weight;
            self.scratch[e.dst] += contribution;
        }
    }

    fn commit_component_values_by_index(&mut self, cid: usize) {
        for ni in 0..self.components[cid].nodes.len() {
            let n = self.components[cid].nodes[ni];
            if self.node_kinds[n] != NodeKind::Sensor {
                let sum = self.scratch[n];
                self.state[n] = relu(sum);
            }
        }
    }

    fn activate_acyclic_component_by_index(&mut self, cid: usize) {
        self.zero_component_scratch_by_index(cid);
        self.accumulate_external_weighted_sum_by_index(cid);
        self.accumulate_internal_weighted_sum_by_index(cid);
        self.commit_component_values_by_index(cid);
    }

    fn activate_recurrent_component_by_index(&mut self, cid: usize) {
        let max_iter = self.config.recurrent_iterations;
        let epsilon = self.config.recurrent_epsilon;

        for _iter in 0..max_iter {
            self.zero_component_scratch_by_index(cid);
            self.accumulate_external_weighted_sum_by_index(cid);
            self.accumulate_internal_weighted_sum_by_index(cid);

            let mut max_delta = 0.0f64;
            for ni in 0..self.components[cid].nodes.len() {
                let n = self.components[cid].nodes[ni];
                if self.node_kinds[n] == NodeKind::Sensor {
                    continue;
                }
                let prev = self.state[n];
                let next = relu(self.scratch[n]);
                let delta = (next - prev).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                self.state[n] = next;
            }

            if max_delta <= epsilon {
                break;
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

        for i in 0..inputs.len() {
            let idx = self.input_indices[i];
            self.state[idx] = inputs[i];
        }

        Ok(())
    }

    pub fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, PhenomeError> {
        self.audit_log.clear();
        self.log_line("activate: begin".to_string());

        self.reset_non_sensor_state();
        self.set_inputs(inputs)?;

        for i in 0..self.active_component_indices.len() {
            let cid = self.active_component_indices[i];
            // SAFETY: we only read from self.components[cid] and write to self.state/scratch
            // which don't overlap with components. Use index-based access to avoid clone.
            let recurrent = self.components[cid].recurrent;

            self.log_line(format!("component {} recurrent={}", cid, recurrent));
            if recurrent {
                self.activate_recurrent_component_by_index(cid);
            } else {
                self.activate_acyclic_component_by_index(cid);
            }
        }

        let mut out = Vec::with_capacity(self.output_indices.len());
        for i in 0..self.output_indices.len() {
            let idx = self.output_indices[i];
            self.log_line(format!(
                "output {} ({:?}) = {:.12}",
                idx, self.node_ids[idx], self.state[idx]
            ));
            out.push(self.state[idx]);
        }

        self.log_line("activate: end".to_string());
        Ok(out)
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
    use assert_approx_eq::assert_approx_eq;

    fn base_genome() -> (InnovationTracker, Genome) {
        let mut t = InnovationTracker::new();
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
                ..ActivationConfig::default()
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
        let mut t = InnovationTracker::new();
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
        let mut t = InnovationTracker::new();
        let g = Genome::minimal_fully_connected(2, 1, &mut t, |_i, _o| 1.0);

        let mut p = Phenome::from_genome(&g).unwrap();
        let out = p.activate(&[2.0, -3.0]).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.0).abs() < 1e-12); // relu((2*1) + (-3*1))
    }

    fn get_innovation(g: &Genome, in_node: NodeId, out_node: NodeId) -> Innovation {
        let key = ConnectionKey { in_node, out_node };
        g.connection_to_innovation.get(&key).copied().unwrap()
    }

    fn get_add_node_mutation(g: &Genome, in_node: NodeId, out_node: NodeId) -> Mutation {
        let innovation = get_innovation(g, in_node, out_node);
        Mutation::AddNode {
            split_innovation: innovation,
        }
    }

    fn get_disable_mutation(g: &Genome, in_node: NodeId, out_node: NodeId) -> Mutation {
        let innovation = get_innovation(&g, in_node, out_node);
        Mutation::DisableConnection { innovation }
    }

    fn get_add_connection_mutation(in_node: NodeId, out_node: NodeId, weight: f64) -> Mutation {
        Mutation::AddConnection {
            in_node,
            out_node,
            weight,
        }
    }

    fn get_weight_adjust_mutation(
        g: &Genome,
        in_node: NodeId,
        out_node: NodeId,
        delta: f64,
    ) -> Mutation {
        let innovation = get_innovation(g, in_node, out_node);
        Mutation::PerturbWeight { innovation, delta }
    }

    fn genome_sample_feed_forward_1() -> Genome {
        let mut t = InnovationTracker::new();
        let mut g = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.);

        let new_node_mutations = vec![
            get_add_node_mutation(&g, NodeId(0), NodeId(3)),
            get_add_node_mutation(&g, NodeId(1), NodeId(3)),
        ];

        g = g.apply_mutations(&mut t, &new_node_mutations).unwrap();

        let disable_mutations = vec![
            get_disable_mutation(&g, NodeId(0), NodeId(2)),
            get_disable_mutation(&g, NodeId(1), NodeId(2)),
        ];

        g = g.apply_mutations(&mut t, &disable_mutations).unwrap();

        let add_connection_mutations = vec![
            get_add_connection_mutation(NodeId(0), NodeId(5), 0.6),
            get_add_connection_mutation(NodeId(5), NodeId(2), 0.4),
        ];

        g = g
            .apply_mutations(&mut t, &add_connection_mutations)
            .unwrap();

        let weight_adjust_mutations = vec![
            get_weight_adjust_mutation(&g, NodeId(0), NodeId(4), -1.1),
            get_weight_adjust_mutation(&g, NodeId(1), NodeId(5), -1.8),
            get_weight_adjust_mutation(&g, NodeId(4), NodeId(3), 0.6),
            get_weight_adjust_mutation(&g, NodeId(5), NodeId(3), -0.9),
        ];

        g.apply_mutations(&mut t, &weight_adjust_mutations).unwrap()
    }

    #[test]
    fn feed_forward() {
        let genome = genome_sample_feed_forward_1();
        let mut p = Phenome::from_genome(&genome).unwrap();
        // p.set_logging_enabled(true);
        // let mermaid = p.to_mermaid();
        // println!("{}", mermaid);
        let output = p.activate(&vec![0.5, -0.2]).unwrap();
        println!("{:?}", output);
        // print all the logs
        for line in p.audit_log() {
            println!("{}", line);
        }
        assert!((output[0] - 0.184).abs() < 1e-12);
        assert!((output[1] - 0.0).abs() < 1e-12);
    }

    fn dead_ends_1() -> Genome {
        let mut t = InnovationTracker::new();
        let mut g = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.);

        let add_node_mutations = vec![
            get_add_node_mutation(&g, NodeId(0), NodeId(3)),
            get_add_node_mutation(&g, NodeId(0), NodeId(2)),
            get_add_node_mutation(&g, NodeId(1), NodeId(3)),
        ];

        g = g.apply_mutations(&mut t, &add_node_mutations).unwrap();

        let mutations_rest = vec![
            get_disable_mutation(&g, NodeId(1), NodeId(2)),
            get_disable_mutation(&g, NodeId(0), NodeId(5)),
            get_disable_mutation(&g, NodeId(5), NodeId(2)),
            get_disable_mutation(&g, NodeId(1), NodeId(6)),
            get_disable_mutation(&g, NodeId(6), NodeId(3)),
            get_add_connection_mutation(NodeId(6), NodeId(4), 0.),
            get_add_connection_mutation(NodeId(4), NodeId(5), 0.),
            get_add_connection_mutation(NodeId(4), NodeId(2), 0.),
            get_add_connection_mutation(NodeId(1), NodeId(4), 0.),
        ];

        g.apply_mutations(&mut t, &mutations_rest).unwrap()
    }

    #[test]
    fn test_dead_ends() {
        let genome = dead_ends_1();
        let mut p = Phenome::from_genome(&genome).unwrap();
        p.set_logging_enabled(true);
        let mermaid = p.to_mermaid(false);
        println!("{}", mermaid);
        let output = p.activate(&vec![0.5, -0.2]).unwrap();
        println!("{:?}", output);

        for line in p.audit_log() {
            println!("{}", line);
        }

        // component 2 is a dead source, component 5 is a dead end, make sure they are not in the logs
        assert!(!p
            .audit_log()
            .iter()
            .any(|line| line.contains("component 2")));
        assert!(!p
            .audit_log()
            .iter()
            .any(|line| line.contains("component 5")));
    }

    fn genome_sample_recurrent_1() -> Genome {
        let mut t = InnovationTracker::new();
        let mut g = Genome::minimal_fully_connected(2, 1, &mut t, |_i, _o| 0.);
        let mutations = vec![
            get_add_node_mutation(&g, NodeId(0), NodeId(2)),
            get_add_node_mutation(&g, NodeId(1), NodeId(2)),
        ];
        g = g.apply_mutations(&mut t, &mutations).unwrap();

        let mutations = vec![
            get_disable_mutation(&g, NodeId(0), NodeId(3)),
            get_add_connection_mutation(NodeId(1), NodeId(2), 0.0),
            get_add_node_mutation(&g, NodeId(1), NodeId(2)),
        ];

        g = g.apply_mutations(&mut t, &mutations).unwrap();

        let mutations = vec![
            get_add_connection_mutation(NodeId(5), NodeId(4), 0.0),
            get_add_connection_mutation(NodeId(4), NodeId(3), 0.0),
            get_add_connection_mutation(NodeId(0), NodeId(4), 0.0),
            get_add_connection_mutation(NodeId(3), NodeId(5), 0.0),
            get_disable_mutation(&g, NodeId(4), NodeId(2)),
            get_disable_mutation(&g, NodeId(1), NodeId(5)),
        ];
        g = g.apply_mutations(&mut t, &mutations).unwrap();

        let mutations = vec![
            get_weight_adjust_mutation(&g, NodeId(1), NodeId(4), -1.8),
            get_weight_adjust_mutation(&g, NodeId(0), NodeId(4), -0.8),
            get_weight_adjust_mutation(&g, NodeId(3), NodeId(2), 0.9),
            get_weight_adjust_mutation(&g, NodeId(4), NodeId(3), 0.1),
            get_weight_adjust_mutation(&g, NodeId(5), NodeId(2), -0.4),
            get_weight_adjust_mutation(&g, NodeId(3), NodeId(5), 0.5),
            get_weight_adjust_mutation(&g, NodeId(5), NodeId(4), -0.1),
        ];

        g.apply_mutations(&mut t, &mutations).unwrap()
    }

    #[test]
    fn test_recurrent() {
        let genome = genome_sample_recurrent_1();
        let mut p = Phenome::from_genome_with_config(
            &genome,
            ActivationConfig {
                recurrent_iterations: 100,
                recurrent_epsilon: 1e-8,
                logging_enabled: true,
            },
        )
        .unwrap();

        // let mermaid = p.to_mermaid(false);
        // println!("{}", mermaid);

        let output = p.activate(&vec![-0.9, 0.6]).unwrap();
        println!("{:?}", output);
        assert_approx_eq!(output[0], 0.016716);

        // for line in p.audit_log() {
        //     println!("{}", line);
        // }

        // the recurrent component should converge to a fixed point
        assert!(p
            .audit_log()
            .iter()
            .any(|line| line.contains("recurrent converged at iter")));
    }
}
