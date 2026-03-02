use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NodeId(pub u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Innovation(pub u64);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum NodeKind {
    Sensor,
    Hidden,
    Output,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ConnectionKey {
    pub in_node: NodeId,
    pub out_node: NodeId,
}

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub key: ConnectionKey,
    pub innovation: Innovation,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct Genome {
    n_inputs: usize,
    n_outputs: usize,
    nodes: BTreeMap<NodeId, NodeKind>,
    // Sorted by innovation for O(n) alignment.
    connections_by_innovation: BTreeMap<Innovation, ConnectionGene>,
    // Fast duplicate checking by structural key.
    connection_to_innovation: HashMap<ConnectionKey, Innovation>,
}

#[derive(Debug, Clone)]
pub struct InnovationTracker {
    next_innovation: u64,
    next_node_id: u32,
}

#[derive(Debug, Clone)]
pub enum GenomeError {
    MissingNode(NodeId),
    InvalidOutputNode(NodeId),
    SelfLoop(NodeId),
    DuplicateConnection(ConnectionKey),
    UnknownInnovation(Innovation),
    ConnectionAlreadyDisabled(Innovation),
    MismatchedIo {
        left_inputs: usize,
        left_outputs: usize,
        right_inputs: usize,
        right_outputs: usize,
    },
    MismatchedNodeKind {
        node: NodeId,
        left: NodeKind,
        right: NodeKind,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ParentFitness {
    Left,
    Right,
    Equal,
}

#[derive(Debug, Clone)]
pub enum Mutation {
    AddConnection {
        in_node: NodeId,
        out_node: NodeId,
        weight: f64,
    },
    AddNode {
        split_innovation: Innovation,
    },
    PerturbWeight {
        innovation: Innovation,
        delta: f64,
    },
}

#[derive(Debug, Copy, Clone)]
pub struct DistanceCoefficients {
    pub excess: f64,
    pub disjoint: f64,
    pub weight: f64,
    pub small_genome_threshold: usize,
}

impl Default for DistanceCoefficients {
    fn default() -> Self {
        Self {
            excess: 1.0,
            disjoint: 1.0,
            weight: 0.4,
            small_genome_threshold: 20,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Alignment<'a> {
    Both(&'a ConnectionGene, &'a ConnectionGene),
    Left(&'a ConnectionGene),
    Right(&'a ConnectionGene),
}

impl InnovationTracker {
    pub fn new(start_node_id: NodeId) -> Self {
        Self {
            next_innovation: 1,
            next_node_id: start_node_id.0,
        }
    }

    pub fn next_connection_innovation(&mut self) -> Innovation {
        let id = Innovation(self.next_innovation);
        self.next_innovation += 1;
        id
    }

    pub fn next_hidden_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }
}

impl Genome {
    fn insert_connection_gene(&mut self, gene: ConnectionGene) {
        self.connection_to_innovation
            .insert(gene.key.clone(), gene.innovation);
        self.connections_by_innovation.insert(gene.innovation, gene);
    }

    pub fn minimal_fully_connected<F>(
        n_inputs: usize,
        n_outputs: usize,
        tracker: &mut InnovationTracker,
        mut initial_weight: F,
    ) -> Self
    where
        F: FnMut(NodeId, NodeId) -> f64,
    {
        let sensors = (0..n_inputs)
            .map(|i| (NodeId(i as u32), NodeKind::Sensor))
            .collect::<BTreeMap<_, _>>();

        let outputs = (0..n_outputs)
            .map(|o| (NodeId((n_inputs + o) as u32), NodeKind::Output))
            .collect::<BTreeMap<_, _>>();

        let mut nodes = sensors;
        nodes.extend(outputs);

        let mut g = Self {
            n_inputs,
            n_outputs,
            nodes,
            connections_by_innovation: BTreeMap::new(),
            connection_to_innovation: HashMap::new(),
        };

        for out_idx in 0..n_outputs {
            let out_node = NodeId((n_inputs + out_idx) as u32);
            for in_idx in 0..n_inputs {
                let in_node = NodeId(in_idx as u32);
                let weight = initial_weight(in_node, out_node);
                let key = ConnectionKey { in_node, out_node };
                let innovation = tracker.next_connection_innovation();
                let gene = ConnectionGene {
                    key: key.clone(),
                    innovation,
                    weight,
                    enabled: true,
                };
                g.insert_connection_gene(gene);
            }
        }

        g
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    pub fn connection_count(&self) -> usize {
        self.connections_by_innovation.len()
    }

    pub fn connection(&self, innovation: Innovation) -> Option<&ConnectionGene> {
        self.connections_by_innovation.get(&innovation)
    }

    pub fn innovations(&self) -> impl Iterator<Item = Innovation> + '_ {
        self.connections_by_innovation.keys().copied()
    }

    // TODO this function does work but does not retain the benefit of it.
    fn validate_connection_endpoints(
        &self,
        in_node: NodeId,
        out_node: NodeId,
    ) -> Result<(), GenomeError> {
        if in_node == out_node {
            return Err(GenomeError::SelfLoop(in_node));
        }

        if !self.nodes.contains_key(&in_node) {
            return Err(GenomeError::MissingNode(in_node));
        }

        let out_kind = self
            .nodes
            .get(&out_node)
            .ok_or(GenomeError::MissingNode(out_node))?;

        if *out_kind == NodeKind::Sensor {
            return Err(GenomeError::InvalidOutputNode(out_node));
        }

        Ok(())
    }

    pub fn with_added_connection(
        &self,
        tracker: &mut InnovationTracker,
        in_node: NodeId,
        out_node: NodeId,
        weight: f64,
    ) -> Result<Self, GenomeError> {
        self.validate_connection_endpoints(in_node, out_node)?;

        let key = ConnectionKey { in_node, out_node };
        if self.connection_to_innovation.contains_key(&key) {
            return Err(GenomeError::DuplicateConnection(key));
        }

        let innovation = tracker.next_connection_innovation();
        let gene = ConnectionGene {
            key: key.clone(),
            innovation,
            weight,
            enabled: true,
        };

        let mut next = self.clone();
        next.insert_connection_gene(gene);
        Ok(next)
    }

    pub fn with_added_node(
        &self,
        tracker: &mut InnovationTracker,
        split_innovation: Innovation,
    ) -> Result<Self, GenomeError> {
        let original = self
            .connections_by_innovation
            .get(&split_innovation)
            .ok_or(GenomeError::UnknownInnovation(split_innovation))?;

        if !original.enabled {
            return Err(GenomeError::ConnectionAlreadyDisabled(split_innovation));
        }

        let mut next = self.clone();

        if let Some(g) = next.connections_by_innovation.get_mut(&split_innovation) {
            g.enabled = false;
        }

        let new_node = tracker.next_hidden_node_id();
        next.nodes.insert(new_node, NodeKind::Hidden);

        let left_key = ConnectionKey {
            in_node: original.key.in_node,
            out_node: new_node,
        };
        let right_key = ConnectionKey {
            in_node: new_node,
            out_node: original.key.out_node,
        };

        let left_gene = ConnectionGene {
            key: left_key.clone(),
            innovation: tracker.next_connection_innovation(),
            weight: 1.0,
            enabled: true,
        };

        let right_gene = ConnectionGene {
            key: right_key.clone(),
            innovation: tracker.next_connection_innovation(),
            weight: original.weight,
            enabled: true,
        };

        next.insert_connection_gene(left_gene);
        next.insert_connection_gene(right_gene);

        Ok(next)
    }

    pub fn with_perturbed_weight(
        &self,
        innovation: Innovation,
        delta: f64,
    ) -> Result<Self, GenomeError> {
        let mut next = self.clone();
        let gene = next
            .connections_by_innovation
            .get_mut(&innovation)
            .ok_or(GenomeError::UnknownInnovation(innovation))?;

        gene.weight += delta;
        Ok(next)
    }

    pub fn apply_mutation(
        &self,
        tracker: &mut InnovationTracker,
        mutation: &Mutation,
    ) -> Result<Self, GenomeError> {
        match mutation {
            Mutation::AddConnection {
                in_node,
                out_node,
                weight,
            } => self.with_added_connection(tracker, *in_node, *out_node, *weight),
            Mutation::AddNode { split_innovation } => {
                self.with_added_node(tracker, *split_innovation)
            }
            Mutation::PerturbWeight { innovation, delta } => {
                self.with_perturbed_weight(*innovation, *delta)
            }
        }
    }

    pub fn apply_mutations(
        &self,
        tracker: &mut InnovationTracker,
        mutations: &[Mutation],
    ) -> Result<Self, GenomeError> {
        mutations
            .iter()
            .try_fold(self.clone(), |acc, m| acc.apply_mutation(tracker, m))
    }

    fn validate_crossover_compatibility(left: &Genome, right: &Genome) -> Result<(), GenomeError> {
        if left.n_inputs != right.n_inputs || left.n_outputs != right.n_outputs {
            return Err(GenomeError::MismatchedIo {
                left_inputs: left.n_inputs,
                left_outputs: left.n_outputs,
                right_inputs: right.n_inputs,
                right_outputs: right.n_outputs,
            });
        }
        Ok(())
    }

    fn merged_nodes_for_crossover(
        left: &Genome,
        right: &Genome,
    ) -> Result<BTreeMap<NodeId, NodeKind>, GenomeError> {
        let mut merged = left.nodes.clone();

        for (node, kind_right) in &right.nodes {
            match merged.get(node) {
                Some(kind_left) if kind_left != kind_right => {
                    return Err(GenomeError::MismatchedNodeKind {
                        node: *node,
                        left: *kind_left,
                        right: *kind_right,
                    });
                }
                Some(_) => {}
                None => {
                    merged.insert(*node, *kind_right);
                }
            }
        }

        Ok(merged)
    }

    fn inherit_unmatched<FEqual>(
        fitness: ParentFitness,
        is_left_gene: bool,
        choose_left_when_equal: &mut FEqual,
    ) -> bool
    where
        FEqual: FnMut() -> bool,
    {
        match fitness {
            ParentFitness::Left => is_left_gene,
            ParentFitness::Right => !is_left_gene,
            ParentFitness::Equal => {
                let choose_left = choose_left_when_equal();
                (is_left_gene && choose_left) || (!is_left_gene && !choose_left)
            }
        }
    }

    pub fn crossover<FMatch, FEqual, FDisabled>(
        left: &Genome,
        right: &Genome,
        fitness: ParentFitness,
        mut choose_left_matching: FMatch,
        mut choose_left_when_equal_for_unmatched: FEqual,
        mut enable_if_either_parent_disabled: FDisabled,
    ) -> Result<Self, GenomeError>
    where
        FMatch: FnMut(&ConnectionGene, &ConnectionGene) -> bool,
        FEqual: FnMut() -> bool,
        FDisabled: FnMut() -> bool,
    {
        Self::validate_crossover_compatibility(left, right)?;
        let nodes = Self::merged_nodes_for_crossover(left, right)?;

        let mut child = Self {
            n_inputs: left.n_inputs,
            n_outputs: left.n_outputs,
            nodes,
            connections_by_innovation: BTreeMap::new(),
            connection_to_innovation: HashMap::new(),
        };

        for_each_aligned_by_innovation(left, right, |aligned| match aligned {
            Alignment::Both(l, r) => {
                let mut inherited = if choose_left_matching(l, r) {
                    l.clone()
                } else {
                    r.clone()
                };

                if !l.enabled || !r.enabled {
                    inherited.enabled = enable_if_either_parent_disabled();
                }

                child.insert_connection_gene(inherited);
            }
            Alignment::Left(l) => {
                if Self::inherit_unmatched(fitness, true, &mut choose_left_when_equal_for_unmatched)
                {
                    child.insert_connection_gene(l.clone());
                }
            }
            Alignment::Right(r) => {
                if Self::inherit_unmatched(
                    fitness,
                    false,
                    &mut choose_left_when_equal_for_unmatched,
                ) {
                    child.insert_connection_gene(r.clone());
                }
            }
        });

        Ok(child)
    }
}

pub fn for_each_aligned_by_innovation<'a, F>(left: &'a Genome, right: &'a Genome, mut f: F)
where
    F: FnMut(Alignment<'a>),
{
    let mut l_it = left.connections_by_innovation.iter().peekable();
    let mut r_it = right.connections_by_innovation.iter().peekable();

    loop {
        match (l_it.peek(), r_it.peek()) {
            (Some((l_innov, l_gene)), Some((r_innov, r_gene))) => {
                if l_innov == r_innov {
                    f(Alignment::Both(l_gene, r_gene));
                    l_it.next();
                    r_it.next();
                } else if l_innov < r_innov {
                    f(Alignment::Left(l_gene));
                    l_it.next();
                } else {
                    f(Alignment::Right(r_gene));
                    r_it.next();
                }
            }
            (Some((_, l_gene)), None) => {
                f(Alignment::Left(l_gene));
                l_it.next();
            }
            (None, Some((_, r_gene))) => {
                f(Alignment::Right(r_gene));
                r_it.next();
            }
            (None, None) => break,
        }
    }
}

pub fn genetic_distance(left: &Genome, right: &Genome, c: DistanceCoefficients) -> f64 {
    let max_left = left.innovations().max().unwrap_or(Innovation(0));
    let max_right = right.innovations().max().unwrap_or(Innovation(0));

    let mut excess = 0usize;
    let mut disjoint = 0usize;
    let mut matching = 0usize;
    let mut weight_diff_sum = 0.0f64;

    for_each_aligned_by_innovation(left, right, |aligned| match aligned {
        Alignment::Both(l, r) => {
            matching += 1;
            weight_diff_sum += (l.weight - r.weight).abs();
        }
        Alignment::Left(l) => {
            if l.innovation > max_right {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
        Alignment::Right(r) => {
            if r.innovation > max_left {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
    });

    let n = left.connection_count().max(right.connection_count());
    let n_norm = if n < c.small_genome_threshold {
        1.0
    } else {
        n as f64
    };

    let avg_weight_diff = if matching == 0 {
        0.0
    } else {
        weight_diff_sum / (matching as f64)
    };

    c.excess * (excess as f64) / n_norm
        + c.disjoint * (disjoint as f64) / n_norm
        + c.weight * avg_weight_diff
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_tracker() -> InnovationTracker {
        InnovationTracker::new(NodeId(3)) // with 2 inputs + 1 output, next hidden starts at 3
    }

    fn base_genome(tracker: &mut InnovationTracker) -> Genome {
        Genome::minimal_fully_connected(2, 1, tracker, |_i, _o| 0.5)
    }

    #[test]
    fn every_new_connection_mutation_gets_unique_innovation() {
        let mut t = InnovationTracker::new(NodeId(4));
        let base = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.0);

        let key = ConnectionKey {
            in_node: NodeId(2),
            out_node: NodeId(3),
        };

        let g1 = base
            .with_added_connection(&mut t, key.in_node, key.out_node, 0.1)
            .unwrap();

        let g2 = base
            .with_added_connection(&mut t, key.in_node, key.out_node, -0.7)
            .unwrap();

        let i1 = g1.connection_to_innovation.get(&key).copied().unwrap();
        let i2 = g2.connection_to_innovation.get(&key).copied().unwrap();

        assert_ne!(i1, i2);
    }

    #[test]
    fn splitting_same_parent_connection_twice_does_not_reuse_split_history() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);

        let split_innov = g.innovations().next().unwrap();

        let g1s = g.with_added_node(&mut t, split_innov).unwrap();
        let g2s = g.with_added_node(&mut t, split_innov).unwrap();

        let added_1 = g1s
            .innovations()
            .filter(|i| !g.innovations().any(|x| x == *i))
            .collect::<Vec<_>>();

        let added_2 = g2s
            .innovations()
            .filter(|i| !g.innovations().any(|x| x == *i))
            .collect::<Vec<_>>();

        assert_eq!(added_1.len(), 2);
        assert_eq!(added_2.len(), 2);
        assert_ne!(added_1, added_2);
    }

    #[test]
    fn alignment_by_innovation_is_correct() {
        let mut t = base_tracker();
        let g1 = base_genome(&mut t);
        let i = g1.innovations().next().unwrap();
        let g2 = g1.with_perturbed_weight(i, 1.0).unwrap();

        let mut both = 0usize;
        let mut left = 0usize;
        let mut right = 0usize;

        for_each_aligned_by_innovation(&g1, &g2, |a| match a {
            Alignment::Both(_, _) => both += 1,
            Alignment::Left(_) => left += 1,
            Alignment::Right(_) => right += 1,
        });

        assert_eq!(both, 2);
        assert_eq!(left, 0);
        assert_eq!(right, 0);
    }
    #[test]
    fn add_connection_adds_new_gene_and_duplicate_fails() {
        let mut t = InnovationTracker::new(NodeId(4));
        let g = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.0);

        let g2 = g
            .with_added_connection(&mut t, NodeId(2), NodeId(3), 0.9)
            .unwrap();

        assert_eq!(g2.connection_count(), g.connection_count() + 1);

        let err = g2
            .with_added_connection(&mut t, NodeId(2), NodeId(3), 0.1)
            .err()
            .unwrap();

        match err {
            GenomeError::DuplicateConnection(_) => {}
            _ => panic!("expected DuplicateConnection"),
        }
    }

    #[test]
    fn add_node_splits_connection_correctly() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);

        let split_innov = g.innovations().next().unwrap();
        let old_weight = g.connection(split_innov).unwrap().weight;

        let g2 = g.with_added_node(&mut t, split_innov).unwrap();

        let old_gene = g2.connection(split_innov).unwrap();
        assert!(!old_gene.enabled);

        let mut created = g2
            .innovations()
            .filter(|i| *i != split_innov)
            .collect::<Vec<_>>();
        created.sort();

        assert_eq!(created.len(), 3); // other original + two newly inserted

        let new_edges = g2
            .connections_by_innovation
            .values()
            .filter(|cg| cg.key.in_node == NodeId(0) || cg.key.out_node == NodeId(2))
            .collect::<Vec<_>>();

        assert!(new_edges.iter().any(|g| (g.weight - 1.0).abs() < 1e-12));
        assert!(new_edges
            .iter()
            .any(|g| (g.weight - old_weight).abs() < 1e-12));
    }

    #[test]
    fn perturb_weight_keeps_innovation() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);
        let i = g.innovations().next().unwrap();
        let old = g.connection(i).unwrap().weight;

        let g2 = g.with_perturbed_weight(i, 0.25).unwrap();
        let new_gene = g2.connection(i).unwrap();

        assert_eq!(new_gene.innovation, i);
        assert!((new_gene.weight - (old + 0.25)).abs() < 1e-12);
    }

    #[test]
    fn genetic_distance_matches_expected_counts() {
        let mut t = base_tracker();
        let mut g1 = base_genome(&mut t);
        let mut g2 = base_genome(&mut t);

        // Force divergence:
        // g1: split first connection => adds two genes.
        let split = g1.innovations().next().unwrap();
        g1 = g1.with_added_node(&mut t, split).unwrap();

        // g2: perturb one matching weight only.
        let match_i = g2.innovations().next().unwrap();
        g2 = g2.with_perturbed_weight(match_i, 0.5).unwrap();

        let d = genetic_distance(
            &g1,
            &g2,
            DistanceCoefficients {
                excess: 1.0,
                disjoint: 1.0,
                weight: 1.0,
                small_genome_threshold: 20,
            },
        );

        assert!(d > 0.0);
    }

    #[test]
    fn apply_mutations_pipeline_works() {
        let mut t = InnovationTracker::new(NodeId(4));
        let g = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.0);

        let first = g.innovations().next().unwrap();
        let out = g
            .apply_mutations(
                &mut t,
                &[
                    Mutation::PerturbWeight {
                        innovation: first,
                        delta: 0.2,
                    },
                    Mutation::AddConnection {
                        in_node: NodeId(2),
                        out_node: NodeId(3),
                        weight: 0.7,
                    },
                ],
            )
            .unwrap();

        assert_eq!(out.connection_count(), g.connection_count() + 1);
        assert!((out.connection(first).unwrap().weight - 0.2).abs() < 1e-12);
    }

    #[test]
    fn crossover_inherits_matching_and_fitter_unmatched_genes() {
        let mut t = InnovationTracker::new(NodeId(4));
        let base = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.0);

        let first = base.innovations().next().unwrap();
        let left = base
            .with_added_connection(&mut t, NodeId(2), NodeId(3), 0.9)
            .unwrap();
        let right = base.with_perturbed_weight(first, -0.2).unwrap();

        let child = Genome::crossover(
            &left,
            &right,
            ParentFitness::Left,
            |_l, _r| false, // take right for matching genes
            || false,       // irrelevant for this test
            || true,        // keep enabled when disabled in either parent
        )
        .unwrap();

        assert_eq!(child.connection_count(), left.connection_count());

        let added_key = ConnectionKey {
            in_node: NodeId(2),
            out_node: NodeId(3),
        };
        assert!(child.connection_to_innovation.contains_key(&added_key));
        assert!((child.connection(first).unwrap().weight + 0.2).abs() < 1e-12);
    }

    #[test]
    fn crossover_respects_disabled_gene_policy() {
        let mut t = base_tracker();
        let base = base_genome(&mut t);
        let split = base.innovations().next().unwrap();

        let left = base.with_added_node(&mut t, split).unwrap(); // split gene disabled
        let right = base.clone(); // split gene enabled

        let child_disabled = Genome::crossover(
            &left,
            &right,
            ParentFitness::Equal,
            |_l, _r| false, // inherit matching from right (enabled in source)
            || true,
            || false, // force disabled when either parent disabled
        )
        .unwrap();

        assert!(!child_disabled.connection(split).unwrap().enabled);

        let child_enabled = Genome::crossover(
            &left,
            &right,
            ParentFitness::Equal,
            |_l, _r| false,
            || true,
            || true, // force enabled when either parent disabled
        )
        .unwrap();

        assert!(child_enabled.connection(split).unwrap().enabled);
    }

    #[test]
    fn add_node_rejects_unknown_and_already_disabled_connection() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);
        let split = g.innovations().next().unwrap();

        let unknown_err = g
            .with_added_node(&mut t, Innovation(u64::MAX))
            .err()
            .unwrap();
        match unknown_err {
            GenomeError::UnknownInnovation(_) => {}
            _ => panic!("expected UnknownInnovation"),
        }

        let g2 = g.with_added_node(&mut t, split).unwrap();
        let disabled_err = g2.with_added_node(&mut t, split).err().unwrap();
        match disabled_err {
            GenomeError::ConnectionAlreadyDisabled(i) => assert_eq!(i, split),
            _ => panic!("expected ConnectionAlreadyDisabled"),
        }
    }

    #[test]
    fn add_connection_rejects_invalid_endpoints() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);

        match g
            .with_added_connection(&mut t, NodeId(0), NodeId(0), 0.1)
            .err()
            .unwrap()
        {
            GenomeError::SelfLoop(NodeId(0)) => {}
            _ => panic!("expected SelfLoop"),
        }

        match g
            .with_added_connection(&mut t, NodeId(2), NodeId(0), 0.1)
            .err()
            .unwrap()
        {
            GenomeError::InvalidOutputNode(NodeId(0)) => {}
            _ => panic!("expected InvalidOutputNode"),
        }

        match g
            .with_added_connection(&mut t, NodeId(999), NodeId(2), 0.1)
            .err()
            .unwrap()
        {
            GenomeError::MissingNode(NodeId(999)) => {}
            _ => panic!("expected MissingNode"),
        }
    }
}
