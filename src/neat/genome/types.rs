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
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub nodes: BTreeMap<NodeId, NodeKind>,
    // Sorted by innovation for O(n) alignment.
    pub connections_by_innovation: BTreeMap<Innovation, ConnectionGene>,
    // Fast duplicate checking by structural key.
    pub connection_to_innovation: HashMap<ConnectionKey, Innovation>,
}

impl Genome {
    pub(super) fn insert_connection_gene(&mut self, gene: ConnectionGene) {
        self.connection_to_innovation
            .insert(gene.key.clone(), gene.innovation);
        self.connections_by_innovation.insert(gene.innovation, gene);
    }

    pub(super) fn validate_connection_endpoints(
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
