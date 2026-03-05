use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Innovation(pub u64);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Sensor,
    Hidden,
    Output,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ConnectionKey {
    pub in_node: NodeId,
    pub out_node: NodeId,
}

impl fmt::Display for ConnectionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.in_node.0, self.out_node.0)
    }
}

impl FromStr for ConnectionKey {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(format!("expected 'in:out', got '{}'", s));
        }
        let in_node = parts[0]
            .parse::<u32>()
            .map_err(|e| format!("invalid in_node: {}", e))?;
        let out_node = parts[1]
            .parse::<u32>()
            .map_err(|e| format!("invalid out_node: {}", e))?;
        Ok(ConnectionKey {
            in_node: NodeId(in_node),
            out_node: NodeId(out_node),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub key: ConnectionKey,
    pub innovation: Innovation,
    pub weight: f64,
    pub enabled: bool,
}

fn serialize_connection_key_map<S>(
    map: &HashMap<ConnectionKey, Innovation>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut ser_map = serializer.serialize_map(Some(map.len()))?;
    for (key, value) in map {
        ser_map.serialize_entry(&key.to_string(), value)?;
    }
    ser_map.end()
}

fn deserialize_connection_key_map<'de, D>(
    deserializer: D,
) -> Result<HashMap<ConnectionKey, Innovation>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let string_map: HashMap<String, Innovation> = HashMap::deserialize(deserializer)?;
    string_map
        .into_iter()
        .map(|(k, v)| {
            ConnectionKey::from_str(&k)
                .map(|ck| (ck, v))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub nodes: BTreeMap<NodeId, NodeKind>,
    // Sorted by innovation for O(n) alignment.
    pub connections_by_innovation: BTreeMap<Innovation, ConnectionGene>,
    // Fast duplicate checking by structural key.
    #[serde(
        serialize_with = "serialize_connection_key_map",
        deserialize_with = "deserialize_connection_key_map"
    )]
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

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ParentFitness {
    Left,
    Right,
    Equal,
}
