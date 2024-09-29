use std::collections::HashSet;
use std::collections::VecDeque;
use std::ops::Index;
use std::ops::IndexMut;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use serde::{Serialize, Deserialize};
use indexmap::IndexMap;
use itertools::Itertools;

use crate::neat::graph::tarjan_scc;

use super::genome::Genome;
use super::genome::NodeId;

#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum NodeType{
    Sensor,
    Hidden,
    Output,
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Hash, Eq, Debug, Serialize, Deserialize)]
pub struct NodeIndex(pub usize);

impl NodeIndex {
    pub fn inc(self) -> NodeIndex {
        NodeIndex(self.0 + 1)
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct EdgeIndex(pub usize);

#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: NodeIndex,
    pub target: NodeIndex,
    pub weight: f64
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    pub value: f64,
    pub inputs: Vec<EdgeIndex>,
    pub node_type: NodeType,
    pub id: NodeId
}

impl Node {
    pub fn create(node_type: NodeType, id: NodeId) -> Node {
        Node{
            value: 0.,
            inputs: Vec::new(),
            node_type,
            id
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NodeMap(IndexMap<NodeId, Node, rustc_hash::FxBuildHasher>);

impl NodeMap {
    fn get_or_create_node_index(&mut self, genome: &Genome, node_id: NodeId) -> NodeIndex {
        let make_node = || -> Node {
            let node_type =
                if node_id.0 < genome.n_sensor_nodes {
                    NodeType::Sensor
                } else if node_id.0 < genome.n_sensor_nodes + genome.n_output_nodes {
                    NodeType::Output
                } else {
                    NodeType::Hidden
                };
            
            Node::create(node_type, node_id)
        };
    
        match self.get_index_of(&node_id) {
            Some(node_index) => {
                // println!("already seen node {}, fetching index {}", node_id.0, node_index);
                node_index
            },
            None => {
                let node_index = NodeIndex(self.0.len());
                self.0.insert(node_id, make_node());
                // println!("new node {}, inserting index {}", node_id.0, node_index.0);
                node_index
            }
        }
    }

    fn with_capacity(capacity: usize) -> NodeMap {
        NodeMap(IndexMap::with_capacity_and_hasher(capacity, rustc_hash::FxBuildHasher))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node> {
        self.0.values()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get_index_of(&self, node_id: &NodeId) -> Option<NodeIndex> {
        self.0.get_index_of(node_id).map(NodeIndex)
    }
}

impl Index<NodeIndex> for NodeMap {
    type Output = Node;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

impl IndexMut<NodeIndex> for NodeMap {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Edges(Vec<Edge>);

impl Edges {
    pub fn iter(&self) -> impl Iterator<Item = &Edge> {
        self.0.iter()
    }
}

impl Index<EdgeIndex> for Edges {
    type Output = Edge;
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Phenome{
    pub nodes: NodeMap,
    pub edges: Edges,
    pub activation_order: Vec<(NodeIndex, Vec<(NodeIndex, f64)>)>,
    pub inputs: Vec<NodeIndex>,
    pub outputs: Vec<NodeIndex>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum NodeStatus {
    Unknown,
    Yes,
    No
}

impl NodeStatus {
    fn plus(&self, other: &NodeStatus) -> NodeStatus {
        match (self, other) {
            (NodeStatus::Yes, _) => NodeStatus::Yes,
            (_, NodeStatus::Yes) => NodeStatus::Yes,
            (NodeStatus::Unknown, _) => NodeStatus::Unknown,
            (_, NodeStatus::Unknown) => NodeStatus::Unknown,
            _ => NodeStatus::No
        }
    }
}

#[derive(Debug)]
struct VisitRecord {
    status: NodeStatus,
    process_count: usize,
    yes_order: usize,
    visited: bool
}

impl VisitRecord {
    fn new(status: NodeStatus) -> VisitRecord {
        VisitRecord{status, process_count: 0, yes_order: 0, visited: false}
    }
}

fn get_activation_order(nodes: &NodeMap, edges: &[Edge], outputs: &[NodeIndex]) -> Vec<(NodeIndex, Vec<(NodeIndex, f64)>)> {
    let mut visisted: Vec<VisitRecord> = (0..nodes.len()).map(|_|VisitRecord::new(NodeStatus::Unknown)).collect_vec();
    let mut to_check: VecDeque<NodeIndex> = VecDeque::with_capacity(nodes.len());
    for node_index in outputs.iter() {
        to_check.push_back(*node_index);
        visisted[node_index.0].visited = true;
    }

    let mut loop_count = 0;
    let mut yes_count = 0;
    while let Some(node_index) = to_check.pop_front() {
        // println!("processing node {}:{} with status {:?}", node_index.0, phenome[node_index].id.0, visisted[node_index.0]);
        let node = &nodes[node_index];

        if node.node_type == NodeType::Sensor {
            // println!("{}:{} is a sensor", node_index.0, node.id.0);
            yes_count += 1;
            visisted[node_index.0].status = NodeStatus::Yes;
            visisted[node_index.0].yes_order = yes_count;
        } else if node.inputs.is_empty() {
            // println!("{}:{} has no inputs", node_index.0, node.id.0);
            visisted[node_index.0].status = NodeStatus::No;
        } else {
            let new_status = node.inputs.iter().fold(NodeStatus::Unknown, |acc_status, edge_index| {
                let edge = &edges[edge_index.0];
                if !visisted[edge.source.0].visited {
                    // println!("{}:{} has not yet been visited", edge.source.0, phenome[edge.source].id.0);
                    to_check.push_back(edge.source);
                    visisted[edge.source.0].visited = true;
                }

                // println!("{}:{} can come from {}:{} with status {:?}", node_index.0, node.id.0, edge.source.0, phenome[edge.source].id.0, visisted[edge.source.0]);
                acc_status.plus(&visisted[edge.source.0].status)
            });

            if visisted[node_index.0].status != new_status {
                visisted[node_index.0].status = new_status;
                // println!("{}:{} status changed from {:?} to {:?}", node_index.0, node.id.0, visisted[node_index.0].status, new_status);
                if new_status == NodeStatus::Yes {
                    yes_count += 1;
                    visisted[node_index.0].yes_order = yes_count;
                }
            } else if visisted[node_index.0].status == NodeStatus::Unknown {
                if visisted[node_index.0].process_count > 2 {
                    visisted[node_index.0].status = NodeStatus::No;
                    // println!("{}:{} status changed from {:?} to No", node_index.0, node.id.0, visisted[node_index.0].status);
                } else {
                    visisted[node_index.0].process_count += 1;
                    to_check.push_back(node_index);
                    // println!("{}:{} status is still unknown", node_index.0, node.id.0);
                }
            } else {
                // println!("{}:{} status is still {:?} which is very unexpected", node_index.0, node.id.0, visisted[node_index.0].status);
            }
            
        }

        loop_count += 1;
        if loop_count >= 16383 {
            panic!("loop count exceeded");
        }
    }

    (0 .. visisted.len()).filter_map(|i| {
        let node = &nodes[NodeIndex(i)];
        if visisted[i].status == NodeStatus::Yes && node.node_type != NodeType::Sensor {
            let node = &nodes[NodeIndex(i)];
            let inputs = node.inputs.iter().filter_map(|edge_index| {
                let edge = &edges[edge_index.0];
                if visisted[edge.source.0].status == NodeStatus::Yes {
                    Some((edge.source, edge.weight))
                } else {
                    None
                }
            }).collect_vec();
            Some((visisted[i].yes_order, (NodeIndex(i), inputs)))
        } else {
            None
        }
    })
    .sorted_by_key(|x|x.0)
    .map(|x|x.1)
    .collect_vec()
}    
impl Phenome {
    

    pub fn create_from_genome(genome: &Genome) -> Phenome {
        let mut nodes = NodeMap::with_capacity(0); //TODO: capacity?
        let mut edges = Vec::with_capacity(genome.len());

        for (gene_key, gene_val) in genome.iter() {
            // if gene_val.enabled {
                let in_node_id = gene_key.in_node_id;
                let out_node_id = gene_key.out_node_id;
                let weight = gene_val.weight;

                let in_node_index = nodes.get_or_create_node_index(genome, in_node_id);
                let out_node_index = nodes.get_or_create_node_index(genome, out_node_id);

                edges.push(Edge{source: in_node_index, target: out_node_index, weight});
            // }
        }

        for (i, edge) in edges.iter().enumerate() {
            nodes[edge.target].inputs.push(EdgeIndex(i));
        }

        //make sure sensor and output nodes are represented in the nodes map
        //TODO this might be redundant given that all sensor and output related connections are retained and thus would be represented by the loop above
        for i in 0..genome.n_sensor_nodes + genome.n_output_nodes {
            nodes.get_or_create_node_index(genome, NodeId(i));
        }
        
        let inputs = (0..genome.n_sensor_nodes).map(|i| nodes.get_index_of(&NodeId(i)).unwrap()).collect_vec();
        let outputs = (genome.n_sensor_nodes..genome.n_sensor_nodes + genome.n_output_nodes).map(|i| nodes.get_index_of(&NodeId(i)).unwrap()).collect_vec();

        let activation_order = get_activation_order(&nodes, &edges, &outputs);
        let edges = Edges(edges);

        
        Phenome{nodes, edges, activation_order, inputs, outputs}
    }

    pub fn create_from_genome3(genome: &Genome) {
        struct MapEntry {
            inputs: Vec<(NodeId, f64)>,
            outputs: Vec<(NodeId, f64)>
        }

        let mut node_map: FxHashMap<NodeId, MapEntry> = FxHashMap::default();

        for (gene_key, gene_val) in genome.iter() {
            let in_node_id = gene_key.in_node_id;
            let out_node_id = gene_key.out_node_id;
            let weight = gene_val.weight;

            let entry = node_map.entry(out_node_id).or_insert(MapEntry{inputs: Vec::new(), outputs: Vec::new()});
            entry.inputs.push((in_node_id, weight));

            let entry = node_map.entry(in_node_id).or_insert(MapEntry{inputs: Vec::new(), outputs: Vec::new()});
            entry.outputs.push((out_node_id, weight));
        }

        // Function to perform BFS and return all reachable nodes from a given set of start nodes
        fn bfs_reachable(nodes: &FxHashMap<NodeId, MapEntry>, initial_nodes: &[NodeId], next_getter: impl Fn(&MapEntry) -> Vec<(NodeId, f64)>) -> FxHashSet<NodeId> {
            let mut visited = FxHashSet::default();
            let mut queue = VecDeque::new();

            for &start in initial_nodes {
                queue.push_back(start);
                visited.insert(start);
            }

            while let Some(node_index) = queue.pop_front() {
                let node_entry = nodes.get(&node_index).unwrap();
                let next_nodes = next_getter(node_entry);
                for (new_index, _) in next_nodes {
                    if visited.insert(new_index) {
                        queue.push_back(new_index);
                    }
                }
            }

            visited
        }

        let sensor_node_ids = (0..genome.n_sensor_nodes).map(NodeId).collect_vec();
        let output_node_ids = (genome.n_sensor_nodes..genome.n_sensor_nodes + genome.n_output_nodes).map(NodeId).collect_vec();

        let get_inputs = |entry: &MapEntry| entry.inputs.clone();
        let get_outputs = |entry: &MapEntry| entry.outputs.clone();

        let reachable_from_inputs = bfs_reachable(&node_map, &output_node_ids, get_inputs);
        let reachable_to_outputs = bfs_reachable(&node_map, &sensor_node_ids, get_outputs);

        let mut node_id_to_index: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut node_ids: Vec<NodeId> = Vec::new();

        for (index, node_id) in reachable_from_inputs.intersection(&reachable_to_outputs).cloned().enumerate() {
            node_id_to_index.insert(node_id, index);
            node_ids.push(node_id);
        }

        let enumerated_graph: Vec<Vec<(usize, f64)>> = 
            node_ids.iter().map(|node_id| {
                let entry = node_map.get(node_id).unwrap();
                let inputs = entry.inputs.iter().filter_map(|(in_node_id, weight)| {
                    match node_id_to_index.get(in_node_id) {
                        Some(&in_index) => Some((in_index, *weight)),
                        None => None
                    }
                }).collect_vec();
                inputs
            }).collect_vec();

        let topo_order = tarjan_scc(&enumerated_graph);
        let activation_order = topo_order.iter().flat_map(|component| {
            component.iter().filter_map(|&node_index| {
                let inputs = &enumerated_graph[node_index];
                if !inputs.is_empty() {
                    Some((node_index, inputs.clone()))
                } else {
                    None
                }
            })
        }).collect_vec();

        // println!("topo order: {:?}", topo_order);
        // println!("enumerated graph: {:?}", enumerated_graph);
        // println!("activation order: {:?}", activation_order);

        println!("graph TD");

        fn get_node_type(node_id: NodeId, genome: &Genome) -> NodeType {
            if node_id.0 < genome.n_sensor_nodes {
                NodeType::Sensor
            } else if node_id.0 < genome.n_sensor_nodes + genome.n_output_nodes {
                NodeType::Output
            } else {
                NodeType::Hidden
            }
        }

        for (node_index, node_id) in node_ids.iter().enumerate() {
            let node_type = match get_node_type(*node_id, genome) {
                NodeType::Sensor => "S",
                NodeType::Hidden => "H",
                NodeType::Output => "O",
            };
            
            println!("{}[{}:{}/{}]", node_index, node_index, node_id.0, node_type);
        }

        for component in topo_order {
            for node_index in component {
                let inputs = &enumerated_graph[node_index];
                for (input_index, weight) in inputs {
                    println!("{} -->|{:.4}|{}", input_index, weight, node_index);
                }
            }
        }
    }

    pub fn activate(&mut self, sensor_values: &[f64]) {
        fn relu(x: f64) -> f64 {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }

        // fn sigmoid(x: f64) -> f64 {
        //     1.0 / (1.0 + (-4.9 * x).exp())
        // }

        for (i, &input) in sensor_values.iter().enumerate() {
            let node_index = self.inputs[i];
            self[node_index].value = input;
        }

        for &(node_index, ref inputs) in &self.activation_order {
            let node = &self[node_index];
            debug_assert!(node.node_type != NodeType::Sensor);
            let active_sum = inputs.iter().fold(0., |acc, &(input_index, w)| {
                acc + w * self[input_index].value
            });
            self.nodes[node_index].value = relu(active_sum);
        }
    }

    pub fn clear_values(&mut self) {
        for node in self.nodes.0.values_mut() {
            node.value = 0.;
        }
    }

    pub fn try_node_id(&self, node_id: NodeId) -> Option<&Node> {
        self.nodes.0.get(&node_id)
    }

    pub fn print_mermaid_graph(&self) {
        println!("graph TD");
        //print the sensor nodes
        for node_index in self.inputs.iter() {
            let node = &self.nodes[*node_index];
            println!("{}[{}:{}/S]", node_index.0, node_index.0, node.id.0);
        }

        //print the rest
        for node_index in self.activation_order.iter().map(|x|x.0) {
            let node = &self.nodes[node_index];
            let node_type = match node.node_type {
                NodeType::Sensor => "S",
                NodeType::Hidden => "H",
                NodeType::Output => "O",
            };
            println!("{}[{}:{}/{}]", node_index.0, node_index.0, node.id.0, node_type);
        }

        for &(node_index, ref inputs) in &self.activation_order {
            for (input_index, weight) in inputs {
                println!("{} -->|{:.4}|{}", input_index.0, weight, node_index.0);
            }
        }
    }

    pub fn print_full_mermaid_graph(&self) {
        println!("graph TD");
        for (i, node) in self.nodes.iter().enumerate() {
            let node_id = node.id;
            let node_type = match node.node_type {
                NodeType::Sensor => "S",
                NodeType::Hidden => "H",
                NodeType::Output => "O",
            };
            println!("{}[{}:{}/{}]", i, i, node_id.0, node_type);
        }

        for edge in self.edges.iter() {
            println!("{} -->|{:.4}|{}", edge.source.0, edge.weight, edge.target.0);
        }
    }

    
}

impl Index<NodeIndex> for Phenome {
    type Output = Node;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index]
    }
}

impl Index<EdgeIndex> for Phenome {
    type Output = Edge;
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[index]
    }
}

impl IndexMut<NodeIndex> for Phenome {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

mod tests {
    use crate::neat::phenome::NodeType;

    #[test]
    fn test_dead_ends() {
        use crate::neat::{genome::{Gene, GeneExt, Genome}, phenome::Phenome};
        let genome = 
            Genome::create(vec![
                Gene::create(0, 4, 0.0),
                Gene::create(4, 2, 0.0),
                Gene::create(7, 6, 0.0),
                Gene::create(4, 6, 0.0),
                Gene::create(4, 8, 0.0),
                Gene::create(6, 5, 0.0),
                Gene::create(5, 4, 0.0),
                Gene::create(1, 5, 0.0),
                Gene::create(5, 3, 0.0),
                Gene::create(9, 7, 0.0),
                
            ], 2, 2);

        let phenome =  Phenome::create_from_genome(&genome);
        phenome.print_full_mermaid_graph();

        phenome.print_mermaid_graph();

        Phenome::create_from_genome3(&genome);
    }
    
    
    
}