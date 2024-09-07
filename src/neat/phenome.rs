use std::ops::Index;
use std::ops::IndexMut;

use indexmap::IndexMap;
use itertools::Itertools;
use rustc_hash::FxHashMap;

use super::genome::Genome;
use super::genome::NodeId;

#[derive(PartialEq, Default, Clone, Debug)]
pub enum NodeType{
    Sensor,
    #[default] //TODO is a default really needed here?
    Hidden,
    Output,
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Hash, Eq)]
pub struct NodeIndex(pub usize);

impl NodeIndex {
    pub fn inc(self) -> NodeIndex {
        NodeIndex(self.0 + 1)
    }
}

#[derive(Clone)]
pub struct EdgeIndex(pub usize);

#[derive(Clone)]
pub struct Edge {
    pub source: NodeIndex,
    pub target: NodeIndex,
    pub weight: f64
}

#[derive(Clone)]
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

#[derive(Clone)]
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
        NodeMap(IndexMap::with_capacity_and_hasher(capacity, rustc_hash::FxBuildHasher::default()))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node> {
        self.0.values()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get_index_of(&self, node_id: &NodeId) -> Option<NodeIndex> {
        self.0.get_index_of(node_id).map(|i| NodeIndex(i))
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

#[derive(Clone)]
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

#[derive(Clone)]
pub struct Phenome{
    pub nodes: NodeMap,
    pub edges: Edges,
    pub activation_order: Vec<(NodeIndex, Vec<(NodeIndex, f64)>)>,
    pub inputs: Vec<NodeIndex>,
    pub outputs: Vec<NodeIndex>,
}

struct VisitedRecord {
    can_come_from_sensor: bool,
    depth: usize,
    inputs: Vec<(NodeIndex, f64)>
}

fn can_come_from_sensor(visited: &mut FxHashMap<NodeIndex, VisitedRecord>, nodes: &NodeMap, edges: &Vec<Edge>, current_node: NodeIndex, current_depth: usize) -> (bool, Vec<(NodeIndex, f64)>) {
    let node = &nodes[current_node];
    println!("start processing {}:{} with depth {}", current_node.0, node.id.0, current_depth);
    let mut inputs = vec![];
    let res = 
        if node.node_type == NodeType::Sensor {
            println!("{}:{} is a sensor", current_node.0, node.id.0);
            true
        } else if node.inputs.len() == 0 {
            println!("{}:{} has no inputs", current_node.0, node.id.0);
            false
        } else {
            node.inputs.iter().fold(false, |acc, elem| {
                let edge = &edges[elem.0];
                println!("processing input {}:? for node {}:{}", edge.source.0, edge.target.0, node.id.0);

                match visited.get(&edge.source) {
                    Some(v) => {
                        println!("already visited index {}:?", edge.source.0);
                        if v.can_come_from_sensor {
                            inputs.push((edge.source, edge.weight));
                        }
                        acc || v.can_come_from_sensor
                    },
                    None => {
                        println!("not yet visited index {}:?", edge.source.0);
                        let v = VisitedRecord{can_come_from_sensor: false, depth: current_depth, inputs: Vec::new()};
                        visited.insert(edge.source, v);
                        let (b, local_inputs) = can_come_from_sensor(visited, nodes, edges, edge.source, current_depth + 1);
                        let v_mut = visited.get_mut(&edge.source).unwrap();
                        v_mut.can_come_from_sensor = b;
                        v_mut.inputs = local_inputs;
                        if b {
                            inputs.push((edge.source, edge.weight));
                        }
                        acc || b
                    }
                }

            })
        };
    (res, inputs)
}

fn get_activation_order(nodes: &NodeMap, edges: &Vec<Edge>, n_sensor_nodes: usize, n_output_nodes: usize) -> Vec<(NodeIndex, Vec<(NodeIndex, f64)>)> {
    let mut visited: FxHashMap<NodeIndex,VisitedRecord> = FxHashMap::default();
    for i in n_sensor_nodes .. n_sensor_nodes + n_output_nodes {
        let v = VisitedRecord{can_come_from_sensor: false, depth: 0, inputs: vec![]};
        let node_id = NodeId(i);
        let node_index = nodes.get_index_of(&node_id).unwrap();
        visited.insert(node_index, v);
        let (b, local_inputs) = can_come_from_sensor(&mut visited, nodes, edges, node_index, 1);
        let v_mut = visited.get_mut(&node_index).unwrap();
        v_mut.can_come_from_sensor = b;
        v_mut.inputs = local_inputs;
    }

    visited.into_iter()
    .filter(|(_, v)| v.can_come_from_sensor && v.inputs.len() > 0)
    .sorted_by(|a, b| b.1.depth.cmp(&a.1.depth))
    .map(|(node_index, v)| (node_index, v.inputs))
    .collect()
}    

impl Phenome {
    pub fn create_from_genome(genome: &Genome) -> Phenome {
        let mut nodes = NodeMap::with_capacity(0); //TODO: capacity?
        let mut edges = Vec::with_capacity(genome.len());

        for (gene_key, gene_val) in genome.iter() {
            if gene_val.enabled {
                let in_node_id = gene_key.in_node_id;
                let out_node_id = gene_key.out_node_id;
                let weight = gene_val.weight;

                let in_node_index = nodes.get_or_create_node_index(genome, in_node_id);
                let out_node_index = nodes.get_or_create_node_index(genome, out_node_id);

                edges.push(Edge{source: in_node_index, target: out_node_index, weight});
            }
        }

        for (i, edge) in edges.iter().enumerate() {
            nodes[edge.target].inputs.push(EdgeIndex(i));
        }
        
        let activation_order = get_activation_order(&nodes, &edges, genome.n_sensor_nodes, genome.n_output_nodes);
        let edges = Edges(edges);

        let inputs = (0..genome.n_sensor_nodes).map(|i| nodes.get_index_of(&NodeId(i)).unwrap()).collect_vec();
        let outputs = (genome.n_sensor_nodes..genome.n_sensor_nodes + genome.n_output_nodes).map(|i| nodes.get_index_of(&NodeId(i)).unwrap()).collect_vec();
        Phenome{nodes, edges, activation_order, inputs, outputs}
    }
    
    pub fn activate(&mut self, sensor_values: &Vec<f64>) {
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
        for (i, node) in self.nodes.iter().enumerate() {
            let node_id = node.id;
            let node_type = match node.node_type {
                NodeType::Sensor => "S",
                NodeType::Hidden => "H",
                NodeType::Output => "O",
            };
            println!("{}[{}:{}/{}]", i, i, node_id.0, node_type);
        }

        for &(node_index, ref inputs) in &self.activation_order {
            // let node = &self.nodes[node_index];
            
            // let incoming_edges = self.phenome.edges_directed(node_index, Direction::Incoming);
            for (input_index, weight) in inputs {
                // let input_node = &self.nodes[*input_index];
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
    use crate::neat::{genome::{Gene, GeneExt, Genome}, phenome::Phenome};

    fn genome_sample_dead_ends() -> Genome {
        Genome::create(vec![
            Gene::create(0, 4, 0.0, 0, true),
            Gene::create(4, 2, 0.0, 1, true),
            Gene::create(7, 6, 0.0, 2, true),
            Gene::create(4, 6, 0.0, 3, true),
            Gene::create(4, 8, 0.0, 4, true),
            Gene::create(6, 5, 0.0, 5, true),
            Gene::create(5, 4, 0.0, 6, true),
            Gene::create(1, 5, 0.0, 7, true),
            Gene::create(5, 3, 0.0, 8, true),
            Gene::create(9, 7, 0.0, 8, true),
            Gene::create(7, 10, 0.0, 8, true),
            Gene::create(10, 9, 0.0, 8, true),
        ], 2, 2)
    }

    #[test]
    fn test_dead_ends() {
        let genome = genome_sample_dead_ends();
        let phenome =  Phenome::create_from_genome(&genome);
        phenome.print_full_mermaid_graph();

        
        println!("");
        for (k, v) in phenome.activation_order {
            print!("{} <- ", k.0);
            for (i, w) in &v {
                print!("{}({:.4})", i.0, w);
            }
            println!("");
        }
    }
}