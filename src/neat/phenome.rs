use std::collections::VecDeque;
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

#[derive(PartialEq, PartialOrd, Clone, Copy, Hash, Eq, Debug)]
pub struct NodeIndex(pub usize);

impl NodeIndex {
    pub fn inc(self) -> NodeIndex {
        NodeIndex(self.0 + 1)
    }
}

#[derive(Copy, Clone)]
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
    can_come_from_sensor: Option<bool>,
    depth: usize,
    inputs: Vec<(NodeIndex, f64)>
}

fn can_come_from_sensor(visited: &mut FxHashMap<NodeIndex, VisitedRecord>, nodes: &NodeMap, edges: &Vec<Edge>, current_node: NodeIndex, current_depth: usize) -> (Option<bool>, Vec<(NodeIndex, f64)>) {
    let node = &nodes[current_node];
    println!("start processing {}:{} with depth {}", current_node.0, node.id.0, current_depth);
    let mut inputs = vec![];
    let res = 
        if node.node_type == NodeType::Sensor {
            println!("{}:{} is a sensor", current_node.0, node.id.0);
            Some(true)
        } else if node.inputs.len() == 0 {
            println!("{}:{} has no inputs", current_node.0, node.id.0);
            Some(false)
        } else {
            node.inputs.iter().fold(None, |acc, elem| {
                let edge = &edges[elem.0];
                println!("\tfor node {}:{}, processing input {}:{}", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);

                match visited.get(&edge.source) {
                    Some(v) => {
                        println!("\tfor node {}:{}, already visited index {}:{}", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);
                        match v.can_come_from_sensor {
                            Some(b) => {
                                println!("\tfor node {}:{}, index {}:{} can_come_from_sensor is {}", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0, b);
                                if b {
                                    inputs.push((edge.source, edge.weight));
                                }
                                acc.map(|a| a || b).or(Some(b))
                            },
                            None => {
                                println!("\tfor node {}:{}, index {}:{} can_come_from_sensor is not yet set", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);
                                acc
                            }
                        }
                    },
                    None => {
                        println!("\tfor node {}:{}, not yet visited index {}:{}, initialising with None", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);
                        let v = VisitedRecord{can_come_from_sensor: None, depth: current_depth, inputs: Vec::new()};
                        visited.insert(edge.source, v);
                        let (b_opt, local_inputs) = can_come_from_sensor(visited, nodes, edges, edge.source, current_depth + 1);
                        let v_mut = visited.get_mut(&edge.source).unwrap();
                        // println!("setting index {}:{} can_come_from_sensor to {}", edge.source.0, nodes[edge.source].id.0, b);
                        v_mut.can_come_from_sensor = b_opt;
                        v_mut.inputs = local_inputs;
                        match b_opt {
                            Some(b) => {
                                if b {
                                    inputs.push((edge.source, edge.weight));
                                }
                                acc.map(|a| a || b).or(Some(b))
                            },
                            None => {
                                acc
                            }
                        }
                    }
                }

            })
        };
    (res, inputs)
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

struct VisitedRecord2 {
    can_come_from_sensor: NodeStatus,
    depth: usize,
    inputs: Vec<(NodeIndex, f64, NodeStatus)>
}

// impl VisitedRecord2 {
//     fn plus(&self, )
// }
//TODO use the Edges wrapper instead of Vec<Edge>
fn can_come_from_sensor2(visited: &mut FxHashMap<NodeIndex, VisitedRecord2>, nodes: &NodeMap, edges: &Vec<Edge>, current_node: NodeIndex, current_depth: usize) -> (NodeStatus, Vec<(NodeIndex, f64, NodeStatus)>) {
    let node = &nodes[current_node];
    println!("start processing {}:{} with depth {}", current_node.0, node.id.0, current_depth);
    let mut inputs: Vec<(NodeIndex, f64, NodeStatus)> = vec![];
    let res : NodeStatus = 
        if node.node_type == NodeType::Sensor {
            println!("{}:{} is a sensor", current_node.0, node.id.0);
            NodeStatus::Yes
        } else if node.inputs.len() == 0 {
            println!("{}:{} has no inputs", current_node.0, node.id.0);
            NodeStatus::No
        } else {
            node.inputs.iter().fold(NodeStatus::Unknown, |acc, elem| {
                let edge = &edges[elem.0];
                println!("\tfor node {}:{}, processing input {}:{}", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);

                match visited.get(&edge.source) {
                    Some(v) => {
                        println!("\tfor node {}:{}, already visited index {}:{}", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);
                        match v.can_come_from_sensor {
                            NodeStatus::Unknown => {
                                println!("\tfor node {}:{}, index {}:{} can_come_from_sensor is uknown", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);
                                inputs.push((edge.source, edge.weight, NodeStatus::Unknown));
                                acc.plus(&NodeStatus::Unknown)
                            },
                            NodeStatus::Yes => {
                                inputs.push((edge.source, edge.weight, NodeStatus::Yes));
                                acc.plus(&NodeStatus::Yes)
                            },
                            NodeStatus::No => {
                                acc.plus(&NodeStatus::No)
                            }
                        }
                    },
                    None => {
                        println!("\tfor node {}:{}, not yet visited index {}:{}, initialising with None", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);
                        let v = VisitedRecord2{can_come_from_sensor: NodeStatus::Unknown, depth: current_depth, inputs: Vec::new()};
                        visited.insert(edge.source, v);
                        let (b_opt, local_inputs) = can_come_from_sensor2(visited, nodes, edges, edge.source, current_depth + 1);
                        let v_mut = visited.get_mut(&edge.source).unwrap();
                        v_mut.can_come_from_sensor = b_opt;
                        v_mut.inputs = local_inputs;
                        match b_opt {
                            NodeStatus::Unknown => {
                                println!("\tfor node {}:{}, index {}:{} can_come_from_sensor is uknown", edge.target.0, node.id.0, edge.source.0, nodes[edge.source].id.0);
                                inputs.push((edge.source, edge.weight, NodeStatus::Unknown));
                                acc.plus(&NodeStatus::Unknown)
                            },
                            NodeStatus::Yes => {
                                inputs.push((edge.source, edge.weight, NodeStatus::Yes));
                                acc.plus(&NodeStatus::Yes)
                            },
                            NodeStatus::No => {
                                acc.plus(&NodeStatus::No)
                            }
                        }
                    }
                }

            })
        };
    (res, inputs)
}

fn get_activation_order(nodes: &NodeMap, edges: &Vec<Edge>, n_sensor_nodes: usize, n_output_nodes: usize) -> Vec<(NodeIndex, Vec<(NodeIndex, f64)>)> {
    let mut visited: FxHashMap<NodeIndex,VisitedRecord> = FxHashMap::default();
    for i in n_sensor_nodes .. n_sensor_nodes + n_output_nodes {
        let v = VisitedRecord{can_come_from_sensor: None, depth: 0, inputs: vec![]};
        let node_id = NodeId(i);
        let node_index = nodes.get_index_of(&node_id).unwrap();
        visited.insert(node_index, v);
        let (b, local_inputs) = can_come_from_sensor(&mut visited, nodes, edges, node_index, 1);
        let v_mut = visited.get_mut(&node_index).unwrap();
        v_mut.can_come_from_sensor = b;
        v_mut.inputs = local_inputs;
    }

    //try again for all the nodes that didnt get set
    // let unset_node_indices = visited.iter().filter(|(_, v)| v.can_come_from_sensor == None).map(|(k, _)| *k).collect_vec();

    // for node_index in unset_node_indices {
    //     let (b, local_inputs) = can_come_from_sensor(&mut visited, nodes, edges, node_index, 1);
    //     let v_mut = visited.get_mut(&node_index).unwrap();
    //     v_mut.can_come_from_sensor = b;
    //     v_mut.inputs = local_inputs;
    // }

    println!("visited is initialised");
    for (k, v) in visited.iter() {
        println!("{}:{} can_come_from_sensor: {:?}, depth: {}, inputs: {:?}", k.0, nodes[*k].id.0, v.can_come_from_sensor, v.depth, v.inputs);
    }
    println!("end of data");

    visited.into_iter()
    // .filter(|(_, v)| v.can_come_from_sensor == Some(true) && v.inputs.len() > 0)
    .sorted_by(|a, b| b.1.depth.cmp(&a.1.depth))
    .map(|(node_index, v)| (node_index, v.inputs))
    .collect()
}    

fn get_activation_order2(nodes: &NodeMap, edges: &Vec<Edge>, n_sensor_nodes: usize, n_output_nodes: usize) -> Vec<(NodeIndex, Vec<(NodeIndex, f64)>)> {
    let mut visited: FxHashMap<NodeIndex,VisitedRecord2> = FxHashMap::default();
    for i in n_sensor_nodes .. n_sensor_nodes + n_output_nodes {
        let v = VisitedRecord2{can_come_from_sensor: NodeStatus::Unknown, depth: 0, inputs: vec![]};
        let node_id = NodeId(i);
        let node_index = nodes.get_index_of(&node_id).unwrap();
        visited.insert(node_index, v);
        let (b, local_inputs) = can_come_from_sensor2(&mut visited, nodes, edges, node_index, 1);
        let v_mut = visited.get_mut(&node_index).unwrap();
        v_mut.can_come_from_sensor = b;
        v_mut.inputs = local_inputs;
    }

    println!("visited is initialised");
    for (k, v) in visited.iter() {
        println!("{}:{} can_come_from_sensor: {:?}, depth: {}, inputs: {:?}", k.0, nodes[*k].id.0, v.can_come_from_sensor, v.depth, v.inputs);
    }
    println!("end of data");

    //try again for all the nodes that didnt get set
    let unset_node_indices = visited.iter().filter(|(_, v)| v.can_come_from_sensor == NodeStatus::Unknown).map(|(k, _)| *k).collect_vec();

    for node_index in unset_node_indices {
        let (b, local_inputs) = can_come_from_sensor2(&mut visited, nodes, edges, node_index, 1);
        let v_mut = visited.get_mut(&node_index).unwrap();
        v_mut.can_come_from_sensor = b;
        v_mut.inputs = local_inputs;
    }

    println!("visited is rerun");
    for (k, v) in visited.iter() {
        println!("{}:{} can_come_from_sensor: {:?}, depth: {}, inputs: {:?}", k.0, nodes[*k].id.0, v.can_come_from_sensor, v.depth, v.inputs);
    }
    println!("end of data");

    visited.into_iter()
    // .filter(|(_, v)| v.can_come_from_sensor == Some(true) && v.inputs.len() > 0)
    .sorted_by(|a, b| b.1.depth.cmp(&a.1.depth))
    .map(|(node_index, v)| (node_index, v.inputs.iter().map(|(i, w, _)| (*i, *w)).collect()))
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
        
        // let activation_order = get_activation_order2(&nodes, &edges, genome.n_sensor_nodes, genome.n_output_nodes);
        let activation_order = vec![];
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
    use std::collections::{HashSet, VecDeque};

    use itertools::Itertools;
    use rustc_hash::FxHashMap;

    use crate::neat::{genome::{Gene, GeneExt, Genome}, phenome::Phenome};

    use super::{NodeIndex, NodeStatus, NodeType};

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
            
        ], 2, 2)
    }

    #[test]
    fn test_dead_ends() {
        let genome = genome_sample_dead_ends();
        let phenome =  Phenome::create_from_genome(&genome);
        phenome.print_full_mermaid_graph();

        
        println!("activation order");
        for (k, v) in phenome.activation_order {
            print!("{} <- ", k.0);
            for (i, w) in &v {
                print!("{}({:.4})", i.0, w);
            }
            println!("");
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

    #[test]
    fn test_dead_ends_2() {
        let genome = genome_sample_dead_ends();
        let phenome =  Phenome::create_from_genome(&genome);
        phenome.print_full_mermaid_graph();

        let mut visisted: Vec<VisitRecord> = (0..phenome.nodes.len()).map(|_|VisitRecord::new(NodeStatus::Unknown)).collect_vec();
        let mut to_check: VecDeque<NodeIndex> = VecDeque::with_capacity(phenome.nodes.len());
        for node_index in phenome.outputs.iter() {
            to_check.push_back(*node_index);
            visisted[node_index.0].visited = true;
        }

        let mut loop_count = 0;
        let mut yes_count = 0;
        while let Some(node_index) = to_check.pop_front() {
            // println!("processing node {}:{} with status {:?}", node_index.0, phenome[node_index].id.0, visisted[node_index.0]);
            let node = &phenome[node_index];

            if node.node_type == NodeType::Sensor {
                // println!("{}:{} is a sensor", node_index.0, node.id.0);
                yes_count += 1;
                visisted[node_index.0].status = NodeStatus::Yes;
                visisted[node_index.0].yes_order = yes_count;
            } else if node.inputs.len() == 0 {
                // println!("{}:{} has no inputs", node_index.0, node.id.0);
                visisted[node_index.0].status = NodeStatus::No;
            } else {
                let new_status = node.inputs.iter().fold(NodeStatus::Unknown, |acc_status, edge_index| {
                    let edge = &phenome.edges[*edge_index];
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
                break;
            }
            println!("to check: {:?}", to_check);
        }

        println!("loop count: {}", loop_count);
       
        for (k, v) in visisted.iter().enumerate() {
            println!("{}:{} status: {:?}", k, phenome[NodeIndex(k)].id.0, v);
        }

        println!("final results");
        visisted.into_iter().enumerate().filter_map(|(k, v)| {
            if v.status == NodeStatus::Yes {
                Some((k, v))
            } else {
                None
            }
        }).sorted_by(|a, b| a.1.yes_order.cmp(&b.1.yes_order)).for_each(|(k, v)| {
            println!("{}:{} depth: {}", k, phenome[NodeIndex(k)].id.0, v.yes_order);
        });
        
    }
}