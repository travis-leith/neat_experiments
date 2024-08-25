use itertools::Itertools;
use rand::{distributions::{Distribution, Uniform}, RngCore};
// use std::collections::{HashSet, HashMap};
use rustc_hash::{FxHashMap, FxHashSet};
use crate::neat::vector;

// #[derive(PartialEq, PartialOrd, Copy, Clone)]
#[derive(Clone)]
pub struct InnovationNumber(pub usize);
impl InnovationNumber {
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(PartialEq, Clone)]
enum NodeType{
    Sensor,
    Hidden,
    Output,
}

#[derive(Clone)]
pub struct Node{
    // id: NodeId,
    value: f64,
    is_active: bool,
    has_active_inputs: bool,
    input_connection_ids: FxHashSet<usize>,
    input_node_ids: FxHashSet<usize>,
    active_sum: f64,
    // is_output: bool,
    // node_type: NodeType,
}

impl Node {
    fn create(node_type: NodeType) -> Node {
        Node{
            value: 0.,
            is_active: node_type == NodeType::Sensor,
            has_active_inputs: false,
            input_connection_ids: FxHashSet::default(),
            input_node_ids: FxHashSet::default(),
            active_sum: 0.,
            // node_type: node_type
        }
    }
}

#[derive(Clone)]
pub struct Connection {
    pub in_node_id: usize,
    pub out_node_id: usize,
    pub weight: f64,
    pub innovation: InnovationNumber,
    enabled: bool
}

#[derive(Clone)]
pub struct Network {
    pub has_bias_node: bool,
    pub genome: Vec<Connection>,
    pub out_of_order: bool, //indicates that some mutation has happened that has re-used a known innovation number, and has led to an out of order genome
    pub n_sensor_nodes: usize,
    pub n_output_nodes: usize,
    pub nodes: Vec<Node>,
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

impl Network {

    pub fn create_from_genome(n_sensor_nodes: usize, n_output_nodes: usize, genome: Vec<Connection>, has_bias_node: bool) -> Network {
        //TODO remove connections involving dead end nodes
        let mut max_node_id = 0;

        for conn in genome.iter() {
            if conn.in_node_id > max_node_id {
                max_node_id = conn.in_node_id
            }
            if conn.out_node_id > max_node_id {
                max_node_id = conn.out_node_id
            }
        }

        let hidden_start = n_sensor_nodes + n_output_nodes;
        let mut nodes: Vec<Node> = (0 .. max_node_id + 1).map(|i:usize|{
            if i < n_sensor_nodes {
                Node::create(NodeType::Sensor)
            } else if i < hidden_start {
                Node::create(NodeType::Output)
            } else {
                Node::create(NodeType::Hidden)
            }
        }).collect();

        for (i, conn) in genome.iter().enumerate() {
            if conn.enabled {
                nodes[conn.out_node_id].input_node_ids.insert(conn.in_node_id);
                nodes[conn.out_node_id].input_connection_ids.insert(i);
            }
        }
        
        for i in 1 .. genome.len() {
            // if genome[i].innovation.0 <= genome[i-1].innovation.0 {
            //     println!("left {}", genome[i].innovation.0);
            //     println!("right {}", genome[i - 1].innovation.0);
            //     println!("found one")
            // }
            debug_assert!(genome[i].innovation.0 > genome[i-1].innovation.0)
        }
        Network {
            has_bias_node,
            n_sensor_nodes,
            n_output_nodes,
            genome,
            nodes,
            out_of_order: false
        }

    }

    pub fn init(rng: &mut dyn RngCore, n_sensor_nodes: usize, n_output_nodes: usize, has_bias_node: bool) -> Network {
        let between = Uniform::from(-1.0..1.0);
        let n_total_nodes = n_sensor_nodes + n_output_nodes;
        let mut nodes = Vec::with_capacity(n_total_nodes);


        for _ in 0 .. n_sensor_nodes {
            let n = Node::create(NodeType::Sensor);
            nodes.push(n);
        }

        for _ in 0 .. n_output_nodes {
            let n = Node::create(NodeType::Output);
            nodes.push(n);
        }
        
        let n_connections = n_sensor_nodes * n_output_nodes;
        let mut genome : Vec<Connection> = Vec::with_capacity(n_connections);

        for out_node_ind in 0..n_output_nodes {
            let out_node_id = out_node_ind + n_sensor_nodes;
            for in_node_ind in 0..n_sensor_nodes {
                let in_node_id = in_node_ind;
                let innovation_number = out_node_ind * n_sensor_nodes + in_node_ind;
                let conn = Connection{
                    in_node_id,
                    out_node_id,
                    weight: between.sample(rng),
                    innovation: InnovationNumber(innovation_number),
                    enabled: true
                };
                genome.push(conn);
                let conn_id = innovation_number;//during init, conneciton index happens to allign with innovation number
                nodes[n_sensor_nodes + out_node_ind].input_connection_ids.insert(conn_id);
                nodes[n_sensor_nodes + out_node_ind].input_node_ids.insert(in_node_id);
            }
        }

        Network {
            has_bias_node,
            genome,
            n_sensor_nodes,
            n_output_nodes,
            nodes,
            out_of_order: false
        }
    }

    fn activation_pulse(&mut self) -> bool {
        let nodes = &mut self.nodes[..];
        for i in self.n_sensor_nodes..nodes.len() {
            nodes[i].has_active_inputs = false;
            nodes[i].active_sum = 0.;

            for i_conn in &nodes[i].input_connection_ids {
                let conn = &self.genome[*i_conn];
                let in_node = conn.in_node_id;
                if nodes[in_node].is_active && conn.enabled {
                    let to_add = conn.weight * nodes[in_node].value;
                    nodes[i].has_active_inputs = true;
                    nodes[i].active_sum += to_add;
                }
            }
        }
        
        // for conn in self.genome.iter() {
        //     let in_node = &self.nodes[conn.in_node_id];
        //     if in_node.is_active && conn.enabled {
		// 		let to_add = conn.weight * in_node.value;
		// 		let out_node = &mut self.nodes[conn.out_node_id];
		// 		out_node.has_active_inputs = true;
		// 		out_node.active_sum += to_add;
		// 	}
        // }

        let mut all_active = true;

        for i_node in self.n_sensor_nodes .. self.nodes.len() {
            let node = &mut self.nodes[i_node];
            node.value = relu(node.active_sum);
            if node.has_active_inputs {
                node.is_active = true;
            } else if node.input_connection_ids.len() > 0 {
                all_active = false;
            }
        }

        all_active
    }

    pub fn activate(&mut self, sensor_values: &Vec<f64>) {
        let bias_offset = if self.has_bias_node {1} else {0};
        debug_assert!(sensor_values.len() + bias_offset == self.n_sensor_nodes, "sensor values not the right length for network");
        // set the sensor values
        self.nodes[0].value = 1.; //TODO this does not need to happen on each activation
        for (i, value) in sensor_values.into_iter().enumerate() {
            self.nodes[i + bias_offset].value = *value;
        }
        
        let mut remaining_iterations = 5;
        loop {
            if remaining_iterations == 0 {
                // let dodgy_conns =
                //     self.genome.iter().group_by(|conn|{
                //         (conn.in_node_id, conn.out_node_id)
                //     }).into_iter()
                //     .map(|(a,b)|{
                //         (a, b.collect_vec().len())
                //     }).filter(|(a, b)| *b > 1)
                //     .collect_vec();
                // println!("genome size: {}", self.genome.len());
                // panic!("Too many iterations :(")
                for node_ix in self.n_sensor_nodes .. (self.n_sensor_nodes + self.n_output_nodes) {
                    self.nodes[node_ix].value = 0. //this will punsh networks with dangling nodes
                }
                break
            } else {
                let all_activated = self.activation_pulse();
                if all_activated {
                    break
                } else {
                    remaining_iterations -= 1;
                }
            }
        }      
    }

    pub fn get_output(&self) -> Vec<f64> {
        self.nodes[self.n_sensor_nodes .. self.n_sensor_nodes + self.n_output_nodes].iter().map(|n| n.value).collect()
    }
}


pub fn add_connection(mut network: Network, in_node_id: usize, out_node_id: usize, weight: f64, global_innovation: &mut InnovationNumber, innovation_record: &mut FxHashMap<(usize, usize), InnovationNumber>) -> Network {
    debug_assert!(in_node_id != out_node_id, "Tried to add a connection where input is the same node as output");
    debug_assert!(in_node_id < network.n_sensor_nodes || in_node_id >= (network.n_sensor_nodes + network.n_output_nodes), "Tried to add a connection that inputs from an output node");
    debug_assert!(out_node_id >= network.n_sensor_nodes, "Tried to add a connection that outputs to a sensor node");
    debug_assert!(out_node_id < network.nodes.len(), "Tried to add a connection that outputs beyond node count");

    let out_node = &mut network.nodes[out_node_id];
    if !out_node.input_node_ids.contains(&in_node_id) {
        let new_key = (in_node_id, out_node_id);
        // let new_innnov = InnovationNumber(global_innovation.len());

        let innov_number = 
            match innovation_record.try_insert(new_key, global_innovation.clone()) {
                Ok(i) => {
                    // println!("adding key: {in_node_id},{out_node_id} and value {}", global_innovation.0);
                    global_innovation.inc();
                    i.clone()
                },
                Err(x) => {
                    network.out_of_order = true;
                    x.entry.get().clone()
                }
            }
            ;
        // println!("selected number {}", innov_number.0);
        let new_conn = Connection{
            in_node_id,
            out_node_id,
            weight,
            innovation: innov_number,
            enabled: true
        };
        let new_conn_ix = network.genome.len();
        
        out_node.input_connection_ids.insert(new_conn_ix);
        out_node.input_node_ids.insert(in_node_id);
        network.genome.push(new_conn);
        
        network
    } else {
        network
    }
}

pub fn add_node(mut network: Network, existing_conn_index: usize, global_innovation: &mut InnovationNumber, innovation_record: &mut FxHashMap<(usize, usize), InnovationNumber>) -> Network {
    let in_node_id = network.genome[existing_conn_index].in_node_id;
    let out_node_id = network.genome[existing_conn_index].out_node_id;
    let weight = network.genome[existing_conn_index].weight;

    let genome = &mut network.genome[..];
    let nodes = &mut network.nodes[..];
    let existing_conn = &mut genome[existing_conn_index];
    if !existing_conn.enabled {
        return network
    }
    
    let new_node_id = nodes.len();
    let output_node = &mut nodes[existing_conn.out_node_id];
    
    let new_hidden_node = Node{
        has_active_inputs: false,
        input_connection_ids: FxHashSet::default(),
        input_node_ids: FxHashSet::default(),
        is_active: false,
        active_sum: 0.,
        value: 0.,
    };

    existing_conn.enabled = false;
    output_node.input_connection_ids.remove(&existing_conn_index);
    output_node.input_node_ids.remove(&in_node_id);

    network.nodes.push(new_hidden_node);

    let new_network = add_connection(network, in_node_id, new_node_id, 1., global_innovation, innovation_record);

    add_connection(new_network, new_node_id, out_node_id, weight, global_innovation, innovation_record)

}

#[derive(Clone)]
pub struct Organism {
    pub network: Network,
    pub fitness: usize
}

impl Organism {
    pub fn init(rng: &mut dyn RngCore , n_sensor_nodes: usize, n_output_nodes: usize, has_bias_node: bool) -> Organism {
        Organism { network: Network::init(rng, n_sensor_nodes, n_output_nodes, has_bias_node), fitness: 0 }
    }

    pub fn activate(&mut self, sensor_values: &Vec<f64>) {
        self.network.activate(sensor_values);
    }
}

use vector::AllignedPair;

pub fn cross_over(rng: &mut dyn RngCore, organism_1: &Organism, organism_2: &Organism) -> Network {
    // let gene_pairs: Vec<(Option<&Connection>, Option<&Connection>)> = align_genes(organism_1, organism_2);
    let between = Uniform::from(0.0..1.0);
    let mut choose_gene = |pair: AllignedPair<Connection>| {
        let r = between.sample(rng);
        match pair{
            AllignedPair::HasBoth(left, right) => {
                if r > 0.5 {
                    Some((*left).clone())
                } else {
                    Some((*right).clone())
                }
            },
            AllignedPair::HasLeft(left) => {
                if organism_1.fitness > organism_2.fitness {
                    Some((*left).clone())
                } else if organism_1.fitness < organism_2.fitness {
                    None
                } else {
                    Some((*left).clone()) //prefer left. do not randomly choose as this could result in dead end nodes if not done consistently
                }
            },
            AllignedPair::HasRight(right) => {
                if organism_1.fitness > organism_2.fitness {
                    None
                } else if organism_1.fitness < organism_2.fitness {
                    Some((*right).clone())
                } else {
                    None //prefer left. do not randomly choose as this could result in dead end nodes if not done consistently
                }
            }
        }
    };

    for i in 1 .. organism_1.network.genome.len() {
        // if organism_1.network.genome[i].innovation.0 <= organism_1.network.genome[i-1].innovation.0 {
        //     println!("left {}", organism_1.network.genome[i].innovation.0);
        //     println!("right {}", organism_1.network.genome[i - 1].innovation.0);
        //     println!("found one")
        // }
        debug_assert!(organism_1.network.genome[i].innovation.0 > organism_1.network.genome[i-1].innovation.0)
    }
    for i in 1 .. organism_2.network.genome.len() {
        // if organism_2.network.genome[i].innovation.0 <= organism_2.network.genome[i-1].innovation.0 {
        //     println!("left {}", organism_2.network.genome[i].innovation.0);
        //     println!("right {}", organism_2.network.genome[i - 1].innovation.0);
        //     println!("found one")
        // }
        debug_assert!(organism_2.network.genome[i].innovation.0 > organism_2.network.genome[i-1].innovation.0)
    }

    let get_id = |conn:&Connection| conn.innovation.0;
    let new_genome = 
        vector::allign(&organism_1.network.genome, &organism_2.network.genome, &get_id, &mut choose_gene)
        .into_iter().flatten().collect_vec();

    // new_genome.sort_by_key(|conn|conn.innovation.0);
        

    debug_assert!(organism_1.network.n_output_nodes == organism_2.network.n_output_nodes, "Organisms with mismatching output size cannot be crossed");
    debug_assert!(organism_1.network.n_sensor_nodes == organism_2.network.n_sensor_nodes, "Organisms with mismatching input size cannot be crossed");
    debug_assert!(organism_1.network.has_bias_node == organism_2.network.has_bias_node, "Organisms with mismatching input size cannot be crossed");
    Network::create_from_genome(organism_1.network.n_sensor_nodes, organism_1.network.n_output_nodes, new_genome, organism_1.network.has_bias_node)
}

pub fn genome_distance(genome_1: &Vec<Connection>, genome_2: &Vec<Connection>, excess_coef: f64, disjoint_coef: f64, weight_diff_coef: f64) -> f64 {
    #[derive(PartialEq, PartialOrd)]
    enum ExcessSide {
        Left,
        Right,
        Neither
    }

    let mut total_weight_diff = 0.;
    let mut excess_side = ExcessSide::Neither;
    let mut excess_count = 0;
    let mut disjoint_count = 0;
    let mut n1 = 0;
    let mut n2 = 0;

    let mut increment_counters = |pair: AllignedPair<Connection>| {
        match pair {
            AllignedPair::HasBoth(left, right) => {
                n1 += 1;
                n2 += 1;
                excess_side = ExcessSide::Neither;
                disjoint_count = disjoint_count + excess_count;
                excess_count = 0;
                total_weight_diff = total_weight_diff + (left.weight - right.weight).abs();
            },
            AllignedPair::HasLeft(_) => {
                n1 +=1;
                match excess_side {
                    ExcessSide::Neither => {
                        excess_side = ExcessSide::Left;
                        excess_count = 1;
                    },
                    ExcessSide::Right => {
                        excess_side = ExcessSide::Left;
                        disjoint_count = disjoint_count + excess_count;
                        excess_count = 1;
                    },
                    ExcessSide::Left => {
                        excess_count += 1;
                    }
                }
            },
            AllignedPair::HasRight(_) => {
                n2 += 1;
                match excess_side {
                    ExcessSide::Neither => {
                        excess_side = ExcessSide::Right;
                        excess_count = 1;
                    },
                    ExcessSide::Right => {
                        excess_count += 1;
                    },
                    ExcessSide::Left => {
                        excess_side = ExcessSide::Right;
                        disjoint_count = disjoint_count + excess_count;
                        excess_count = 1;
                    }
                }
            }
        }
    };

    let get_id = |conn:&Connection| conn.innovation.0;
    vector::allign(genome_1, &genome_2, &get_id, &mut increment_counters);

    let n = std::cmp::max(n1, n2) as f64;
    let excess_term = excess_coef * (excess_count as f64) / n;
    let disjoint_term = disjoint_coef * (disjoint_count as f64) / n;
    let weight_term = weight_diff_coef * total_weight_diff / n;
    excess_term + disjoint_term + weight_term
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    impl Network {
        fn empty(n_sensor_nodes: usize, n_output_nodes: usize) -> Network {

            let n_total_nodes = n_sensor_nodes + n_output_nodes;
            let mut nodes = Vec::with_capacity(n_total_nodes);


            for _ in 0 .. n_sensor_nodes {
                let n = Node::create(NodeType::Sensor);
                nodes.push(n);
            }

            for _ in 0 .. n_output_nodes {
                let n = Node::create(NodeType::Output);
                nodes.push(n);
            }
            
            Network {
                has_bias_node: false,
                genome: Vec::new(),
                n_sensor_nodes,
                n_output_nodes,
                nodes,
                out_of_order: false
            }
        }
    }

    fn add_node_by_in_out(network: Network, in_id: usize, out_id: usize, new_weight: f64, global_innovation: &mut InnovationNumber, innovation_record: &mut FxHashMap<(usize, usize), InnovationNumber>) -> Network {
        let conn_index = match network.genome.iter().enumerate().find(|(_,x)| x.in_node_id == in_id && x.out_node_id == out_id && x.enabled) {
            Some((index, _)) => index,
            None => panic!("Cannot add a node to a connection that does not exist")
        };
        let new_conn_index = network.genome.len();
        let mut new_network = add_node(network, conn_index, global_innovation, innovation_record);
        new_network.genome[new_conn_index].weight = new_weight;
        new_network
    }

    fn diable_connection_by_in_out(mut network: Network, in_id: usize, out_id: usize) -> Network {
        let conn_index = match network.genome.iter().enumerate().find(|(_,x)| x.in_node_id == in_id && x.out_node_id == out_id && x.enabled) {
            Some((index, _)) => index,
            None => panic!("Cannot add a node to a connection that does not exist")
        };
        network.genome[conn_index].enabled = false;
        network
    }

    #[test]
    fn network_creation() {
        let mut rng = rand::thread_rng();
        let n_sensors = 3;
        let n_outputs = 2;
        let n_total = n_sensors + n_outputs;
        let network =  Network::init(&mut rng, n_sensors, n_outputs, false);
        assert_eq!(network.nodes.len(), n_total);
        assert_eq!(network.n_output_nodes, n_outputs);
        assert_eq!(network.n_sensor_nodes, n_sensors);

        for node in network.nodes[network.n_sensor_nodes ..].iter() {
            let l = node.input_connection_ids.len();
            assert_eq!(l, n_sensors)
        }
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that outputs to a sensor node")]
    fn cannot_add_connection_out_to_sensor() {
        let network = Network::empty(2, 2);
        let _ = add_connection(network, 1, 0, 0., &mut InnovationNumber(0), &mut FxHashMap::default());
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that inputs from an output node")]
    fn cannot_add_connection_in_from_output() {
        let network = Network::empty(2, 2);
        let _ = add_connection(network, 2, 3, 0., &mut InnovationNumber(0), &mut FxHashMap::default());
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that outputs beyond node count")]
    fn cannot_add_connection_beyond_length() {
        let network = Network::empty(2, 2);
        let _ = add_connection(network, 1, 5, 0., &mut InnovationNumber(0), &mut FxHashMap::default());
    }

    #[test]
    fn can_add_valid_node(){
        let mut rng = rand::thread_rng();
        let network = Network::init(&mut rng, 2, 2, false);
        let existing_conn_weight= network.genome[0].weight;
        let network = add_node(network, 0, &mut InnovationNumber(0), &mut FxHashMap::default());

        assert_eq!(network.genome[0].enabled, false);
        assert_eq!(network.genome[4].in_node_id, 0);
        assert_eq!(network.genome[4].out_node_id, 4);
        assert_eq!(network.genome[4].weight, 1.);
        assert_eq!(network.genome[5].in_node_id, 4);
        assert_eq!(network.genome[5].out_node_id, 2);
        assert_eq!(network.genome[5].weight, existing_conn_weight);
    }

    #[test]
    fn can_add_valid_connection(){
        let mut rng = rand::thread_rng();
        let network = Network::init(&mut rng, 2, 2, false);
        let mut global_innovation = InnovationNumber(4);
        let mut innovation_rec = FxHashMap::default();
        let network = add_node(network, 0, &mut global_innovation, &mut innovation_rec);
        let network = add_connection(network, 1, 4, 0.5, &mut global_innovation, &mut innovation_rec);
        assert_eq!(network.nodes[3].input_connection_ids.len(), 2);
        assert_eq!(network.genome[6].in_node_id, 1);
        assert_eq!(network.genome[6].out_node_id, 4);
    }

    #[test]
    fn add_node_at_end(){
        let mut rng = rand::thread_rng();
        let network = Network::init(&mut rng, 3, 2, true);
        // let existing_conn_weight= network.genome[3].weight;
        assert_eq!(network.genome.len(), 6);
        let network = add_node(network, 5, &mut InnovationNumber(0), &mut FxHashMap::default());
        assert_eq!(network.genome.len(), 8);
    }

    #[test]
    fn feed_forward() {
        let network = Network::empty(2, 2);
        let mut global_innov = InnovationNumber(4);
        let mut inno_rec = FxHashMap::default();
        //these connections will be disabled by adding nodes
        assert_eq!(network.genome.len(), 0);
        let network = add_connection(network, 0, 3, 0.6, &mut global_innov, &mut inno_rec);
        let network = add_connection(network, 1, 3, -0.9, &mut global_innov, &mut inno_rec);
        assert_eq!(network.genome.len(), 2);
        assert_eq!(network.nodes[3].input_connection_ids.len(), 2);
        let network = add_node_by_in_out(network, 0, 3, -0.1, &mut global_innov, &mut inno_rec);
        assert_eq!(network.nodes[4].input_connection_ids.len(), 1);
        let network = add_node_by_in_out(network, 1, 3, -0.8, &mut global_innov, &mut inno_rec);
        let network = add_connection(network, 0, 5, 0.6, &mut global_innov, &mut inno_rec);
        let mut network = add_connection(network, 5, 2, 0.4, &mut global_innov, &mut inno_rec);

        assert_eq!(network.genome.len(), 8);

        network.activate(&vec![0.5, -0.2]);
        assert_approx_eq!(network.nodes[2].value, 0.184);
        assert_approx_eq!(network.nodes[3].value, 0.);
        assert_eq!(inno_rec.len(), 8);
    }

    #[test]
    fn recurrent() {
        let network = Network::empty(2, 1);
        let mut global_innov = InnovationNumber(2);
        let mut inno_rec = FxHashMap::default();

        let network = add_connection(network, 1, 2, 0.9, &mut global_innov, &mut inno_rec);
        let network = add_node_by_in_out(network, 1, 2, 0.1, &mut global_innov, &mut inno_rec); //this creates node 3 between 1 and 2
        let network = add_node_by_in_out(network, 1, 3, -0.8, &mut global_innov, &mut inno_rec); //this creates node 4 between 1 and 3
        let network = add_connection(network, 0, 2, -0.4, &mut global_innov, &mut inno_rec);
        let network = add_node_by_in_out(network, 0, 2, 0.0, &mut global_innov, &mut inno_rec); //this create node 5 between 0 and 2
        let network = diable_connection_by_in_out(network, 0, 5);
        let network = add_connection(network, 0, 4, -0.8, &mut global_innov, &mut inno_rec);
        let network = add_connection(network, 3, 5, 0.5, &mut global_innov, &mut inno_rec);
        let mut network = add_connection(network, 5, 4, -0.1, &mut global_innov, &mut inno_rec);
        
        network.activate(&vec![-0.9, 0.6]);
        let first_output_ix = network.n_sensor_nodes;
        assert_approx_eq!(network.nodes[first_output_ix].value, 0.0216);
    }

    fn testing_organism_pair() -> (Organism, Organism) {
        //inspired by the crossover example from the original paper by K Stanley
        // let innovation_map:HashMap<(InputNodeId, OutputNodeId), InnovationNumber> = vec![((InputNodeId(0),OutputNodeId(3)),InnovationNumber(0)), ((InputNodeId(1),OutputNodeId(3)),InnovationNumber(1)), ((InputNodeId(2),OutputNodeId(3)),InnovationNumber(2))].into_iter().collect();
        let mut rng = rand::thread_rng();
        let ancestor = Network::init(&mut rng, 3, 1, false);
        let mut global_innov = InnovationNumber(3);
        let mut inno_rec = FxHashMap::default();
        let ancestor = add_node_by_in_out(ancestor, 1, 3, 0.5, &mut global_innov, &mut inno_rec);

        let parent2 = add_node_by_in_out(ancestor.clone(), 4, 3, 0.5, &mut global_innov, &mut inno_rec);
        let parent1 = add_connection(ancestor.clone(), 0, 4, 0.5, &mut global_innov, &mut inno_rec);
        let parent2 = add_connection(parent2, 2, 4, 0.5, &mut global_innov, &mut inno_rec);
        let parent2 = add_connection(parent2, 0, 5, 0.5, &mut global_innov, &mut inno_rec);

        let organism_1 = {
            Organism{
                network: parent1,
                fitness: 3
            }
        };

        let organism_2 = {
           Organism{
                network: parent2,
                fitness: 4
            }
        };
        (organism_1, organism_2)
    }
    #[test]
    fn cross_over_works() {
        let mut rng = rand::thread_rng();
        let (organism_1, organism_2) = testing_organism_pair();
        let child = cross_over(&mut rng, &organism_1, &organism_2);
        assert_eq!(child.genome.len(), 9)
    }

    #[test]
    fn genetic_distance_works() {
        let (organism_1, organism_2) = testing_organism_pair();
        assert_approx_eq!(genome_distance(&organism_1.network.genome, &organism_2.network.genome, 1., 0., 0.), 2. / 9.);
        assert_approx_eq!(genome_distance(&organism_1.network.genome, &organism_2.network.genome, 0., 1., 0.), 3. / 9.);
    }

    #[test]
    fn large_init(){
        let mut rng = rand::thread_rng();
        let network = Network::init(&mut rng, 9, 10, false);
        assert_eq!(network.genome[89].innovation.0, 89);
        assert_eq!(network.genome.len(), 90);
    }
}