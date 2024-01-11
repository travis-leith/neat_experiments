use tailcall::tailcall;
use rand::{distributions::{Distribution, Uniform}, rngs::ThreadRng};
use std::collections::HashSet;

mod vector {
    enum AllignedPair<T>{
        HasBoth(T, T),
        HasLeft(T),
        HasRight(T),
    }

    fn allign<T,I,R:Copy>(v1: Vec<T>, v2: Vec<T>, get_id: &dyn Fn(&T) -> I, map: &dyn Fn(AllignedPair<&T>) -> R) -> Vec<R> where I: std::cmp::PartialOrd{
        let n1 = v1.len();
        let n2 = v2.len();
        let n_res = std::cmp::max(n1,n2);
        let mut i1 = 0;
        let mut i2 = 0;
        let mut res = Vec::with_capacity(n_res);

        while i1 < n1 || i2 < n2 {
            if i1 < n1 {
                let x1 = &v1[i1];
                let id1 = get_id(x1);
                if i2 < n2 {
                    //still processing v1 and v2
                    let x2 = &v2[i2];
                    let id2 = get_id(x2);
                    if id1 == id2 {
                        let pair = AllignedPair::HasBoth(x1, x2);
                        res.push(map(pair));
                        i1 += 1;
                        i2 += 1;
                    } else if id1 < id2 {
                        let pair = AllignedPair::HasLeft(x1);
                        res.push(map(pair));
                        i1 += 1;
                    } else {
                        let pair = AllignedPair::HasRight(x2);
                        res.push(map(pair));
                        i2 += 1;
                    }
                } else {
                    //still processing v1 but finished with v2
                    let pair = AllignedPair::HasLeft(x1);
                    res.push(map(pair));
                    i1 += 1;
                }
            } else {
                //finished processing ar1 but still busy with ar2
                let x2 = &v2[i2];
                let pair = AllignedPair::HasRight(x2);
                res.push(map(pair));
                i2 += 1;
            }
        }
        res
    }

    fn prefer_left<T:Copy,I>(v1: Vec<T>, v2: Vec<T>, get_id: &dyn Fn(&T) -> I) -> Vec<T> where I: std::cmp::PartialOrd {
        let map = |x: AllignedPair<&T>| {
            match x {
                AllignedPair::HasBoth(a, _) | AllignedPair::HasLeft(a) => *a,
                AllignedPair::HasRight(b) => *b
            }
        };
        allign(v1, v2, get_id, &map)
    }
}

// #[derive(Eq, PartialEq, Hash, PartialOrd, Copy, Clone)]
// #[derive(Eq, PartialEq, Hash)]
// pub struct NodeId(usize);
// pub struct ConnectionId(usize);


// #[derive(PartialEq, PartialOrd, Copy, Clone)]
#[derive(Copy, Clone)]
pub struct InnovationNumber(usize);
impl InnovationNumber {
    fn inc(self) -> InnovationNumber {
        InnovationNumber(self.0 + 1)
    }
}

// struct SensorNode{
//     // id: NodeId,
//     value: f64,
// }

#[derive(PartialEq)]
enum NodeType{
    Sensor,
    Hidden,
    Output,
}

struct Node{
    // id: NodeId,
    value: f64,
    is_active: bool,
    has_active_inputs: bool,
    input_connection_ids: HashSet<usize>,
    input_node_ids: HashSet<usize>,
    active_sum: f64,
    // is_output: bool,
    node_type: NodeType,
}

impl Node {
    fn create(node_type: NodeType) -> Node {
        Node{
            // id: NodeId(i),
            value: 0.,
            is_active: node_type == NodeType::Sensor,
            has_active_inputs: false,
            input_connection_ids: HashSet::new(),
            input_node_ids: HashSet::new(),
            active_sum: 0.,
            // is_output: true,
            node_type: node_type
        }
    }
}

// enum Node { //TODO consider just one node type as a potential perf improvement
//     Relay(&'a Node),
//     Sensor(&'a SensorNode),
// }

// impl Node<'_> {
//     // fn id(&self) -> NodeId {
//     //     match self {
//     //         Node::Relay(node) => node.id,
//     //         Node::Sensor(node) => node.id,
//     //     }
//     // }

//     fn value(&self) -> f64 {
//         match self {
//             Node::Relay(node) => node.value,
//             Node::Sensor(node) => node.value,
//         }
//     }

//     fn is_active(&self) -> bool {
//         match self {
//             Node::Relay(node) => node.is_active,
//             Node::Sensor(_) => true,
//         }
//     }
// }
struct Connection {
    in_node_id: usize,
    out_node_id: usize,
    weight: f64,
    innovation: InnovationNumber,
    enabled: bool
}

struct Network {
    genome: Vec<Connection>,
    n_sensor_nodes: usize,
    n_output_nodes: usize,
    n_hidden_nodes: usize,
    nodes: Vec<Node>,
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

impl Network {
    fn set_phenotype(mut self) -> Network {
        for node in self.nodes.iter_mut() { //TODO skip sensor nodes
            node.input_connection_ids.clear();
            node.input_node_ids.clear();
        }

        for conn_ix in 0 .. self.genome.len() {
            let in_id = self.genome[conn_ix].in_node_id;
            let out_id = self.genome[conn_ix].out_node_id;
            let out_node = &mut self.nodes[out_id];
            out_node.input_connection_ids.insert(conn_ix);
            out_node.input_node_ids.insert(in_id);
        }
        self
    }

    // fn empty(n_sensor_nodes: usize, n_output_nodes: usize) -> Network {
    //     let mut active_nodes = vec![true; n_sensor_nodes];
    //     active_nodes.append(&mut vec![false; n_output_nodes]);
    //     Network {
    //         n_sensor_nodes,
    //         n_output_nodes,
    //         n_hidden_nodes: 0,
    //         genome: Vec::new(),
    //         input_connections_map: Vec::new(),
    //         active_nodes,
    //         nodes_with_active_inputs: vec![false; n_output_nodes], //sensors do not have inputs so no need to have values for them here 
    //         active_sums: vec![0.; n_output_nodes],
    //         node_values: vec![0.; n_sensor_nodes + n_output_nodes]
    //     }
    // }

    // fn create_from_genome(n_sensor_nodes: usize, n_output_nodes: usize, genome: Vec<Connection>) -> Network {
    //     let sensor_nodes = genome[0 .. n_sensor_nodes - 1].iter().map(|conn| conn.in_node)
    //     let sensor_nodes = (0..n_sensor_nodes-1).map(|i| )
        
    //     let hidden_index_start = n_sensor_nodes + n_output_nodes;
    //     let mut hidden_nodes = HashSet::new();
    //     for conn in genome.iter() {
    //         if conn.in_node_id.0 >= hidden_index_start {hidden_nodes.insert(conn.in_node_id.0);}
    //         if conn.out_node_id.0 >= hidden_index_start {hidden_nodes.insert(conn.out_node_id.0);}
    //     }
    //     let n_hidden_nodes = hidden_nodes.len();
    //     let n_non_sensor = n_output_nodes + n_hidden_nodes;
    //     let n_total_nodes = n_non_sensor + n_sensor_nodes;
    //     let mut active_nodes = vec![true; n_sensor_nodes];
    //     active_nodes.append(&mut vec![false; n_non_sensor]);
    //     Network {
    //         n_sensor_nodes,
    //         n_output_nodes,
    //         n_hidden_nodes,
    //         genome,
    //         input_connections_map: Vec::new(),
    //         active_nodes,
    //         nodes_with_active_inputs: vec![false; n_non_sensor],
    //         active_sums: vec![0.; n_non_sensor],
    //         node_values: vec![0.; n_total_nodes]
    //     }

    // }

    fn init(mut rng: ThreadRng, n_sensor_nodes: usize, n_output_nodes: usize) -> Network {
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
                    weight: between.sample(&mut rng),
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
            genome,
            n_sensor_nodes,
            n_output_nodes,
            n_hidden_nodes: 0,
            nodes
        }
    }

    fn activation_pulse(mut self) -> (Network, bool) {
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

        let mut all_active = true;

        for i_node in self.n_sensor_nodes .. self.nodes.len() {
            let mut node = &mut self.nodes[i_node];
            node.value = relu(node.active_sum);
            if node.has_active_inputs {
                node.is_active = true;
            } else {
                all_active = false;
            }
        }

        (self, all_active)
    }

    fn activate(mut self, sensor_values: Vec<f64>) -> Network {
        debug_assert!(sensor_values.len() == self.n_sensor_nodes, "sensor values not the right length for network");
        // set the sensor values
        for (i, value) in sensor_values.into_iter().enumerate() {
            self.nodes[i].value = value;
        }
        
        #[tailcall]
        fn activate_inner(network: Network, remaining_iterations: usize) -> Network {
            if remaining_iterations == 0 {
                panic!("Too many iterations :(")
            } else {
                let (new_network, all_activated) = network.activation_pulse();
                if all_activated {
                    new_network
                } else {
                    activate_inner(new_network, remaining_iterations - 1)
                }
            }
        }
        activate_inner(self, 20)        
    }
}


fn add_connection(mut network: Network, in_node_id: usize, out_node_id: usize, weight: f64, global_innovation: InnovationNumber) -> (Network, InnovationNumber) {
    debug_assert!(in_node_id != out_node_id, "Tried to add a connection where input is the same node as output");
    debug_assert!(in_node_id < network.n_sensor_nodes || in_node_id >= (network.n_sensor_nodes + network.n_output_nodes), "Tried to add a connection that inputs from an output node");
    debug_assert!(out_node_id >= network.n_sensor_nodes, "Tried to add a connection that outputs to a sensor node");
    debug_assert!(out_node_id < network.nodes.len(), "Tried to add a connection that outputs beyond node count");

    let out_node = &mut network.nodes[out_node_id];
    debug_assert!(!out_node.input_node_ids.contains(&in_node_id), "Tried to connect 2 ndoes that are already connected");

    let new_conn = Connection{
        in_node_id,
        out_node_id,
        weight: weight,
        innovation: global_innovation,
        enabled: true
    };
    let new_conn_ix = network.genome.len();
    
    out_node.input_connection_ids.insert(new_conn_ix);
    out_node.input_node_ids.insert(in_node_id);
    network.genome.push(new_conn);
    
    (network, global_innovation.inc())
}

fn add_node(mut network: Network, existing_conn_index: usize, global_innovation: InnovationNumber) -> (Network, InnovationNumber) {
    let in_node_id = network.genome[existing_conn_index].in_node_id;
    let out_node_id = network.genome[existing_conn_index].out_node_id;
    let weight = network.genome[existing_conn_index].weight;

    let genome = &mut network.genome[..];
    let nodes = &mut network.nodes[..];
    let mut existing_conn = &mut genome[existing_conn_index];
    debug_assert!(existing_conn.enabled, "Tried to add a node to a disabled connection");
    let new_node_id = nodes.len();
    let output_node = &mut nodes[existing_conn.out_node_id];
    
    let new_hidden_node = Node{
        has_active_inputs: false,
        input_connection_ids: HashSet::new(),
        input_node_ids: HashSet::new(),
        is_active: false,
        active_sum: 0.,
        value: 0.,
        node_type: NodeType::Hidden
    };

    existing_conn.enabled = false;
    output_node.input_connection_ids.remove(&existing_conn_index);
    output_node.input_node_ids.remove(&in_node_id);

    network.nodes.push(new_hidden_node);
    network.n_hidden_nodes += 1;

    
    
    let (new_network, global_innovation) = add_connection(network, in_node_id, new_node_id, 1., global_innovation);

    add_connection(new_network, new_node_id, out_node_id, weight, global_innovation)

}

// pub struct Organism {
//     network: Network,
//     fitness: usize // consider changing to generic type
// }

// impl Organism {
//     pub fn init(n_sensor_nodes: usize, n_output_nodes: usize) -> Organism {
//         Organism { network: Network::init(n_sensor_nodes, n_output_nodes), fitness: 0 }
//     }

//     pub fn activate(mut self, sensor_values: Vec<f64>) -> Organism {
//         let network = self.network.activate(sensor_values);
//         self.network = network;
//         self
//     }
// }

// fn align_genes(organism_1: &'a Organism, organism_2: &'a Organism) -> Vec<(Option<&'a Connection>, Option<&'a Connection>)> {
//     let mut gene_index_1 = 0;
//     let mut gene_index_2 = 0;
//     let genome_size_1 = organism_1.network.genome.len();
//     let genome_size_2 = organism_2.network.genome.len();
//     let max_genome_size = std::cmp::max(genome_size_2, genome_size_1);
//     let mut gene_pairs = Vec::with_capacity(max_genome_size);

//     while gene_index_1 < genome_size_1 || gene_index_2 < genome_size_2 {
//         if gene_index_1 < genome_size_1 {
//             if gene_index_2 < genome_size_2 {
//                 //still processing org_1 and org_2
//                 if organism_1.network.genome[gene_index_1].innovation_num == organism_2.network.genome[gene_index_2].innovation_num {
//                     let gene_pair = (Some(&organism_1.network.genome[gene_index_1]), Some(&organism_2.network.genome[gene_index_2]));
//                     gene_pairs.push(gene_pair);
//                     // gene_pairs[gene_pair_index] = 
//                     // gene_pair_index+=1;
//                     gene_index_1+=1;
//                     gene_index_2+=1;
//                 } else if organism_1.network.genome[gene_index_1].innovation_num < organism_2.network.genome[gene_index_2].innovation_num {
//                     let gene_pair = (Some(&organism_1.network.genome[gene_index_1]), None);
//                     gene_pairs.push(gene_pair);
//                     // gene_pairs[gene_pair_index] = (Some(&organism_1.network.genome[gene_index_1]), None);
//                     // gene_pair_index+=1;
//                     gene_index_1+=1;
//                 } else {
//                     let gene_pair = (None, Some(&organism_2.network.genome[gene_index_2]));
//                     gene_pairs.push(gene_pair);
//                     // gene_pairs[gene_pair_index] = (None, Some(&organism_2.network.genome[gene_index_2]));
//                     // gene_pair_index+=1;
//                     gene_index_2+=1;
//                 }
//             } else {
//                 //still processing org_1 but finished with org_2
//                 let gene_pair = (Some(&organism_1.network.genome[gene_index_1]), None);
//                 gene_pairs.push(gene_pair);
//                 // gene_pairs[gene_pair_index] = (Some(&organism_1.network.genome[gene_index_1]), None);
//                 // gene_pair_index+=1;
//                 gene_index_1+=1;
//             }
//         } else if gene_index_2 < genome_size_2 {
//             //finished processing org_1 but still busy with org_2
//             let gene_pair = (None, Some(&organism_2.network.genome[gene_index_2]));
//             gene_pairs.push(gene_pair);
//             // gene_pairs[gene_pair_index] = (None, Some(&organism_2.network.genome[gene_index_2]));
//             // gene_pair_index+=1;
//             gene_index_2+=1;
//         } else {
//             //if we are finished with org_1 and org_2, why are we still here?
//             unreachable!("Cross over did not complete when expected")
//         }
//     }
//     gene_pairs
// }

// fn cross_over(gene_pairs: &Vec<(Option<&Connection>, Option<&Connection>)>, fitness_1: usize, fitness2: usize, n_sensors: usize, n_output: usize) -> Organism {
//     // let gene_pairs: Vec<(Option<&Connection>, Option<&Connection>)> = align_genes(organism_1, organism_2);
//     let between = Uniform::from(0.0..1.0);
//     let mut rng = rand::thread_rng();
//     let new_genome : Vec<Connection> = gene_pairs.into_iter().map(|pair| {
//         let r = between.sample(&mut rng);
//         match pair {
//             (Some(conn1), Some(_)) if r > 0.5 => Some(conn1.copy_conn()),
//             (Some(_), Some(conn2)) => Some(conn2.copy_conn()),
//             (Some(conn1), None) if fitness_1 > fitness2 => Some(conn1.copy_conn()),
//             (Some(_), None) if fitness_1 < fitness2 => None,
//             (Some(conn1), None) if r > 0.5 => Some(conn1.copy_conn()),
//             (Some(_), None) => None,
//             (None, Some(_)) if fitness_1 > fitness2 => None,
//             (None, Some(conn2)) if fitness_1 < fitness2 => Some(conn2.copy_conn()),
//             (None, Some(_)) if r > 0.5 => None,
//             (None, Some(conn2)) => Some(conn2.copy_conn()),
//             (None, None) => unreachable!("Cross over did not complete when expected")
//         }
//     }).flatten().collect();
//     let network = Network::create_from_genome(n_sensors, n_output, new_genome);
//     Organism { 
//         network,
//         fitness: 0
//      }
// }

// fn genome_distance(gene_pairs: &Vec<(Option<&Connection>, Option<&Connection>)>, excess_coef: f64, disjoint_coef: f64, weight_diff_coef: f64) -> f64 {
//     let mut total_weight_diff = 0.;
//     let mut first_is_excess = false;
//     let mut excess_count = 0;
//     let mut disjoint_count = 0;
//     let mut n1 = 0;
//     let mut n2 = 0;

//     for gene_pair in gene_pairs {
//         match gene_pair {
//             (Some(conn1), Some(conn2)) => {
//                 n1 += 1;
//                 n2 += 1;
//                 first_is_excess = false;
//                 total_weight_diff = total_weight_diff + (conn1.weight - conn2.weight).abs();
//             },
//             (Some(_), None) if first_is_excess => {
//                 n1 += 1;
//                 excess_count += 1;
//             },
//             (Some(_), None) => {
//                 n1 += 1;
//                 first_is_excess = true;
//                 disjoint_count = disjoint_count + excess_count;
//                 excess_count = 1;
//             },
//             (None, Some(_)) if first_is_excess => {
//                 n2 += 1;
//                 first_is_excess = false;
//                 disjoint_count = disjoint_count + excess_count;
//                 excess_count = 1;
//             },
//             (None, Some(_)) => {
//                 n2 += 1;
//                 excess_count += 1;
//             },
//             (None, None) => unreachable!("Unexpected gene pairing")
//         }
//     }

//     let n = std::cmp::max(n1, n2) as f64;
//     let excess_term = excess_coef * (excess_count as f64) / n;
//     let disjoint_term = disjoint_coef * (disjoint_count as f64) / n;
//     let weight_term = weight_diff_coef * total_weight_diff / n;
//     excess_term + disjoint_term + weight_term
// }

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use rand::RngCore;

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
                genome: Vec::new(),
                n_sensor_nodes,
                n_output_nodes,
                n_hidden_nodes: 0,
                nodes
            }
        }
    }

    // fn add_connection_by_int(mut network: Network, in_id: usize, out_id: usize, weight: f64, global_innovation: usize) -> (Network, usize) {
    //     let (net, inn) = add_connection(network, NodeId(in_id), NodeId(out_id), weight, InnovationNumber(global_innovation));
    //     (net, inn.0)
    // }

    fn add_node_by_in_out(network: Network, in_id: usize, out_id: usize, new_weight: f64, global_innovation: InnovationNumber) -> (Network, InnovationNumber) {
        let conn_index = match network.genome.iter().enumerate().find(|(i,x)| x.in_node_id == in_id && x.out_node_id == out_id && x.enabled) {
            Some((index, conn)) => index,
            None => panic!("Cannot add a node to a connection that does not exist")
        };
        let new_conn_index = network.genome.len();
        let (mut new_network, global_innov) = add_node(network, conn_index, global_innovation);
        new_network.genome[new_conn_index].weight = new_weight;
        (new_network, global_innov)
    }

    fn diable_connection_by_in_out(mut network: Network, in_id: usize, out_id: usize) -> Network {
        let conn_index = match network.genome.iter().enumerate().find(|(i,x)| x.in_node_id == in_id && x.out_node_id == out_id && x.enabled) {
            Some((index, conn)) => index,
            None => panic!("Cannot add a node to a connection that does not exist")
        };
        network.genome[conn_index].enabled = false;
        network
    }

    #[test]
    fn network_creation() {
        let rng = rand::thread_rng();
        let n_sensors = 3;
        let n_outputs = 2;
        let n_total = n_sensors + n_outputs;
        let network =  Network::init(rng, n_sensors, n_outputs);
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
        let _ = add_connection(network, 1, 0, 0., InnovationNumber(4));
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that inputs from an output node")]
    fn cannot_add_connection_in_from_output() {
        let network = Network::empty(2, 2);
        let _ = add_connection(network, 2, 3, 0., InnovationNumber(4));
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that outputs beyond node count")]
    fn cannot_add_connection_beyond_length() {
        let network = Network::empty(2, 2);
        let _ = add_connection(network, 1, 5, 0., InnovationNumber(4));
    }

    #[test]
    fn can_add_valid_node(){
        let mut rng = rand::thread_rng();
        let network = Network::init(rng, 2, 2);
        let existing_conn_weight= network.genome[0].weight;
        let (network, global_innvation) = add_node(network, 0, InnovationNumber(4));

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
        let rng = rand::thread_rng();
        let network = Network::init(rng, 2, 2);
        let (network, global_innvation) = add_node(network, 0, InnovationNumber(4));
        let (network, _) = add_connection(network, 1, 4, 0.5, global_innvation);
        // let network = network.set_phenotype();
        assert_eq!(network.nodes[3].input_connection_ids.len(), 2);
        assert_eq!(network.genome[6].in_node_id, 1);
        assert_eq!(network.genome[6].out_node_id, 4);
    }

    #[test]
    fn feed_forward() {
        let network = Network::empty(2, 2);
        let global_innov = InnovationNumber(0);
        //these connections will be disabled by adding nodes
        assert_eq!(network.genome.len(), 0);
        let (network, global_innov) = add_connection(network, 0, 3, 0.6, global_innov);
        let (network, global_innov) = add_connection(network, 1, 3, -0.9, global_innov);
        assert_eq!(network.genome.len(), 2);
        assert_eq!(network.nodes[3].input_connection_ids.len(), 2);
        let (network, global_innov) = add_node_by_in_out(network, 0, 3, -0.1, global_innov);
        assert_eq!(network.nodes[4].input_connection_ids.len(), 1);
        let (network, global_innov) = add_node_by_in_out(network, 1, 3, -0.8, global_innov);
        let (network, global_innov) = add_connection(network, 0, 5, 0.6, global_innov);
        let (network, global_innov) = add_connection(network, 5, 2, 0.4, global_innov);


        assert_eq!(network.genome.len(), 8);


        // let network = network.set_phenotype();
        let new_network = network.activate(vec![0.5, -0.2]);
        assert_approx_eq!(new_network.nodes[2].value, 0.184);
        assert_approx_eq!(new_network.nodes[3].value, 0.);
        assert_eq!(global_innov.0, 8);
    }

    #[test]
    fn recurrent() {
        let network = Network::empty(2, 1);
        let global_innov = InnovationNumber(0);

        let (network, global_innov) = add_connection(network, 1, 2, 0.9, global_innov);
        let (network, global_innov) = add_node_by_in_out(network, 1, 2, 0.1, global_innov); //this creates node 3 between 1 and 2
        let (network, global_innov) = add_node_by_in_out(network, 1, 3, -0.8, global_innov); //this creates node 4 between 1 and 3
        let (network, global_innov) = add_connection(network, 0, 2, -0.4, global_innov);
        let (network, global_innov) = add_node_by_in_out(network, 0, 2, 0.0, global_innov); //this create node 5 between 0 and 2
        let network = diable_connection_by_in_out(network, 0, 5);
        let (network, global_innov) = add_connection(network, 0, 4, -0.8, global_innov);
        let (network, global_innov) = add_connection(network, 3, 5, 0.5, global_innov);
        let (network, global_innov) = add_connection(network, 5, 4, -0.1, global_innov);
        
        // let network = network.set_phenotype();
        let new_network = network.activate(vec![-0.9, 0.6]);
        let first_output_ix = new_network.n_sensor_nodes;
        assert_approx_eq!(new_network.nodes[first_output_ix].value, 0.0216);
    }

    // fn testing_organism_pair() -> (Organism, Organism) {
    //     //inspired by the crossover example from the original paper by K Stanley
    //     // let innovation_map:HashMap<(InputNodeId, OutputNodeId), InnovationNumber> = vec![((InputNodeId(0),OutputNodeId(3)),InnovationNumber(0)), ((InputNodeId(1),OutputNodeId(3)),InnovationNumber(1)), ((InputNodeId(2),OutputNodeId(3)),InnovationNumber(2))].into_iter().collect();

    //     let organism_1 = {
    //         let network = Network::init(3, 1);
    //         let (network, _) = add_node_by_in_out(network, 1, 3, 0.5, 3);
    //         let (network, _) = add_connection_by_int(network, 0, 4, 0.5, 7);
    //         Organism{
    //             network,
    //             fitness: 3
    //         }
    //     };

    //     let organism_2 = {
    //         let network = Network::init(3, 1);
    //         let (network, _) = add_node_by_in_out(network, 1, 3, 0.5, 3);
    //         let (network, _) = add_node_by_in_out(network, 4, 3, 0.5, 5);
    //         let (network, _) = add_connection_by_int(network, 2, 4, 0.5, 8);
    //         let (network, _) = add_connection_by_int(network, 0, 5, 0.5, 9);

    //         Organism{
    //             network,
    //             fitness: 4
    //         }
    //     };
    //     (organism_1, organism_2)
    // }
    // #[test]
    // fn cross_over_works() {
    //     let (organism_1, organism_2) = testing_organism_pair();
    //     let gene_pairs: Vec<(Option<&Connection>, Option<&Connection>)> = align_genes(&organism_1, &organism_2);
    //     let organism_child = cross_over(&gene_pairs, organism_1.fitness, organism_2.fitness, 3, 1);
    //     assert_eq!(organism_child.network.genome.len(), 9)
    // }

    // #[test]
    // fn geneic_distance_works() {
    //     let (organism_1, organism_2) = testing_organism_pair();
    //     let gene_pairs: Vec<(Option<&Connection>, Option<&Connection>)> = align_genes(&organism_1, &organism_2);
    //     assert_approx_eq!(genome_distance(&gene_pairs, 1., 0., 0.), 2. / 9.);
    //     assert_approx_eq!(genome_distance(&gene_pairs, 0., 1., 0.), 3. / 9.);
    // }

    // #[test]
    // fn large_init(){
    //     let network = Network::init(9, 10);
    //     assert_eq!(network.genome[89].innovation_num.0, 89);
    //     assert_eq!(network.genome.len(), 90);
    // }
}