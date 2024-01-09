use tailcall::tailcall;
use rand::{distributions::{Distribution, Uniform}, rngs::ThreadRng};
use std::collections::{HashSet, hash_set};

mod Vector {
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

        while (i1 < n1 || i2 < n2) {
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

#[derive(Eq, PartialEq, Hash, PartialOrd, Copy, Clone)]
pub struct NodeId(usize);

#[derive(PartialEq, PartialOrd, Copy, Clone)]
pub struct InnovationNumber(usize);
impl InnovationNumber {
    fn inc(self) -> InnovationNumber {
        InnovationNumber(self.0 + 1)
    }
}

struct SensorNode{
    id: NodeId,
    value: f64,
}

struct RelayNode<'a>{
    id: NodeId,
    value: f64,
    is_active: bool,
    has_active_inputs: bool,
    inputs: Vec<&'a Connection<'a>>,
    input_ids: HashSet<NodeId>,
    active_sum: f64,
    is_output: bool,
}

impl RelayNode<'_> {
    fn create_output<'a>(i:usize) -> RelayNode<'a> {
        RelayNode{
            id: NodeId(i),
            value: 0.,
            is_active: false,
            has_active_inputs: false,
            inputs: Vec::new(),
            input_ids: HashSet::new(),
            active_sum: 0.,
            is_output: true,
        }
    }
}

enum Node<'a> { //TODO consider just one node type as a potential perf improvement
    Relay(&'a RelayNode<'a>),
    Sensor(&'a SensorNode),
}

impl Node<'_> {
    fn id(&self) -> NodeId {
        match self {
            Node::Relay(node) => node.id,
            Node::Sensor(node) => node.id,
        }
    }

    fn value(&self) -> f64 {
        match self {
            Node::Relay(node) => node.value,
            Node::Sensor(node) => node.value,
        }
    }

    fn is_active(&self) -> bool {
        match self {
            Node::Relay(node) => node.is_active,
            Node::Sensor(_) => true,
        }
    }
}
struct Connection<'a> {
    in_node: Node<'a>,
    out_node: &'a mut RelayNode<'a>,
    weight: f64,
    innovation: InnovationNumber,
    enabled: bool
}

struct Network<'a> {
    genome: Vec<Connection<'a>>,
    sensor_nodes: Vec<SensorNode>,
    output_nodes: Vec<RelayNode<'a>>,
    hidden_nodes: Vec<RelayNode<'a>>,
    node_lookup: Vec<Node<'a>>,
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

impl Network<'_> {
    fn set_phenotype<'a>(mut self) -> Network<'a> {
        for node in self.hidden_nodes.iter_mut() {
            node.inputs.clear();
            node.input_ids.clear();
        }

        for node in self.output_nodes.iter_mut() {
            node.inputs.clear();
            node.input_ids.clear();
        }

        for iConn in 0 .. self.genome.len() {
            let in_id = self.genome[iConn].in_node.id();
            self.genome[iConn].out_node.input_ids.insert(in_id);
        }
        for conn in self.genome.iter_mut() {
            conn.out_node.inputs.push(conn);
            conn.out_node.input_ids.insert(conn.in_node.id());
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

    fn init<'a>(mut rng: ThreadRng, n_sensor_nodes: usize, n_output_nodes: usize) -> Network<'a> {
        let between = Uniform::from(-1.0..1.0);
        let n_total_nodes = n_sensor_nodes + n_output_nodes;
        let mut node_lookup = Vec::with_capacity(n_total_nodes);

        let sensor_nodes : Vec<_> = (0 .. n_sensor_nodes).map(|i| {
            let n = SensorNode{id: NodeId(i), value: 0.};
            node_lookup.push(Node::Sensor(&n));
            n
        }).collect();
        let mut output_nodes : Vec<_> = (0 .. n_output_nodes).map(|i| {
            let n = RelayNode::create_output(i + n_sensor_nodes);
            node_lookup.push(Node::Relay(&n));
            n
        }).collect();

        let n_connections = n_sensor_nodes * n_output_nodes;
        let mut genome : Vec<Connection> = Vec::with_capacity(n_connections);

        for out_index in 0..n_output_nodes {
            for in_index in 0..n_sensor_nodes {
                let innovation_number = out_index * n_sensor_nodes + in_index;
                let conn = Connection{
                    in_node: Node::Sensor(&sensor_nodes[in_index]),
                    out_node: &mut output_nodes[out_index],
                    weight: between.sample(&mut rng),
                    innovation: InnovationNumber(innovation_number),
                    enabled: true
                };
                genome.push(conn);
                output_nodes[out_index].inputs.push(&conn);
                output_nodes[out_index].input_ids.insert(conn.in_node.id());
            }
        }

        Network {
            genome,
            sensor_nodes,
            output_nodes,
            hidden_nodes: Vec::new(),
            node_lookup
        }
    }

    fn activation_pulse<'a>(self) -> (Network<'a>, bool) {
        let accumulate_node_array_inputs = |mut ar: Vec<RelayNode>| {
            for i in 0 .. ar.len() {
                ar[i].has_active_inputs = false;
                ar[i].active_sum = 0.;
                for conn in ar[i].inputs {
                    if conn.in_node.is_active() && conn.enabled {
                        let to_add = conn.weight * conn.in_node.value();
                        ar[i].has_active_inputs = true;
                        ar[i].active_sum += to_add;
                    }
                }
            }
        };

        accumulate_node_array_inputs(self.hidden_nodes);
        accumulate_node_array_inputs(self.output_nodes);

        let mut all_active = true;

        let mut activate_nodes  = |mut ar: Vec<RelayNode>| {
            for i in 0 .. ar.len() {
                ar[i].value = relu(ar[i].active_sum);
                if ar[i].has_active_inputs {
                    ar[i].is_active = true;
                } else {
                    all_active = false;
                }
            }
        };

        activate_nodes(self.hidden_nodes);
        activate_nodes(self.output_nodes);

        (self, all_active)
    }

    fn activate<'a>(mut self, sensor_values: Vec<f64>) -> Network<'a> {
        debug_assert!(sensor_values.len() == self.sensor_nodes.len(), "sensor values not the right length for network");
        // set the sensor values
        for (i, value) in sensor_values.into_iter().enumerate() {
            self.sensor_nodes[i].value = value;
        }
        
        #[tailcall]
        fn activate_inner(network: Network, remaining_iterations: usize) -> Network {
            if remaining_iterations == 0 {
                panic!("Too many iterations :(")
            } else {
                let (new_network, all_activated) = network.activation_pulse();
                let x = network.output_nodes[0].value;
                println!("The value of the output node which I should not be allowed to access is {x}");
                let y = new_network.output_nodes[0].value;
                println!("The value of the output node which I should be allowed to access is {y}");
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


fn add_connection(mut network: Network, in_id: NodeId, out_id: NodeId, weight: f64, global_innovation: InnovationNumber) -> (Network, InnovationNumber) {
    debug_assert!(in_id.0 < network.sensor_nodes.len() || in_id.0 >= (network.sensor_nodes.len() + network.output_nodes.len()), "Tried to add a connection that inputs from an output node");
    
    let out_node =
        match network.node_lookup[out_id.0] {
            Node::Relay(node) => {
                if node.input_ids.contains(&in_id) {
                    panic!("Tried to connect 2 ndoes that are already connected")
                } else {
                    node
                }
            },
            Node::Sensor(_) => panic!("Tried to add a connection that outputs to a sensor node")
        };

    let new_conn = Connection{
        in_node: network.node_lookup[in_id.0],
        out_node: &mut out_node,
        weight: weight,
        innovation: global_innovation,
        enabled: true
    };

    network.genome.push(new_conn);
    (network, global_innovation.inc())
}

fn add_node(mut network: Network, existing_conn_index: usize, global_innovation: InnovationNumber) -> (Network, InnovationNumber) {
    let existing_conn = &network.genome[existing_conn_index];
    debug_assert!(existing_conn.enabled, "Tried to add a node to a disabled connection");

    let new_node_id = NodeId(network.node_lookup.len());
    
    let new_relay_node = RelayNode{
        id: new_node_id,
        has_active_inputs: false,
        inputs: vec![existing_conn], //ResizeArray() //TODO!!! - validate that this new node has correct inputs in a test
        input_ids: HashSet::from([existing_conn.in_node.id()]),
        is_active: false,
        is_output: false,
        active_sum: 0.,
        value: 0.
    };

    existing_conn.enabled = false;
    let conn_in_id = network.genome[existing_conn_index].in_node.id();
    let conn_out_id = network.genome[existing_conn_index].out_node.id;
    let conn_weight = network.genome[existing_conn_index].weight;

    network.hidden_nodes.push(new_relay_node);
    network.node_lookup.push (Node::Relay(&new_relay_node));
    
    let (new_network, global_innovation) = add_connection(network, conn_in_id, new_node_id, 1., global_innovation);
    add_connection(new_network, new_node_id, conn_out_id, conn_weight, global_innovation)

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

// fn align_genes<'a>(organism_1: &'a Organism, organism_2: &'a Organism) -> Vec<(Option<&'a Connection>, Option<&'a Connection>)> {
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

    impl Network<'_> {
        fn empty<'a>(n_sensor_nodes: usize, n_output_nodes: usize) -> Network<'a> {
            let n_total_nodes = n_sensor_nodes + n_output_nodes;
            let mut node_lookup = Vec::with_capacity(n_total_nodes);
            let sensor_nodes : Vec<_> = (0 .. n_sensor_nodes).map(|i| {
                let n = SensorNode{id: NodeId(i), value: 0.};
                node_lookup.push(Node::Sensor(&n));
                n
            }).collect();
            let mut output_nodes : Vec<_> = (0 .. n_output_nodes).map(|i| {
                let n = RelayNode::create_output(i + n_sensor_nodes);
                node_lookup.push(Node::Relay(&n));
                n
            }).collect();

            Network{
                genome: Vec::new(),
                sensor_nodes,
                output_nodes,
                hidden_nodes: Vec::new(),
                node_lookup
            }
        }
    }

    fn add_connection_by_int(mut network: Network, in_id: usize, out_id: usize, weight: f64, global_innovation: usize) -> (Network, usize) {
        let (net, inn) = add_connection(network, NodeId(in_id), NodeId(out_id), weight, InnovationNumber(global_innovation));
        (net, inn.0)
    }

    fn add_node_by_in_out(network: Network, in_id: usize, out_id: usize, new_weight: f64, global_innovation: usize) -> (Network, usize) {
        let conn_index = match network.genome.iter().enumerate().find(|(i,x)| x.in_node.id().0 == in_id && x.out_node.id.0 == out_id && x.enabled) {
            Some((index, conn)) => index,
            None => panic!("Cannot add a node to a connection that does not exist")
        };
        let new_conn_index = network.genome.len();
        let (mut new_network, global_innov) = add_node(network, conn_index, InnovationNumber(global_innovation));
        new_network.genome[new_conn_index].weight = new_weight;
        (new_network, global_innov.0)
    }

    fn diable_connection_by_in_out(mut network: Network, in_id: usize, out_id: usize) -> Network {
        let conn_index = match network.genome.iter().enumerate().find(|(i,x)| x.in_node.id().0 == in_id && x.out_node.id.0 == out_id && x.enabled) {
            Some((index, conn)) => index,
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
        let network =  Network::init(rng, n_sensors, n_outputs).set_phenotype();
        assert_eq!(network.node_lookup.len(), n_total);
        assert_eq!(network.output_nodes.len(), n_outputs);
        assert_eq!(network.sensor_nodes.len(), n_sensors);
        for node in network.output_nodes {
            assert_eq!(node.inputs.len(), n_sensors)
        }
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that outputs to a sensor node")]
    fn cannot_add_connection_out_to_sensor() {
        let network = Network::empty(2, 2);
        let _ = add_connection_by_int(network, 1, 0, 0., 0);
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that inputs from an output node")]
    fn cannot_add_connection_in_from_output() {
        let network = Network::empty(2, 2);
        let _ = add_connection_by_int(network, 2, 3, 0., 0);
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection with an output (5) that skips an available id (4)")]
    fn cannot_add_connection_beyond_length() {
        let network = Network::empty(2, 2);
        let _ = add_connection_by_int(network, 1, 5, 0., 0);
    }

    #[test]
    fn can_add_valid_node(){
        let mut rng = rand::thread_rng();
        let network = Network::init(rng, 2, 2);
        let existing_conn_weight= network.genome[0].weight;
        let (network, global_innvation) = add_node(network, 0, InnovationNumber(4));

        assert_eq!(network.genome[0].enabled, false);
        assert_eq!(network.genome[4].in_node.id().0, 1);
        assert_eq!(network.genome[4].out_node.id.0, 4);
        assert_eq!(network.genome[4].weight, 1.);
        assert_eq!(network.genome[5].in_node.id().0, 4);
        assert_eq!(network.genome[5].out_node.id.0, 2);
        assert_eq!(network.genome[2].weight, existing_conn_weight);
    }

    #[test]
    fn can_add_valid_connection(){
        let mut rng = rand::thread_rng();
        let network = Network::init(rng, 2, 2);
        let (network, global_innvation) = add_node(network, 0, InnovationNumber(4));
        let (network, _) = add_connection_by_int(network, 1, 4, 0.5, 1);
        // let network = network.set_phenotype();
        assert_eq!(network.output_nodes[1].inputs.len(), 2);
        assert_eq!(network.genome[6].in_node.id().0, 1);
        assert_eq!(network.genome[6].out_node.id.0, 4);
    }

    

    

    #[test]
    fn feed_forward() {
        let network = Network::empty(2, 2);
        let global_innov = 0;
        //these connections will be disabled by adding nodes
        let (network, global_innov) = add_connection_by_int(network, 0, 3, 0.6, global_innov);
        let (network, global_innov) = add_connection_by_int(network, 1, 3, -0.9, global_innov);

        let (network, global_innov) = add_node_by_in_out(network, 0, 3, -0.1, global_innov);
        let (network, global_innov) = add_node_by_in_out(network, 1, 3, -0.8, global_innov);
        let (network, global_innov) = add_connection_by_int(network, 0, 5, 0.6, global_innov);
        let (network, global_innov) = add_connection_by_int(network, 5, 2, 0.4, global_innov);

        let network = network.set_phenotype();
        let new_network = network.activate(vec![0.5, -0.2]);
        assert_approx_eq!(new_network.output_nodes[0].value, 0.584);
        assert_approx_eq!(new_network.output_nodes[1].value, 0.);
        assert_eq!(global_innov, 8);
    }

    #[test]
    fn recurrent() {
        let network = Network::empty(2, 1);
        let global_innov = 0;

        let (network, global_innov) = add_connection_by_int(network, 1, 2, 0.9, global_innov);
        let (network, global_innov) = add_node_by_in_out(network, 1, 2, 0.1, global_innov); //this creates node 3 between 1 and 2
        let (network, global_innov) = add_node_by_in_out(network, 1, 3, -0.8, global_innov); //this creates node 4 between 1 and 3
        let (network, global_innov) = add_connection_by_int(network, 0, 2, -0.4, global_innov);
        let (network, global_innov) = add_node_by_in_out(network, 0, 2, 0.0, global_innov); //this create node 5 between 0 and 2
        let network = diable_connection_by_in_out(network, 0, 5);
        let (network, global_innov) = add_connection_by_int(network, 0, 4, -0.8, global_innov);
        let (network, global_innov) = add_connection_by_int(network, 3, 5, 0.5, global_innov);
        let (network, global_innov) = add_connection_by_int(network, 5, 4, -0.1, global_innov);
        
        let network = network.set_phenotype();
        let new_network = network.activate(vec![-0.9, 0.6]);
        assert_approx_eq!(new_network.output_nodes[0].value, 0.0216);
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