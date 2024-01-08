use tailcall::tailcall;
use rand::{distributions::{Distribution, Uniform}, rngs::ThreadRng};
use std::collections::{HashMap, HashSet};

mod Vector {
    enum AllignedPair<T>{
        HasBoth(T, T),
        HasLeft(T),
        HasRight(T),
    }

    fn allign<T,I,R>(v1: Vec<T>, v2: Vec<T>, get_id: &dyn Fn(T) -> I, map: &dyn Fn(AllignedPair<T>) -> R) -> Vec<R> where I: std::cmp::PartialOrd{
        let n1 = v1.len();
        let n2 = v2.len();
        let n_res = std::cmp::max(n1,n2);
        let mut i1 = 0;
        let mut i2 = 0;
        let mut res = Vec::with_capacity(n_res);

        while (i1 < n1 || i2 < n2) {
            if i1 < n1 {
                let x1 = v1[i1];
                let id1 = get_id(x1);
                if i2 < n2 {
                    //still processing v1 and v2
                    let x2 = v2[i2];
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
                let x2 = v2[i2];
                let id2 = get_id(x2);
                let pair = AllignedPair::HasRight(x2);
                res.push(map(pair));
                i2 += 1;
            }
        }
        res
    }

    fn prefer_left<T,I>(v1: Vec<T>, v2: Vec<T>, get_id: &dyn Fn(T) -> I) -> Vec<T> where I: std::cmp::PartialOrd {
        let map = |x| {
            match x {
                AllignedPair::HasBoth(a, _) | AllignedPair::HasLeft(a) => a,
                AllignedPair::HasRight(b) => b
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
            active_sum: 0.,
            is_output: true,
        }
    }
}

enum Node<'a> {
    Relay(&'a RelayNode<'a>),
    Sensor(&'a SensorNode),
}

impl Node<'_> {
    fn id(self) -> NodeId {
        match self {
            Node::Relay(node) => node.id,
            Node::Sensor(node) => node.id,
        }
    }

    fn value(self) -> f64 {
        match self {
            Node::Relay(node) => node.value,
            Node::Sensor(node) => node.value,
        }
    }

    fn is_active(self) -> bool {
        match self {
            Node::Relay(node) => node.is_active,
            Node::Sensor(node) => true,
        }
    }
}
struct Connection<'a> {
    in_node: Node<'a>,
    out_node: &'a RelayNode<'a>,
    weight: f64,
    innovation: InnovationNumber,
    enabled: bool
}

// impl Connection {
//     fn copy_conn(&self) -> Connection {
//         Connection { in_node_id: self.in_node_id, out_node_id: self.out_node_id, weight: self.weight, innovation_num: self.innovation_num, enabled: self.enabled }
//     }
// }

struct Network<'a> {
    genome: Vec<Connection<'a>>,
    sensor_nodes: Vec<SensorNode>,
    output_nodes: Vec<RelayNode<'a>>,
    hidden_nodes: Vec<RelayNode<'a>>,
    node_lookup: Vec<Node<'a>>,
}

impl Network<'_> {
    fn set_phenotype<'a>(self) -> Network<'a> {
        for node in self.hidden_nodes.iter() {
            node.inputs.clear()
        }

        for node in self.output_nodes.iter() {
            node.inputs.clear()
        }

        for conn in self.genome.iter() {
            conn.out_node.inputs.push(conn)
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

    // fn init_from_starting_innovation_map(n_sensor_nodes: usize, n_output_nodes: usize, innovation_map: &HashMap<(InputNodeId, OutputNodeId), InnovationNumber>) -> Network {
    //     let mut res = Network::empty(n_sensor_nodes, n_output_nodes);
    //     let between = Uniform::from(-1.0..1.0);
    //     let mut rng = rand::thread_rng();
    //     for ((in_node_id, out_node_id), innovation_num) in innovation_map.iter() {
    //         let weight = between.sample(&mut rng);
    //         res.genome.push(Connection { in_node_id: *in_node_id, out_node_id: *out_node_id, weight, innovation_num: *innovation_num, enabled: true });
    //     }
    //     res.set_phenotype()
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
                    out_node: &output_nodes[out_index],
                    weight: between.sample(&mut rng),
                    innovation: InnovationNumber(innovation_number),
                    enabled: true
                };
                genome.push(conn);
                output_nodes[out_index].inputs.push(&conn)
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

    fn activate(mut self, sensor_values: Vec<f64>) -> Network {
        // set the sensor values
        if sensor_values.len() != self.n_sensor_nodes {
            panic!("sensor values not the right length for network")
        }
    
        for (i, value) in sensor_values.iter().enumerate() {
            self.node_values[i] = *value;
        }
    
        #[tailcall]
        fn activate_inner(network: Network, remaining_iterations: usize) -> Network {
            println!("remaining iters: {remaining_iterations}");
            if remaining_iterations == 0 {
                panic!("Too many iterations :(")
            } else {
                let (new_network, all_activated) = activation_pulse(network);
                if all_activated {
                    new_network
                } else {
                    activate_inner(new_network, remaining_iterations - 1)
                }
            }
        }
        activate_inner(self, 20)        
    }

    fn input_is_active(&self, input_node: InputNodeId) -> bool {
        self.active_nodes[input_node.0]
    }

    fn input_value(&self, input_node: InputNodeId) -> f64 {
        self.node_values[input_node.0]
    }
}

pub struct Organism {
    network: Network,
    fitness: usize // consider changing to generic type
}

impl Organism {
    pub fn init(n_sensor_nodes: usize, n_output_nodes: usize) -> Organism {
        Organism { network: Network::init(n_sensor_nodes, n_output_nodes), fitness: 0 }
    }

    pub fn activate(mut self, sensor_values: Vec<f64>) -> Organism {
        let network = self.network.activate(sensor_values);
        self.network = network;
        self
    }
}

fn align_genes<'a>(organism_1: &'a Organism, organism_2: &'a Organism) -> Vec<(Option<&'a Connection>, Option<&'a Connection>)> {
    let mut gene_index_1 = 0;
    let mut gene_index_2 = 0;
    let genome_size_1 = organism_1.network.genome.len();
    let genome_size_2 = organism_2.network.genome.len();
    let max_genome_size = std::cmp::max(genome_size_2, genome_size_1);
    let mut gene_pairs = Vec::with_capacity(max_genome_size);

    while gene_index_1 < genome_size_1 || gene_index_2 < genome_size_2 {
        if gene_index_1 < genome_size_1 {
            if gene_index_2 < genome_size_2 {
                //still processing org_1 and org_2
                if organism_1.network.genome[gene_index_1].innovation_num == organism_2.network.genome[gene_index_2].innovation_num {
                    let gene_pair = (Some(&organism_1.network.genome[gene_index_1]), Some(&organism_2.network.genome[gene_index_2]));
                    gene_pairs.push(gene_pair);
                    // gene_pairs[gene_pair_index] = 
                    // gene_pair_index+=1;
                    gene_index_1+=1;
                    gene_index_2+=1;
                } else if organism_1.network.genome[gene_index_1].innovation_num < organism_2.network.genome[gene_index_2].innovation_num {
                    let gene_pair = (Some(&organism_1.network.genome[gene_index_1]), None);
                    gene_pairs.push(gene_pair);
                    // gene_pairs[gene_pair_index] = (Some(&organism_1.network.genome[gene_index_1]), None);
                    // gene_pair_index+=1;
                    gene_index_1+=1;
                } else {
                    let gene_pair = (None, Some(&organism_2.network.genome[gene_index_2]));
                    gene_pairs.push(gene_pair);
                    // gene_pairs[gene_pair_index] = (None, Some(&organism_2.network.genome[gene_index_2]));
                    // gene_pair_index+=1;
                    gene_index_2+=1;
                }
            } else {
                //still processing org_1 but finished with org_2
                let gene_pair = (Some(&organism_1.network.genome[gene_index_1]), None);
                gene_pairs.push(gene_pair);
                // gene_pairs[gene_pair_index] = (Some(&organism_1.network.genome[gene_index_1]), None);
                // gene_pair_index+=1;
                gene_index_1+=1;
            }
        } else if gene_index_2 < genome_size_2 {
            //finished processing org_1 but still busy with org_2
            let gene_pair = (None, Some(&organism_2.network.genome[gene_index_2]));
            gene_pairs.push(gene_pair);
            // gene_pairs[gene_pair_index] = (None, Some(&organism_2.network.genome[gene_index_2]));
            // gene_pair_index+=1;
            gene_index_2+=1;
        } else {
            //if we are finished with org_1 and org_2, why are we still here?
            unreachable!("Cross over did not complete when expected")
        }
    }
    gene_pairs
}

fn cross_over(gene_pairs: &Vec<(Option<&Connection>, Option<&Connection>)>, fitness_1: usize, fitness2: usize, n_sensors: usize, n_output: usize) -> Organism {
    // let gene_pairs: Vec<(Option<&Connection>, Option<&Connection>)> = align_genes(organism_1, organism_2);
    let between = Uniform::from(0.0..1.0);
    let mut rng = rand::thread_rng();
    let new_genome : Vec<Connection> = gene_pairs.into_iter().map(|pair| {
        let r = between.sample(&mut rng);
        match pair {
            (Some(conn1), Some(_)) if r > 0.5 => Some(conn1.copy_conn()),
            (Some(_), Some(conn2)) => Some(conn2.copy_conn()),
            (Some(conn1), None) if fitness_1 > fitness2 => Some(conn1.copy_conn()),
            (Some(_), None) if fitness_1 < fitness2 => None,
            (Some(conn1), None) if r > 0.5 => Some(conn1.copy_conn()),
            (Some(_), None) => None,
            (None, Some(_)) if fitness_1 > fitness2 => None,
            (None, Some(conn2)) if fitness_1 < fitness2 => Some(conn2.copy_conn()),
            (None, Some(_)) if r > 0.5 => None,
            (None, Some(conn2)) => Some(conn2.copy_conn()),
            (None, None) => unreachable!("Cross over did not complete when expected")
        }
    }).flatten().collect();
    let network = Network::create_from_genome(n_sensors, n_output, new_genome);
    Organism { 
        network,
        fitness: 0
     }
}

fn genome_distance(gene_pairs: &Vec<(Option<&Connection>, Option<&Connection>)>, excess_coef: f64, disjoint_coef: f64, weight_diff_coef: f64) -> f64 {
    let mut total_weight_diff = 0.;
    let mut first_is_excess = false;
    let mut excess_count = 0;
    let mut disjoint_count = 0;
    let mut n1 = 0;
    let mut n2 = 0;

    for gene_pair in gene_pairs {
        match gene_pair {
            (Some(conn1), Some(conn2)) => {
                n1 += 1;
                n2 += 1;
                first_is_excess = false;
                total_weight_diff = total_weight_diff + (conn1.weight - conn2.weight).abs();
            },
            (Some(_), None) if first_is_excess => {
                n1 += 1;
                excess_count += 1;
            },
            (Some(_), None) => {
                n1 += 1;
                first_is_excess = true;
                disjoint_count = disjoint_count + excess_count;
                excess_count = 1;
            },
            (None, Some(_)) if first_is_excess => {
                n2 += 1;
                first_is_excess = false;
                disjoint_count = disjoint_count + excess_count;
                excess_count = 1;
            },
            (None, Some(_)) => {
                n2 += 1;
                excess_count += 1;
            },
            (None, None) => unreachable!("Unexpected gene pairing")
        }
    }

    let n = std::cmp::max(n1, n2) as f64;
    let excess_term = excess_coef * (excess_count as f64) / n;
    let disjoint_term = disjoint_coef * (disjoint_count as f64) / n;
    let weight_term = weight_diff_coef * total_weight_diff / n;
    excess_term + disjoint_term + weight_term
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn activation_pulse(mut network: Network) -> (Network, bool) {
    
    for (node_index_offset, input_connections) in network.input_connections_map.iter().enumerate() {
        // let node_index = node_index_offset + network.n_sensor_nodes;
        network.nodes_with_active_inputs[node_index_offset] = false;
        println!("node_index_offset is {node_index_offset}");
        network.active_sums[node_index_offset] = 0.;
        for conn_id in input_connections {
            println!("conn_id: {conn_id}");
            let conn = &network.genome[*conn_id];
            // if network.active_nodes[conn.in_node_id] && conn.enabled {
            if network.input_is_active(conn.in_node_id) && conn.enabled {
                network.nodes_with_active_inputs[node_index_offset] = true;
                let to_add = conn.weight * network.input_value(conn.in_node_id);
                network.active_sums[node_index_offset] += to_add;
                let f = network.active_sums[node_index_offset];
                println!("node_index_offset: {node_index_offset}; active_sum: {f}");
            }
        }
    }

    let mut all_active = true;
    for (node_index_offset, f) in &mut network.node_values[network.n_sensor_nodes ..].iter_mut().enumerate() {
        let node_index = node_index_offset + network.n_sensor_nodes;
        *f = relu(network.active_sums[node_index_offset]);
        println!("node_index_offset: {node_index_offset}; f: {f}");
        if network.nodes_with_active_inputs[node_index_offset] {
            network.active_nodes[node_index] = true;
        } else {
            all_active = false;
        }
    }

    (network, all_active)
}

fn add_connection(mut network: Network, in_id: InputNodeId, out_id: OutputNodeId, weight: f64, global_innovation: InnovationNumber) -> (Network, InnovationNumber) {
    if out_id.0 < network.n_sensor_nodes {
        panic!("Tried to add a connection that outputs to a sensor node")
    }

    if in_id.0 >= network.n_sensor_nodes && in_id.0 < (network.n_sensor_nodes + network.n_output_nodes) {
        panic!("Tried to add a connection that inputs from an output node")
    }

    if out_id.0 > network.n_total_nodes() {
        panic!("Tried to add a connection with an output ({}) that skips an available id ({})", out_id.0, network.n_total_nodes())
    }

    network.genome.push(Connection { in_node_id: in_id, out_node_id: out_id, weight, innovation_num: global_innovation, enabled: true });
    (network, global_innovation.inc())
}

fn add_node(mut network: Network, existing_conn_index: usize, global_innovation: InnovationNumber) -> (Network, InnovationNumber) {
    
    if !network.genome[existing_conn_index].enabled {
        panic!("Tried to add a node to a disabled connection")
    }

    let new_node_id = network.n_total_nodes();
    network.genome[existing_conn_index].enabled = false;
    network.n_hidden_nodes += 1;
    network.node_values.push(0.);
    network.active_sums.push(0.);
    network.active_nodes.push(false);
    network.nodes_with_active_inputs.push(false);
    
    let conn_in_id = network.genome[existing_conn_index].in_node_id;
    let conn_out_id = network.genome[existing_conn_index].out_node_id;
    let conn_weight = network.genome[existing_conn_index].weight;
    let (new_network, global_innovation) = add_connection(network, conn_in_id, OutputNodeId(new_node_id), 1., global_innovation);
    add_connection(new_network, InputNodeId(new_node_id), conn_out_id, conn_weight, global_innovation)

}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn network_creation() {
        let n_sensors = 3;
        let n_outputs = 2;
        let n_total = n_sensors + n_outputs;
        let network = 
            Network::empty(n_sensors, n_outputs).set_phenotype();
        assert_eq!(network.input_connections_map.len(), n_outputs);
        assert_eq!(network.active_nodes.len(), n_total);
        assert_eq!(network.nodes_with_active_inputs.len(), n_outputs);
        assert_eq!(network.active_sums.len(), n_outputs);
        assert_eq!(network.node_values.len(), n_total);

    }

    fn add_connection_by_int(mut network: Network, in_id: usize, out_id: usize, weight: f64, global_innovation: usize) -> (Network, usize) {
        let (net, inn) = add_connection(network, InputNodeId(in_id), OutputNodeId(out_id), weight, InnovationNumber(global_innovation));
        (net, inn.0)
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
    fn can_add_valid_connection(){
        let network = Network::empty(2, 2);
        let (network, _) = add_connection_by_int(network, 1, 2, 0.5, 1);
        let network = network.set_phenotype();
        assert_eq!(network.input_connections_map[0].len(), 1);
        let conn_index = network.input_connections_map[0][0];
        assert_eq!(network.genome[conn_index].in_node_id.0, 1);
        assert_eq!(network.genome[conn_index].innovation_num.0, 1);
    }

    #[test]
    fn can_add_valid_node(){
        let network = Network::empty(2, 2);
        let (network, global_innvation) = add_connection_by_int(network, 1, 2, 0.5, 1);
        let (network, global_innvation) = add_node(network, 0, InnovationNumber(global_innvation));

        assert_eq!(network.genome[0].enabled, false);
        assert_eq!(network.genome[1].in_node_id.0, 1);
        assert_eq!(network.genome[1].out_node_id.0, 4);
        assert_eq!(network.genome[1].weight, 1.);
        assert_eq!(network.genome[2].in_node_id.0, 4);
        assert_eq!(network.genome[2].out_node_id.0, 2);
        assert_eq!(network.genome[2].weight, 0.5);
    }

    fn add_node_by_in_out(network: Network, in_id: usize, out_id: usize, new_weight: f64, global_innovation: usize) -> (Network, usize) {
        let conn_index = match network.genome.iter().enumerate().find(|(i,x)| x.in_node_id.0 == in_id && x.out_node_id.0 == out_id && x.enabled) {
            Some((index, conn)) => index,
            None => panic!("Cannot add a node to a connection that does not exist")
        };
        let new_conn_index = network.genome.len();
        let (mut new_network, global_innov) = add_node(network, conn_index, InnovationNumber(global_innovation));
        new_network.genome[new_conn_index].weight = new_weight;
        (new_network, global_innov.0)
    }

    fn diable_connection_by_in_out(mut network: Network, in_id: usize, out_id: usize) -> Network {
        let conn_index = match network.genome.iter().enumerate().find(|(i,x)| x.in_node_id.0 == in_id && x.out_node_id.0 == out_id && x.enabled) {
            Some((index, conn)) => index,
            None => panic!("Cannot add a node to a connection that does not exist")
        };
        network.genome[conn_index].enabled = false;
        network
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
        let i = network.input_connections_map[0].len();
        let j = network.input_connections_map[0][0];
        println!("i: {i}; j: {j}");
        let new_network = network.activate(vec![0.5, -0.2]);
        assert_approx_eq!(new_network.node_values[2], 0.584);
        assert_approx_eq!(new_network.node_values[3], 0.);
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
        assert_approx_eq!(new_network.node_values[2], 0.0216);
    }

    fn testing_organism_pair() -> (Organism, Organism) {
        //inspired by the crossover example from the original paper by K Stanley
        // let innovation_map:HashMap<(InputNodeId, OutputNodeId), InnovationNumber> = vec![((InputNodeId(0),OutputNodeId(3)),InnovationNumber(0)), ((InputNodeId(1),OutputNodeId(3)),InnovationNumber(1)), ((InputNodeId(2),OutputNodeId(3)),InnovationNumber(2))].into_iter().collect();

        let organism_1 = {
            let network = Network::init(3, 1);
            let (network, _) = add_node_by_in_out(network, 1, 3, 0.5, 3);
            let (network, _) = add_connection_by_int(network, 0, 4, 0.5, 7);
            Organism{
                network,
                fitness: 3
            }
        };

        let organism_2 = {
            let network = Network::init(3, 1);
            let (network, _) = add_node_by_in_out(network, 1, 3, 0.5, 3);
            let (network, _) = add_node_by_in_out(network, 4, 3, 0.5, 5);
            let (network, _) = add_connection_by_int(network, 2, 4, 0.5, 8);
            let (network, _) = add_connection_by_int(network, 0, 5, 0.5, 9);

            Organism{
                network,
                fitness: 4
            }
        };
        (organism_1, organism_2)
    }
    #[test]
    fn cross_over_works() {
        let (organism_1, organism_2) = testing_organism_pair();
        let gene_pairs: Vec<(Option<&Connection>, Option<&Connection>)> = align_genes(&organism_1, &organism_2);
        let organism_child = cross_over(&gene_pairs, organism_1.fitness, organism_2.fitness, 3, 1);
        assert_eq!(organism_child.network.genome.len(), 9)
    }

    #[test]
    fn geneic_distance_works() {
        let (organism_1, organism_2) = testing_organism_pair();
        let gene_pairs: Vec<(Option<&Connection>, Option<&Connection>)> = align_genes(&organism_1, &organism_2);
        assert_approx_eq!(genome_distance(&gene_pairs, 1., 0., 0.), 2. / 9.);
        assert_approx_eq!(genome_distance(&gene_pairs, 0., 1., 0.), 3. / 9.);
    }

    #[test]
    fn large_init(){
        let network = Network::init(9, 10);
        assert_eq!(network.genome[89].innovation_num.0, 89);
        assert_eq!(network.genome.len(), 90);
    }
}