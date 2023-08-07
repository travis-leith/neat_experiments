use tailcall::tailcall;
use rand::distributions::{Distribution, Uniform};
use std::collections::{HashMap, HashSet};

pub struct Connection {
    in_node_id: usize,
    out_node_id: usize,
    weight: f64,
    innovation_num: usize,
    enabled: bool
}

impl Connection {
    fn copy_conn(&self) -> Connection {
        Connection { in_node_id: self.in_node_id, out_node_id: self.out_node_id, weight: self.weight, innovation_num: self.innovation_num, enabled: self.enabled }
    }
}

pub struct Network {
    n_sensor_nodes: usize,
    n_output_nodes: usize,
    n_hidden_nodes: usize,
    genome: Vec<Connection>,
    input_connections_map: Vec<Vec<usize>>,
    nodes_with_active_inputs: Vec<bool>,
    active_nodes: Vec<bool>,
    active_sums: Vec<f64>,
    node_values: Vec<f64>
}

impl Network {
    fn n_total_nodes(&self) -> usize {
        self.n_hidden_nodes + self.n_output_nodes + self.n_sensor_nodes
    }

    fn set_phenotype(mut self) -> Network {
        let n_activateable = self.n_output_nodes + self.n_hidden_nodes;
        let node_offset = self.n_sensor_nodes;
        self.input_connections_map = (0..n_activateable).map(|_| Vec::new()).collect();
        for (conn_index, conn) in self.genome.iter().enumerate() {
            self.input_connections_map[conn.out_node_id - node_offset].push(conn_index)
        }
        self
    }

    fn empty(n_sensor_nodes: usize, n_output_nodes: usize) -> Network {
        let mut active_nodes = vec![true; n_sensor_nodes];
        active_nodes.append(&mut vec![false; n_output_nodes]);
        Network {
            n_sensor_nodes,
            n_output_nodes,
            n_hidden_nodes: 0,
            genome: Vec::new(),
            input_connections_map: Vec::new(),
            active_nodes,
            nodes_with_active_inputs: vec![false; n_output_nodes], //sensors do not have inputs so no need to have values for them here 
            active_sums: vec![0.; n_output_nodes],
            node_values: vec![0.; n_sensor_nodes + n_output_nodes]
        }
    }

    fn create_from_genome(n_sensor_nodes: usize, n_output_nodes: usize, genome: Vec<Connection>) -> Network {
        let hidden_index_start = n_sensor_nodes + n_output_nodes;
        let mut hidden_nodes = HashSet::new();
        for conn in genome.iter() {
            if conn.in_node_id >= hidden_index_start {hidden_nodes.insert(conn.in_node_id);}
            if conn.out_node_id >= hidden_index_start {hidden_nodes.insert(conn.out_node_id);}
        }
        let n_hidden_nodes = hidden_nodes.len();
        let n_non_sensor = n_output_nodes + n_hidden_nodes;
        let n_total_nodes = n_non_sensor + n_sensor_nodes;
        let mut active_nodes = vec![true; n_sensor_nodes];
        active_nodes.append(&mut vec![false; n_non_sensor]);
        Network {
            n_sensor_nodes,
            n_output_nodes,
            n_hidden_nodes,
            genome,
            input_connections_map: Vec::new(),
            active_nodes,
            nodes_with_active_inputs: vec![false; n_non_sensor],
            active_sums: vec![0.; n_non_sensor],
            node_values: vec![0.; n_total_nodes]
        }

    }

    fn init_from_starting_innovation_map(n_sensor_nodes: usize, n_output_nodes: usize, innovation_map: &HashMap<(usize, usize), usize>) -> Network {
        let mut res = Network::empty(n_sensor_nodes, n_output_nodes);
        let between = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();
        for ((in_node_id, out_node_id), innovation_num) in innovation_map.iter() {
            let weight = between.sample(&mut rng);
            res.genome.push(Connection { in_node_id: *in_node_id, out_node_id: *out_node_id, weight, innovation_num: *innovation_num, enabled: true });
        }
        res.set_phenotype()
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

    
}

pub struct Organism {
    network: Network,
    fitness: usize // consider changing to generic type
}

fn cross_over(organism_1: &Organism, organism_2: &Organism) -> Organism {
    let mut gene_index_1 = 0;
    let mut gene_index_2 = 0;
    let genome_size_1 = organism_1.network.genome.len();
    let genome_size_2 = organism_2.network.genome.len();
    let max_genome_size = std::cmp::max(genome_size_2, genome_size_1);
    // let mut gene_pair_index = 0;
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
    let between = Uniform::from(0.0..1.0);
    let mut rng = rand::thread_rng();
    let new_genome : Vec<Connection> = gene_pairs.into_iter().map(|pair| {
        let r = between.sample(&mut rng);
        match pair {
            (Some(conn1), Some(_)) if r > 0.5 => Some(conn1.copy_conn()),
            (Some(_), Some(conn2)) => Some(conn2.copy_conn()),
            (Some(conn1), None) if organism_1.fitness > organism_2.fitness => Some(conn1.copy_conn()),
            (Some(_), None) if organism_1.fitness < organism_2.fitness => None,
            (Some(conn1), None) if r > 0.5 => Some(conn1.copy_conn()),
            (Some(_), None) => None,
            (None, Some(_)) if organism_1.fitness > organism_2.fitness => None,
            (None, Some(conn2)) if organism_1.fitness < organism_2.fitness => Some(conn2.copy_conn()),
            (None, Some(_)) if r > 0.5 => None,
            (None, Some(conn2)) => Some(conn2.copy_conn()),
            (None, None) => unreachable!("Cross over did not complete when expected")
        }
    }).flatten().collect();
    let network = Network::create_from_genome(organism_1.network.n_sensor_nodes, organism_2.network.n_output_nodes, new_genome);
    Organism { 
        network,
        fitness: 0
     }
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
        let node_index = node_index_offset + network.n_sensor_nodes;
        network.nodes_with_active_inputs[node_index_offset] = false;

        network.active_sums[node_index_offset] = 0.;
        for conn_id in input_connections {
            let conn = &network.genome[*conn_id];
            if network.active_nodes[conn.in_node_id] && conn.enabled {
                network.nodes_with_active_inputs[node_index_offset] = true;
                let to_add = conn.weight * network.node_values[conn.in_node_id];
                network.active_sums[node_index_offset] += to_add;
            }
        }
    }

    let mut all_active = true;
    for (node_index_offset, f) in &mut network.node_values[network.n_sensor_nodes ..].iter_mut().enumerate() {
        let node_index = node_index_offset + network.n_sensor_nodes;
        *f = relu(network.active_sums[node_index_offset]);
        if network.nodes_with_active_inputs[node_index_offset] {
            network.active_nodes[node_index] = true;
        } else {
            all_active = false;
        }
    }

    (network, all_active)
}

fn add_connection(mut network: Network, in_id: usize, out_id: usize, weight: f64, global_innovation: usize) -> (Network, usize) {
    if out_id < network.n_sensor_nodes {
        panic!("Tried to add a connection that outputs to a sensor node")
    }

    if in_id >= network.n_sensor_nodes && in_id < (network.n_sensor_nodes + network.n_output_nodes) {
        panic!("Tried to add a connection that inputs from an output node")
    }

    if out_id > network.n_total_nodes() {
        panic!("Tried to add a connection with an output ({out_id}) that skips an available id ({})", network.n_total_nodes())
    }

    network.genome.push(Connection { in_node_id: in_id, out_node_id: out_id, weight, innovation_num: global_innovation, enabled: true });
    (network, global_innovation + 1)
}

fn add_node(mut network: Network, existing_conn_index: usize, global_innovation: usize) -> (Network, usize) {
    
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
    let (new_network, global_innovation) = add_connection(network, conn_in_id, new_node_id, 1., global_innovation);
    add_connection(new_network, new_node_id, conn_out_id, conn_weight, global_innovation)

}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use itertools::Itertools;

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

    #[test]
    #[should_panic(expected = "Tried to add a connection that outputs to a sensor node")]
    fn cannot_add_connection_out_to_sensor() {
        let network = Network::empty(2, 2);
        let _ = add_connection(network, 1, 0, 0., 0);
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that inputs from an output node")]
    fn cannot_add_connection_in_from_output() {
        let network = Network::empty(2, 2);
        let _ = add_connection(network, 2, 3, 0., 0);
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection with an output (5) that skips an available id (4)")]
    fn cannot_add_connection_beyond_length() {
        let network = Network::empty(2, 2);
        let _ = add_connection(network, 1, 5, 0., 0);
    }

    #[test]
    fn can_add_valid_connection(){
        let network = Network::empty(2, 2);
        let (network, _) = add_connection(network, 1, 2, 0.5, 1);
        let network = network.set_phenotype();
        assert_eq!(network.input_connections_map[0].len(), 1);
        let conn_index = network.input_connections_map[0][0];
        assert_eq!(network.genome[conn_index].in_node_id, 1);
        assert_eq!(network.genome[conn_index].innovation_num, 1);
    }

    #[test]
    fn can_add_valid_node(){
        let network = Network::empty(2, 2);
        let (network, global_innvation) = add_connection(network, 1, 2, 0.5, 1);
        let (network, global_innvation) = add_node(network, 0, global_innvation);

        assert_eq!(network.genome[0].enabled, false);
        assert_eq!(network.genome[1].in_node_id, 1);
        assert_eq!(network.genome[1].out_node_id, 4);
        assert_eq!(network.genome[1].weight, 1.);
        assert_eq!(network.genome[2].in_node_id, 4);
        assert_eq!(network.genome[2].out_node_id, 2);
        assert_eq!(network.genome[2].weight, 0.5);
    }

    fn add_node_by_in_out(network: Network, in_id: usize, out_id: usize, new_weight: f64, global_innovation: usize) -> (Network, usize) {
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
    fn feed_formward() {
        let network = Network::empty(2, 2);
        let global_innov = 0;
        //these connections will be disabled by adding nodes
        let (network, global_innov) = add_connection(network, 0, 3, 0.6, global_innov);
        let (network, global_innov) = add_connection(network, 1, 3, -0.9, global_innov);

        let (network, global_innov) = add_node_by_in_out(network, 0, 3, -0.1, global_innov);
        let (network, global_innov) = add_node_by_in_out(network, 1, 3, -0.8, global_innov);
        let (network, global_innov) = add_connection(network, 0, 5, 0.6, global_innov);
        let (network, global_innov) = add_connection(network, 5, 2, 0.4, global_innov);

        let network = network.set_phenotype();
        let new_network = network.activate(vec![0.5, -0.2]);
        assert_approx_eq!(new_network.node_values[2], 0.184);
        assert_approx_eq!(new_network.node_values[3], 0.);
        assert_eq!(global_innov, 8);
    }

    #[test]
    fn recurrent() {
        let network = Network::empty(2, 1);
        let global_innov = 0;

        let (network, global_innov) = add_connection(network, 1, 2, 0.9, global_innov);
        let (network, global_innov) = add_node_by_in_out(network, 1, 2, 0.1, global_innov); //this creates node 3 between 1 and 2
        let (network, global_innov) = add_node_by_in_out(network, 1, 3, -0.8, global_innov); //this creates node 4 between 1 and 3
        let (network, global_innov) = add_connection(network, 0, 2, -0.4, global_innov);
        let (network, global_innov) = add_node_by_in_out(network, 0, 2, 0.0, global_innov); //this create node 5 between 0 and 2
        let network = diable_connection_by_in_out(network, 0, 5);
        let (network, global_innov) = add_connection(network, 0, 4, -0.8, global_innov);
        let (network, global_innov) = add_connection(network, 3, 5, 0.5, global_innov);
        let (network, global_innov) = add_connection(network, 5, 4, -0.1, global_innov);
        
        let network = network.set_phenotype();
        let new_network = network.activate(vec![-0.9, 0.6]);
        assert_approx_eq!(new_network.node_values[2], 0.0216);
    }

    #[test]
    fn cross_over_works() {
        //inspired by the crossover example from the original paper by K Stanley
        let innovation_map:HashMap<(usize, usize), usize> = vec![((0,3),0), ((1,3),1), ((2,3),2)].into_iter().collect();

        let organism_1 = {
            let network = Network::init_from_starting_innovation_map(3, 1, &innovation_map);
            let (network, _) = add_node_by_in_out(network, 1, 3, 0.5, 3);
            let (network, _) = add_connection(network, 0, 4, 0.5, 7);
            Organism{
                network,
                fitness: 3
            }
        };

        let organism_2 = {
            let network = Network::init_from_starting_innovation_map(3, 1, &innovation_map);
            let (network, _) = add_node_by_in_out(network, 1, 3, 0.5, 3);
            let (network, _) = add_node_by_in_out(network, 4, 3, 0.5, 5);
            let (network, _) = add_connection(network, 2, 4, 0.5, 8);
            let (network, _) = add_connection(network, 0, 5, 0.5, 9);

            Organism{
                network,
                fitness: 4
            }
        };

        let organism_child = cross_over(&organism_1, &organism_2);
        assert_eq!(organism_child.network.genome.len(), 9)
    }
}