use tailcall::tailcall;
use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;

pub struct Connection {
    in_node_id: usize,
    out_node_id: usize,
    weight: f64,
    innovation_num: usize,
    enabled: bool
}

impl Connection {
    fn new(in_node_id: usize, out_node_id: usize, weight: f64, innovation_num: usize) -> Connection {
        Connection{
            in_node_id,
            out_node_id,
            weight,
            innovation_num,
            enabled: true
        }
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

    fn new(n_sensor_nodes: usize, n_output_nodes: usize, innovation_map: HashMap<(usize, usize), usize>) -> Network {
        let mut res = Network::empty(n_sensor_nodes, n_output_nodes);
        let between = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();
        for ((in_node_id, out_node_id), innovation_num) in innovation_map.iter() {
            let weight = between.sample(&mut rng);
            res.genome.push(Connection::new(*in_node_id, *out_node_id, weight, *innovation_num));
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

    fn set_phenotype(mut self) -> Network {
        let n_activateable = self.n_output_nodes + self.n_hidden_nodes;
        let node_offset = self.n_sensor_nodes;
        self.input_connections_map = (0..n_activateable).map(|_| Vec::new()).collect();
        for (conn_index, conn) in self.genome.iter().enumerate() {
            self.input_connections_map[conn.out_node_id - node_offset].push(conn_index)
        }
        self
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

    let new_conn = Connection{
        in_node_id: in_id,
        out_node_id: out_id,
        weight: weight,
        innovation_num: global_innovation,
        enabled: true
    };

    network.genome.push(new_conn);
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
}