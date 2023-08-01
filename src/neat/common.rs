// use std::iter;
use tailcall::tailcall;

pub struct Connection {
    in_node_id: usize,
    out_node_id: usize,
    weight: f64,
    innovation_num: usize
    // enabled: bool,
    // innovation_num: usize
}

pub struct Network {
    // genes: Vec<ConnectionGene>,
    n_sensor_nodes: usize,
    n_output_nodes: usize,
    // n_total_nodes: usize,
    // node_values: Vec<f64>,
    // node_types: Vec<NodeType>,
    input_connections: Vec<Vec<Connection>>,
    nodes_with_active_inputs: Vec<bool>,
    active_nodes: Vec<bool>,
    active_sums: Vec<f64>,
    node_values: Vec<f64>
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn print_genome(network: &Network) {
    println!("Connections...");
    for inputs in &network.input_connections {
        for connection in inputs {

            println!("{} -> {} = {}", connection.in_node_id, connection.out_node_id, connection.weight);
        }
    }
    println!("Nodes active status...");
    for (i, b) in network.active_nodes.iter().enumerate() {
        println!("{i}:{b}");
    }
    println!("Node values...");
    for (i, f) in network.node_values.iter().enumerate() {
        println!("{i}:{f}");
    }
}

fn activation_pulse(mut network: Network) -> (Network, bool) {
    
    for (node_index_offset, input_connections) in network.input_connections.iter().enumerate() {
        let node_index = node_index_offset + network.n_sensor_nodes;
        network.nodes_with_active_inputs[node_index_offset] = false;

        network.active_sums[node_index] = 0.;
        for conn in input_connections {
            if network.active_nodes[conn.in_node_id] {
                network.nodes_with_active_inputs[node_index_offset] = true;
                let to_add = conn.weight * network.node_values[conn.in_node_id];
                // println!("node {node_index} input {} to add {} * {} = {to_add}", conn.in_node_id, conn.weight, genome.node_values[conn.in_node_id]);
                network.active_sums[node_index] += to_add;
            }
            // } else {
            //     let in_id = conn.in_node_id;
            //     println!("input {in_id} for node {node_index} not active");
                
            // }
        }
    }

    let mut all_active = true;
    for (node_index_offset, f) in &mut network.node_values[network.n_sensor_nodes ..].iter_mut().enumerate() {
        let node_index = node_index_offset + network.n_sensor_nodes;
        *f = relu(network.active_sums[node_index]);
        // println!("node {node_index} value is now {f}");
        if network.nodes_with_active_inputs[node_index_offset] {
            // println!("setting node {node_index} to active");
            network.active_nodes[node_index] = true;
        } else {
            all_active = false;
        }
    }

    (network, all_active)
}

//TODO: add validate genome function and associated tests

pub fn activate(network: Network) -> Network {
    #[tailcall]
    fn activate_inner(genome: Network, remaining_iterations: usize) -> Network {
        if remaining_iterations == 0 {
            panic!("Too many iterations :(")
        } else {
            let (new_network, all_activated) = activation_pulse(genome);
            if all_activated {
                new_network
            } else {
                activate_inner(new_network, remaining_iterations - 1)
            }
        }
    }
    activate_inner(network, 20)        
}

fn add_connection(mut network: Network, in_id: usize, out_id: usize, weight: f64, global_innovation: usize) -> (Network, usize) {
    if out_id < network.n_sensor_nodes {
        panic!("Tried to add a connection that outputs to a sensor node")
    }

    if in_id >= network.n_sensor_nodes && in_id < (network.n_sensor_nodes + network.n_output_nodes) {
        panic!("Tried to add a connection that inputs from an output node")
    }

    if out_id > network.n_sensor_nodes + network.input_connections.len() {
        panic!("Tried to add a connection with an output that skips an available id")
    }

    let new_conn = Connection{
        in_node_id: in_id,
        out_node_id: out_id,
        weight: weight,
        innovation_num: global_innovation
    };

    if out_id == network.n_sensor_nodes + network.input_connections.len() {
        network.input_connections.push(vec![new_conn]);
        network.active_nodes.push(false);
        network.nodes_with_active_inputs.push(false);
        network.active_sums.push(0.);
        network.node_values.push(0.);
    } else {
        let out_id_offset = out_id - network.n_sensor_nodes;
        network.input_connections[out_id_offset].push(new_conn)
    }

    (network, global_innovation + 1)
}

fn create_empty_network(n_sensors: usize, n_outputs: usize) -> Network {
    let mut active_nodes = vec![true; n_sensors];
    active_nodes.append(&mut vec![false; n_outputs]);
    Network {
        n_sensor_nodes: n_sensors,
        n_output_nodes: n_outputs,
        input_connections: (0..n_outputs).map(|_| Vec::new()).collect(),
        active_nodes: active_nodes,
        nodes_with_active_inputs: vec![false; n_outputs], //sensors do not have inputs so no need to have values for them here 
        active_sums: vec![0.; n_sensors + n_outputs], //TODO change active sums to be non sensor only
        node_values: vec![0.; n_sensors + n_outputs]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn make_connection(in_id: usize, out_id: usize, weight: f64, innov_num: usize) -> Connection{
        Connection {
            in_node_id: in_id,
            out_node_id: out_id,
            weight: weight,
            innovation_num: innov_num
        }
    }

    #[test]
    fn network_creation() {
        let n_sensors = 3;
        let n_outputs = 2;
        let n_total = n_sensors + n_outputs;
        let network = create_empty_network(n_sensors, n_outputs);
        assert_eq!(network.input_connections.len(), n_outputs);
        assert_eq!(network.active_nodes.len(), n_total);
        assert_eq!(network.nodes_with_active_inputs.len(), n_outputs);
        assert_eq!(network.active_sums.len(), n_total);
        assert_eq!(network.node_values.len(), n_total);

    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that outputs to a sensor node")]
    fn cannot_add_connection_out_to_sensor() {
        let network = create_empty_network(2, 2);
        let _ = add_connection(network, 1, 0, 0., 0);
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection that inputs from an output node")]
    fn cannot_add_connection_in_from_output() {
        let network = create_empty_network(2, 2);
        let _ = add_connection(network, 2, 3, 0., 0);
    }

    #[test]
    #[should_panic(expected = "Tried to add a connection with an output that skips an available id")]
    fn cannot_add_connection_beyond_length() {
        let network = create_empty_network(2, 2);
        let _ = add_connection(network, 1, 5, 0., 0);
    }

    #[test]
    fn add_interior_connection(){
        let network = create_empty_network(2, 2);
        let (network, _) = add_connection(network, 1, 2, 0.5, 1);
        assert_eq!(network.input_connections[0].len(), 1);
        assert_eq!(network.input_connections[0][0].in_node_id, 1);
        assert_eq!(network.input_connections[0][0].innovation_num, 1);
    }
    // #[test]
    fn feed_formward() {
        let conn_0_to_2 = make_connection(0, 2, -0.1, 0);
        let conn_0_to_3 = make_connection(0, 3, 0.6, 1);
        let conn_1_to_3 = make_connection(1, 3, -0.8, 2);
        let conn_2_to_5 = make_connection(2, 5, 0.6, 3);
        let conn_3_to_4 = make_connection(3, 4, 0.4, 4);
        let conn_3_to_5 = make_connection(3, 5, -0.9, 5);

        let inputs_2 = vec![conn_0_to_2];
        let inputs_3 = vec![conn_0_to_3, conn_1_to_3];
        let inputs_4 = vec![conn_3_to_4];
        let inputs_5 = vec![conn_2_to_5, conn_3_to_5];

        let mut genome = Network {
            n_sensor_nodes: 2,
            n_output_nodes: 2,
            input_connections: vec![inputs_2, inputs_3, inputs_4, inputs_5],
            active_nodes: vec![true, true, false, false, false, false], //sensors are always active
            nodes_with_active_inputs: vec![false, false, false, false], //sensors do not have inputs so no need to have values for them here 
            active_sums: vec![0., 0., 0., 0., 0., 0.],
            node_values: vec![0.5, -0.2, 0., 0., 0., 0.] //inputs are initialized with their values, other nodes values are initialized with zero

        };
        let new_genome = activate(genome);
        assert_approx_eq!(new_genome.node_values[4], 0.184);
        assert_approx_eq!(new_genome.node_values[5], 0.);
    }

    // #[test]
    fn recurrent() {
        let conn_0_to_2 = make_connection(0, 2, -0.8, 0);
        let conn_1_to_2 = make_connection(1, 2, -0.8, 1);
        let conn_2_to_4 = make_connection(2, 4, 0.1, 2);
        let conn_4_to_3 = make_connection(4, 3, 0.5, 3);
        let conn_3_to_2 = make_connection(3, 2, -0.1, 4);
        let conn_3_to_5 = make_connection(3, 5, -0.4, 5);
        let conn_4_to_5 = make_connection(4, 5, 0.9, 6);

        let inputs_2 = vec![conn_0_to_2, conn_1_to_2, conn_3_to_2];
        let inputs_3 = vec![conn_4_to_3];
        let inputs_4 = vec![conn_2_to_4];
        let inputs_5 = vec![conn_3_to_5, conn_4_to_5];

        let mut genome = Network {
            n_sensor_nodes: 2,
            n_output_nodes: 1,
            input_connections: vec![inputs_2, inputs_3, inputs_4, inputs_5],
            active_nodes: vec![true, true, false, false, false, false], //sensors are always active
            nodes_with_active_inputs: vec![false, false, false, false], //sensors do not have inputs so no need to have values for them here 
            active_sums: vec![0., 0., 0., 0., 0., 0.],
            node_values: vec![-0.9, 0.6, 0., 0., 0., 0.] //inputs are initialized with their values, other nodes values are initialized with zero
        };
        let new_genome = activate(genome);
        assert_approx_eq!(new_genome.node_values[5], 0.0216);
    }
}