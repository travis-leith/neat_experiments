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

// enum NodeType {
//     Sensor,
//     Hidden,
//     Output
// }

//TODO change Genome to Network
pub struct Genome {
    // genes: Vec<ConnectionGene>,
    n_sensor_nodes: usize,
    // n_output_nodes: usize,
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

fn print_genome(genome: &Genome) {
    println!("Connections...");
    for inputs in &genome.input_connections {
        for connection in inputs {

            println!("{} -> {} = {}", connection.in_node_id, connection.out_node_id, connection.weight);
        }
    }
    println!("Nodes active status...");
    for (i, b) in genome.active_nodes.iter().enumerate() {
        println!("{i}:{b}");
    }
    println!("Node values...");
    for (i, f) in genome.node_values.iter().enumerate() {
        println!("{i}:{f}");
    }
}

fn activation_pulse(mut genome: Genome) -> (Genome, bool) {
    
    for (node_index_offset, input_connections) in genome.input_connections.iter().enumerate() {
        let node_index = node_index_offset + genome.n_sensor_nodes;
        genome.nodes_with_active_inputs[node_index_offset] = false;

        genome.active_sums[node_index] = 0.;
        for conn in input_connections {
            if genome.active_nodes[conn.in_node_id] {
                genome.nodes_with_active_inputs[node_index_offset] = true;
                let to_add = conn.weight * genome.node_values[conn.in_node_id];
                // println!("node {node_index} input {} to add {} * {} = {to_add}", conn.in_node_id, conn.weight, genome.node_values[conn.in_node_id]);
                genome.active_sums[node_index] += to_add;
            }
            // } else {
            //     let in_id = conn.in_node_id;
            //     println!("input {in_id} for node {node_index} not active");
                
            // }
        }
    }

    let mut all_active = true;
    for (node_index_offset, f) in &mut genome.node_values[genome.n_sensor_nodes ..].iter_mut().enumerate() {
        let node_index = node_index_offset + genome.n_sensor_nodes;
        *f = relu(genome.active_sums[node_index]);
        // println!("node {node_index} value is now {f}");
        if genome.nodes_with_active_inputs[node_index_offset] {
            // println!("setting node {node_index} to active");
            genome.active_nodes[node_index] = true;
        } else {
            all_active = false;
        }
    }

    (genome, all_active)
}

//TODO: add validate genome function and associated tests

pub fn activate(genome: Genome) -> Genome {
    #[tailcall]
    fn activate_inner(genome: Genome, remaining_iterations: usize) -> Genome {
        if remaining_iterations == 0 {
            panic!("Too many iterations :(")
        } else {
            let (new_genome, all_activated) = activation_pulse(genome);
            if all_activated {
                new_genome
            } else {
                activate_inner(new_genome, remaining_iterations - 1)
            }
        }
    }
    activate_inner(genome, 20)        
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

        let mut genome = Genome {
            n_sensor_nodes: 2,
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

    #[test]
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

        let mut genome = Genome {
            n_sensor_nodes: 2,
            input_connections: vec![inputs_2, inputs_3, inputs_4, inputs_5],
            active_nodes: vec![true, true, false, false, false, false], //sensors are always active
            nodes_with_active_inputs: vec![false, false, false, false], //sensors do not have inputs so no need to have values for them here 
            active_sums: vec![0., 0., 0., 0., 0., 0.],
            node_values: vec![-0.9, 0.6, 0., 0., 0., 0.] //inputs are initialized with their values, other nodes values are initialized with zero
        };
        let new_genome = activate(genome);
        assert_approx_eq!(new_genome.node_values[4], 0.024);
        assert_approx_eq!(new_genome.node_values[5], 0.0216);
    }
}