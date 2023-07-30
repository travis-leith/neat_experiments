struct ConnectionGene {
    in_node_id: usize,
    out_node_id: usize,
    weight: f64,
    // enabled: bool,
    // innovation_num: usize
}

// enum NodeType {
//     Sensor,
//     Hidden,
//     Output
// }

struct Genome {
    // genes: Vec<ConnectionGene>,
    n_sensor_nodes: usize,
    // n_output_nodes: usize,
    // n_total_nodes: usize,
    // node_values: Vec<f64>,
    // node_types: Vec<NodeType>,
    input_connections: Vec<Vec<ConnectionGene>>,
    active_nodes: Vec<bool>,
    activation_values: Vec<f64>
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn activation_pulse(genome: &mut Genome) -> bool {
    let mut all_active = true;
    for (node_index_offset, input_connections) in genome.input_connections.iter().enumerate() {
        let node_index = node_index_offset + genome.n_sensor_nodes;
        genome.active_nodes[node_index] = false;
        for conn in input_connections {
            if genome.active_nodes[conn.in_node_id] {
                genome.active_nodes[node_index] = true;
                genome.activation_values[node_index] += conn.weight * genome.activation_values[conn.in_node_id];
            } else {
                all_active = false;
            }
        }
    }

    for x in &mut genome.activation_values[genome.n_sensor_nodes ..] {
        *x = relu(*x);
    }

    all_active
}

fn activate(genome: &mut Genome) {
    loop {
        if activation_pulse(genome) {break;}
    }
}