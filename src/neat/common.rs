use std::iter;

pub struct Connection {
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

pub struct Genome {
    // genes: Vec<ConnectionGene>,
    n_sensor_nodes: usize,
    // n_output_nodes: usize,
    // n_total_nodes: usize,
    // node_values: Vec<f64>,
    // node_types: Vec<NodeType>,
    input_connections: Vec<Vec<Connection>>,
    active_nodes: Vec<bool>,
    node_values: Vec<f64>
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
                genome.node_values[node_index] += conn.weight * genome.node_values[conn.in_node_id];
            } else {
                let in_id = conn.in_node_id;
                println!("input {in_id} for node {node_index} not active");
                all_active = false;
            }
        }
    }

    for x in &mut genome.node_values[genome.n_sensor_nodes ..] {
        *x = relu(*x);
    }

    all_active
}

pub fn activate(genome: &mut Genome) {
    let mut iteration_count = 0;
    loop {
        iteration_count += 1;
        if activation_pulse(genome) || iteration_count > 20 {break;}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    #[test]
    fn feed_formward() {
        fn make_connection(in_id: usize, out_id: usize, weight: f64) -> Connection{
            Connection {
                in_node_id: in_id - 1,
                out_node_id: out_id - 1,
                weight: weight
            }
        }

        let conn_1_to_3 = make_connection(1, 3, 0.7);
        let conn_1_to_4 = make_connection(1, 4, 0.6);
        let conn_2_to_4 = make_connection(2, 4, -0.8);
        let conn_3_to_6 = make_connection(3, 6, 0.6);
        let conn_4_to_5 = make_connection(4, 5, 0.4);
        let conn_4_to_6 = make_connection(4, 6, -0.9);

        let inputs_3 = vec![conn_1_to_3];
        let inputs_4 = vec![conn_1_to_4, conn_2_to_4];
        let inputs_5 = vec![conn_4_to_5];
        let inputs_6 = vec![conn_3_to_6, conn_4_to_6];

        let mut genome = Genome {
            n_sensor_nodes: 2,
            input_connections: vec![inputs_3, inputs_4, inputs_5, inputs_6],
            active_nodes: vec![true, true, false, false, false, false], //inputs are always active
            node_values: vec![0.5, 0.2, 0., 0., 0., 0.] //inputs are initialized with their values, other nodes values are initialized with zero
        };
        activate(&mut genome);
        assert_approx_eq!(genome.node_values[4], 0.056);
        assert_approx_eq!(genome.node_values[5], 0.084);
    }

}