#[derive(Clone)]
struct Gene {
    in_node_id: usize,
    out_node_id: usize,
    weight: f64,
    enabled: bool,
}

#[derive(Clone)]
struct Genome {
    genes: Vec<Gene>,
}

impl Genome {
    fn new() -> Self {
        Genome { genes: Vec::new() }
    }

    fn add_gene(&mut self, gene: Gene) {
        self.genes.push(gene);
    }
}

// Define the Network structure
struct Network {
    n_sensor_nodes: usize,
    n_output_nodes: usize,
    genome: Genome,
    node_values: Vec<f64>, // Store the current state of the nodes
}

impl Network {
    fn create_from_genome(n_sensor_nodes: usize, n_output_nodes: usize, genome: Genome) -> Self {
        let total_nodes = n_sensor_nodes + n_output_nodes;
        Network {
            n_sensor_nodes,
            n_output_nodes,
            genome,
            node_values: vec![0.0; total_nodes], // Initialize node values to zero
        }
    }

    fn activate(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        // Initialize sensor node values
        for (i, &input) in inputs.iter().enumerate() {
            self.node_values[i] = input;
        }

        // Define the activation function (sigmoid)
        fn sigmoid(x: f64) -> f64 {
            1.0 / (1.0 + (-x).exp())
        }

        // Propagate values through the network
        let mut new_values = self.node_values.clone();
        let mut iterations = 0;
        let max_iterations = 10; // To handle recurrent connections
        let epsilon = 1e-6; // Convergence threshold

        while iterations < max_iterations {
            for gene in self.genome.genes.iter() {
                if gene.enabled {
                    let input_value = self.node_values[gene.in_node_id];
                    new_values[gene.out_node_id] += input_value * gene.weight;
                }
            }

            // Apply activation function to all nodes except sensor nodes
            for i in self.n_sensor_nodes..(self.n_sensor_nodes + self.n_output_nodes) {
                new_values[i] = sigmoid(new_values[i]);
            }

            // Check for convergence
            let mut converged = true;
            for i in 0..new_values.len() {
                if (new_values[i] - self.node_values[i]).abs() > epsilon {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }

            self.node_values = new_values.clone();
            iterations += 1;
        }

        self.node_values[self.n_sensor_nodes..].to_vec()
    }
}

fn main() {
    let mut genome = Genome::new();
    genome.add_gene(Gene {
        in_node_id: 0,
        out_node_id: 3,
        weight: 1.0,
        enabled: true,
    });
    genome.add_gene(Gene {
        in_node_id: 1,
        out_node_id: 3,
        weight: 1.0,
        enabled: true,
    });
    genome.add_gene(Gene {
        in_node_id: 2,
        out_node_id: 3,
        weight: 1.0,
        enabled: true,
    });

    let mut network = Network::create_from_genome(3, 1, genome);
    let inputs = vec![1.0, 0.5, -1.0];
    let outputs = network.activate(inputs);
    println!("{:?}", outputs);

    // Feed new data to the network
    let new_inputs = vec![0.5, 1.0, 0.0];
    let new_outputs = network.activate(new_inputs);
    println!("{:?}", new_outputs);
}