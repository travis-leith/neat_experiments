use super::genome::{Gene, GeneExt, GeneIndex, Genome};
use super::phenome::{Phenome, NodeIndex, NodeType};
use super::innovation::{InnovationNumber, InnovationContext};

pub struct Network {
    pub phenome: Phenome,
    pub genome: Genome,
    pub activation_order: Vec<Vec<NodeIndex>>,
    pub n_sensor_nodes: usize,
    pub n_output_nodes: usize,
}

use rand::{distributions::{Distribution, Uniform}, RngCore};
impl Network {
    pub fn create_from_genome(n_sensor_nodes: usize, n_output_nodes: usize, genome: Genome) -> Network {
        //TODO remove connections involving dead end nodes
        let mut phenome = Phenome::create_disconnected(n_sensor_nodes, n_output_nodes, genome.next_node_id.0);

        for (i, (gene_key, gene_val)) in genome.iter().enumerate() {
            if gene_val.enabled {
                phenome[gene_key.out_node_id].inputs.push(GeneIndex(i));
            }
        }

        let activation_order = genome.tarjan_scc();
        Network {
            phenome,
            genome,
            activation_order,
            n_sensor_nodes,
            n_output_nodes,
        }
    }

    pub fn init(rng: &mut dyn RngCore, n_sensor_nodes: usize, n_output_nodes: usize) -> Network {
        let genome = Genome::init(rng, n_sensor_nodes, n_output_nodes);

        Network::create_from_genome(n_sensor_nodes, n_output_nodes, genome)
    }

    pub fn activate(&mut self, inputs: Vec<f64>) {
        fn relu(x: f64) -> f64 {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }

        for (i, &input) in inputs.iter().enumerate() {
            self.phenome[NodeIndex(i)].value = input;
        }

        for scc in self.activation_order.iter() {
            for node_index in scc.iter() {
                let node = &self.phenome[*node_index];
                if node.node_type == NodeType::Sensor {
                    continue;
                }
                let active_sum = node.inputs.iter().fold(0., |acc, gene_index| {
                    let (gene_key, gene_value) = &self.genome.get_index(*gene_index);
                    if gene_value.enabled {
                        acc + gene_value.weight * self.phenome[gene_key.in_node_id].value
                    } else {
                        acc
                    }
                });
                self.phenome[*node_index].value = relu(active_sum);
            }
        }
    }

    

    // pub fun add_new_node
}

// pub fn cross_over(rng: &mut dyn RngCore, network_1: &Network, fitness_1: usize, network_2: &Network, fitness_2: usize) -> Network {
//     debug_assert!(network_1.n_output_nodes == network_2.n_output_nodes, "Organisms with mismatching output size cannot be crossed");
//     debug_assert!(network_1.n_sensor_nodes == network_2.n_sensor_nodes, "Organisms with mismatching input size cannot be crossed");

//     let new_genome = super::genome::cross_over(rng, &network_1.genome, fitness_1, &network_2.genome, fitness_2);
//     Network::create_from_genome(network_1.n_sensor_nodes, network_1.n_output_nodes, new_genome)
// }

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn genome_sample_feed_forward_1() -> Genome{
        Genome::create(vec![
            Gene::create(0, 4, -0.1, 0, true),
            Gene::create(4, 3, 0.6, 1, true),
            Gene::create(1, 5, -0.8, 2, true),
            Gene::create(5, 3, -0.9, 3, true),
            Gene::create(0, 5, 0.6, 4, true),
            Gene::create(5, 2, 0.4, 5, true),
        ])
    }

    fn genome_sample_recurrent_1() -> Genome{
        Genome::create(vec![
            Gene::create(3, 2, 0.9, 0, true),
            Gene::create(1, 4, -0.8, 1, true),
            Gene::create(4, 3, 0.1, 2, true),
            Gene::create(5, 2, -0.4, 3, true),
            Gene::create(0, 4, -0.8, 4, true),
            Gene::create(3, 5, 0.5, 5, true),
            Gene::create(5, 4, -0.1, 6, true),
        ])
    }

    #[test]
    fn network_creation() {
        let n_sensors = 3;
        let n_outputs = 1;
        
        let network =  Network::create_from_genome(n_sensors, n_outputs, genome_sample_feed_forward_1());
        assert_eq!(network.phenome.len(), 6);
        assert_eq!(network.n_output_nodes, n_outputs);
        assert_eq!(network.n_sensor_nodes, n_sensors);
        assert_eq!(network.phenome[NodeIndex(2)].inputs.len(), 1);
        assert_eq!(network.phenome[NodeIndex(3)].inputs.len(), 2);
        assert_eq!(network.phenome[NodeIndex(4)].inputs.len(), 1);
    }

    #[test]
    fn network_init(){
        let mut rng = rand::thread_rng();
        let n_sensor_nodes = 9;
        let n_output_nodes = 10;
        let n_total = n_sensor_nodes + n_output_nodes;
        let network = Network::init(&mut rng, n_sensor_nodes, n_output_nodes);
        assert_eq!(network.genome.get_index(GeneIndex(89)).1.innovation.0, 89);
        assert_eq!(network.genome.len(), 90);
        assert_eq!(network.phenome.len(), n_total);
        assert_eq!(network.n_output_nodes, n_output_nodes);
        assert_eq!(network.n_sensor_nodes, n_sensor_nodes);

        for node_index in network.n_sensor_nodes..network.phenome.len() {
            let node = &network.phenome[NodeIndex(node_index)];
            let l = node.inputs.len();
            assert_eq!(l, n_sensor_nodes)
        }
    }

    #[test]
    fn feed_forward() {
        let n_sensors = 2;
        let n_outputs = 2;
        let genome = genome_sample_feed_forward_1();
        let mut network =  Network::create_from_genome(n_sensors, n_outputs, genome);

        network.activate(vec![0.5, -0.2]);
        assert_approx_eq!(network.phenome[NodeIndex(2)].value, 0.184);
        assert_approx_eq!(network.phenome[NodeIndex(3)].value, 0.);
    }

    #[test]
    fn recurrent() {
        let genome = genome_sample_recurrent_1();
        let mut network =  Network::create_from_genome(2, 1, genome);

        let inputs = vec![-0.9, 0.6];
        network.activate(inputs.clone());
        assert_approx_eq!(network.phenome[NodeIndex(2)].value, 0.);

        network.activate(inputs.clone());
        assert_approx_eq!(network.phenome[NodeIndex(2)].value, 0.0216);

        network.activate(inputs.clone());
        assert_approx_eq!(network.phenome[NodeIndex(2)].value, 0.0168);
        
    }
}