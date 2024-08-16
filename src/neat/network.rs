use super::genome::{Genome, GeneIndex, Gene};
use super::phenome::{Phenome, NodeIndex, NodeType};

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
        let max_node_id = genome.calculate_max_node_id();
        let mut phenome = Phenome::create_disconnected(n_sensor_nodes, n_output_nodes, max_node_id);

        for (i, gene) in genome.iter().enumerate() {
            if gene.enabled {
                phenome[gene.out_node_id].inputs.push(GeneIndex(i));
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
        let between = Uniform::from(-1.0..1.0);

        let n_connections = n_sensor_nodes * n_output_nodes;
        let mut genome : Genome = Genome::create(Vec::with_capacity(n_connections));

        for out_node_ind in 0..n_output_nodes {
            let out_node_id = out_node_ind + n_sensor_nodes;
            for in_node_ind in 0..n_sensor_nodes {
                let in_node_id = in_node_ind;
                let innovation_number = out_node_ind * n_sensor_nodes + in_node_ind;
                let conn = Gene::create(in_node_id, out_node_id, between.sample(rng), innovation_number, true);
                genome.push(conn);
            }
        }

        Network::create_from_genome(n_sensor_nodes, n_output_nodes, genome)
    }

    pub fn activate(mut self, inputs: Vec<f64>) -> Network {
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
                    let gene = &self.genome[*gene_index];
                    if gene.enabled {
                        acc + gene.weight * self.phenome[gene.in_node_id].value
                    } else {
                        acc
                    }
                });
                self.phenome[*node_index].value = relu(active_sum);
            }
        }
        self
    }
}

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
        assert_eq!(network.genome[GeneIndex(89)].innovation.0, 89);
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
        let network =  Network::create_from_genome(n_sensors, n_outputs, genome);

        let network = network.activate(vec![0.5, -0.2]);
        assert_approx_eq!(network.phenome[NodeIndex(2)].value, 0.184);
        assert_approx_eq!(network.phenome[NodeIndex(3)].value, 0.);
    }
}