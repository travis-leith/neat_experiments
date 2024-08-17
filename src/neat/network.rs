use super::genome::{Genome, GeneIndex, Gene};
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
    }

    

    // pub fun add_new_node
}

pub fn add_connection(mut network: Network, mut innovation_context: InnovationContext, in_node_id: NodeIndex, out_node_id: NodeIndex, weight: f64) -> (Network, InnovationContext) {
    debug_assert!(in_node_id != out_node_id, "Tried to add a connection where input is the same node as output");
    debug_assert!(in_node_id.0 < network.n_sensor_nodes || in_node_id.0 >= (network.n_sensor_nodes + network.n_output_nodes), "Tried to add a connection that inputs from an output node");
    debug_assert!(out_node_id.0 >= network.n_sensor_nodes, "Tried to add a connection that outputs to a sensor node");
    debug_assert!(out_node_id.0 < network.phenome.len(), "Tried to add a connection that outputs beyond node count");

    // let out_node = &mut network.nodes[out_node_id];
    // if !out_node.input_node_ids.contains(&in_node_id) {
    //     let new_key = (in_node_id, out_node_id);
    //     // let new_innnov = InnovationNumber(global_innovation.len());

    //     let innov_number = 
    //         match innovation_record.try_insert(new_key, global_innovation.clone()) {
    //             Ok(i) => {
    //                 // println!("adding key: {in_node_id},{out_node_id} and value {}", global_innovation.0);
    //                 global_innovation.inc();
    //                 i.clone()
    //             },
    //             Err(x) => {
    //                 network.out_of_order = true;
    //                 x.entry.get().clone()
    //             }
    //         }
    //         ;
    //     // println!("selected number {}", innov_number.0);
    //     let new_conn = Connection{
    //         in_node_id,
    //         out_node_id,
    //         weight,
    //         innovation: innov_number,
    //         enabled: true
    //     };
    //     let new_conn_ix = network.genome.len();
        
    //     out_node.input_connection_ids.insert(new_conn_ix);
    //     out_node.input_node_ids.insert(in_node_id);
    //     network.genome.push(new_conn);
        
    //     network
    // } else {
    //     network
    // }
    todo!()
}


pub fn add_node(mut network: Network, mut innovation_context: InnovationContext, existing_conn_index: GeneIndex) -> (Network, InnovationContext) {
    let in_node_id = network.genome[existing_conn_index].in_node_id;
    let out_node_id = network.genome[existing_conn_index].out_node_id;
    let weight = network.genome[existing_conn_index].weight;

    // let genome = &mut network.genome[..];
    // let nodes = &mut network.nodes[..];
    // let existing_conn = &mut genome[existing_conn_index];
    // if !existing_conn.enabled {
    //     return network
    // }
    
    // let new_node_id = nodes.len();
    // let output_node = &mut nodes[existing_conn.out_node_id];
    
    // let new_hidden_node = Node{
    //     has_active_inputs: false,
    //     input_connection_ids: FxHashSet::default(),
    //     input_node_ids: FxHashSet::default(),
    //     is_active: false,
    //     active_sum: 0.,
    //     value: 0.,
    // };

    // existing_conn.enabled = false;
    // output_node.input_connection_ids.remove(&existing_conn_index);
    // output_node.input_node_ids.remove(&in_node_id);

    // network.nodes.push(new_hidden_node);

    // let new_network = add_connection(network, in_node_id, new_node_id, 1., global_innovation, innovation_record);

    // add_connection(new_network, new_node_id, out_node_id, weight, global_innovation, innovation_record)
    todo!()

}

pub fn cross_over(rng: &mut dyn RngCore, network_1: &Network, fitness_1: usize, network_2: &Network, fitness_2: usize) -> Network {
    debug_assert!(network_1.n_output_nodes == network_2.n_output_nodes, "Organisms with mismatching output size cannot be crossed");
    debug_assert!(network_1.n_sensor_nodes == network_2.n_sensor_nodes, "Organisms with mismatching input size cannot be crossed");

    let new_genome = super::genome::cross_over(rng, &network_1.genome, fitness_1, &network_2.genome, fitness_2);
    Network::create_from_genome(network_1.n_sensor_nodes, network_1.n_output_nodes, new_genome)
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
        let mut network =  Network::create_from_genome(n_sensors, n_outputs, genome);

        network.activate(vec![0.5, -0.2]);
        assert_approx_eq!(network.phenome[NodeIndex(2)].value, 0.184);
        assert_approx_eq!(network.phenome[NodeIndex(3)].value, 0.);
    }
}