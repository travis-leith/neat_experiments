
#[derive(PartialEq, PartialOrd, Clone, Copy)]

pub struct NodeIndex(pub usize);
pub struct GeneIndex(pub usize);

mod genome {
    use std::ops::Index;
    use super::NodeIndex;
    use super::GeneIndex;

    pub struct InnovationNumber(pub usize);
    impl InnovationNumber {
        fn inc(mut self) -> InnovationNumber {
            self.0 += 1;
            self
        }
    }

    pub struct Gene {
        pub in_node_id: NodeIndex,
        pub out_node_id: NodeIndex,
        pub weight: f64,
        pub innovation: InnovationNumber,
        pub enabled: bool
    }

    impl Gene {
        pub fn create(in_node_id: usize, out_node_id: usize, weight: f64, innovation: usize, enabled: bool) -> Gene {
            Gene {
                in_node_id: NodeIndex(in_node_id),
                out_node_id: NodeIndex(out_node_id),
                weight,
                innovation: InnovationNumber(innovation),
                enabled,
            }
        }
    }
    
    pub struct Genome(Vec<Gene>);
    impl Genome {
        pub fn calculate_max_node_id(&self) -> usize {
            let res = 
                self.0.iter().fold(NodeIndex(0), |acc, conn| {
                    if conn.in_node_id > acc {
                        conn.in_node_id
                    } else if conn.out_node_id > acc {
                        conn.out_node_id
                    } else {
                        acc
                    }
                });
            res.0
        }

        pub fn iter(&self) -> std::slice::Iter<Gene> {
            self.0.iter()
        }

        pub fn create(data: Vec<Gene>) -> Genome {
            Genome(data)
        }

        pub fn push(&mut self, gene: Gene) {
            self.0.push(gene);
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }        
    }

    impl Index<GeneIndex> for Genome {
        type Output = Gene;
        fn index(&self, index: GeneIndex) -> &Self::Output {
            &self.0[index.0]
        }
    }
}


mod phenome {
    use std::ops::Index;
    use std::ops::IndexMut;
    use super::NodeIndex;
    use super::GeneIndex;
    #[derive(PartialEq)]
    enum NodeType{
        Sensor,
        Hidden,
        Output,
    }
    
    pub struct Node {
        value: f64,
        is_active: bool,
        has_active_inputs: bool,
        pub inputs: Vec<GeneIndex>,
        active_sum: f64,
    }
    
    impl Node {
        fn create(node_type: NodeType) -> Node {
            Node{
                value: 0.,
                is_active: node_type == NodeType::Sensor,
                has_active_inputs: false,
                inputs: Vec::new(),
                active_sum: 0.,
            }
        }
    }
    
    pub struct Phenome(Vec<Node>);
    impl Phenome {
        pub fn create_disconnected(n_sensor_nodes: usize, n_output_nodes: usize, max_node_id: usize) -> Phenome {
            let hidden_start = n_sensor_nodes + n_output_nodes;
            let nodes: Vec<Node> = (0 .. max_node_id + 1).map(|i:usize|{
                if i < n_sensor_nodes {
                    Node::create(NodeType::Sensor)
                } else if i < hidden_start {
                    Node::create(NodeType::Output)
                } else {
                    Node::create(NodeType::Hidden)
                }
            }).collect();
            Phenome(nodes)
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }
    }

    impl Index<NodeIndex> for Phenome {
        type Output = Node;
        fn index(&self, index: NodeIndex) -> &Self::Output {
            &self.0[index.0]
        }
    }

    impl IndexMut<NodeIndex> for Phenome {
        fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
            &mut self.0[index.0]
        }
    }

}

use genome::*;
use phenome::*;

pub struct Network {
    pub phenome: Phenome,
    pub genome: Genome,
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

        Network {
            phenome,
            genome,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use genome::Gene;

    fn genome_sample_1() -> Genome{
        Genome::create(vec![
            Gene::create(0, 3, 0.0, 0, true),
            Gene::create(1, 3, 0.0, 1, true),
            Gene::create(1, 4, 0.0, 2, true),
            Gene::create(2, 4, 0.0, 3, true),
            Gene::create(3, 4, 0.0, 4, true),
        ])
    } 
    
    #[test]
    fn test_genome_max_node_id() {
        let genome = genome_sample_1();
        assert_eq!(genome.calculate_max_node_id(), 5);
    }

    #[test]
    fn network_creation() {
        let n_sensors = 3;
        let n_outputs = 1;
        
        let network =  Network::create_from_genome(n_sensors, n_outputs, genome_sample_1());
        assert_eq!(network.phenome.len(), 5);
        assert_eq!(network.n_output_nodes, n_outputs);
        assert_eq!(network.n_sensor_nodes, n_sensors);
        assert_eq!(network.phenome[NodeIndex(2)].inputs.len(), 0);
        assert_eq!(network.phenome[NodeIndex(3)].inputs.len(), 2);
        assert_eq!(network.phenome[NodeIndex(4)].inputs.len(), 3);
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
}