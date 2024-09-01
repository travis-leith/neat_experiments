use std::ops::{Index, IndexMut};

use rand::{seq::SliceRandom, RngCore};

use crate::neat::{genome::Genome, phenome::NodeType};

use super::{genome::GeneNumber, phenome::{NodeNumber, Phenome}};

// use super::network::Network;

#[derive(Clone, Copy)]
pub struct OrganismIndex(pub usize);


#[derive(Clone)]
pub struct Organism {
    pub phenome: Phenome,
    pub genome: Genome,
    pub activation_order: Vec<NodeNumber>,
    pub fitness: usize
}

impl Organism {
    

    pub fn create_from_genome(genome: Genome, initial_fitness: usize) -> Organism {
        //TODO remove connections involving dead end nodes
        //TODO create phenome::create_from_genome
        let mut phenome = Phenome::create_disconnected(genome.n_sensor_nodes, genome.n_output_nodes, genome.next_node_id.0);
        

        for (i, (gene_key, gene_val)) in genome.iter().enumerate() {
            if gene_val.enabled {
                phenome[gene_key.out_node_id].inputs.push(GeneNumber(i));
            }
        }

        let activation_order = genome.rev_dfs_order_petgraph();
        
        Organism {
            phenome,
            genome,
            activation_order,
            fitness: initial_fitness
        }
    }

    pub fn init<R: RngCore>(rng: &mut R, n_sensor_nodes: usize, n_output_nodes: usize, initial_fitness: usize) -> Organism {
        let genome = Genome::init(rng, n_sensor_nodes, n_output_nodes);
        Self::create_from_genome(genome, initial_fitness)
    }
    pub fn activate(&mut self, sensor_values: &Vec<f64>) -> Vec<f64> {
        debug_assert!(sensor_values.len() == self.genome.n_sensor_nodes);
        fn relu(x: f64) -> f64 {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }

        // fn sigmoid(x: f64) -> f64 {
        //     1.0 / (1.0 + (-4.9 * x).exp())
        // }

        for (i, &input) in sensor_values.iter().enumerate() {
            self.phenome[NodeNumber(i)].value = input;
        }

        for &node_index in &self.activation_order {
            let node = &self.phenome[node_index];
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
            self.phenome[node_index].value = relu(active_sum);
        }

        let outputs = self.phenome.iter().skip(self.genome.n_sensor_nodes).take(self.genome.n_output_nodes).map(|node| node.value).collect();
        outputs
    }

    pub fn clear_values(&mut self) {
        for node in self.phenome.iter_mut() {
            node.value = 0.;
        }
    }
}

pub struct Organisms(Vec<Organism>);
use rayon::prelude::*;

impl Organisms {
    pub fn push(&mut self, organism: Organism) {
        self.0.push(organism);
    }

    pub fn new(data: Vec<Organism>) -> Organisms {
        Organisms(data)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<Organism> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Organism> {
        self.0.iter_mut()
    }

    pub fn par_chunks_mut(&mut self, chunk_size: usize) -> rayon::slice::ChunksMut<Organism> {
        self.0.par_chunks_mut(chunk_size)
    }

    pub fn shuffle<R: RngCore>(&mut self, rng: &mut R) {
        self.0.shuffle(rng);
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [Organism], &mut [Organism]) {
        self.0.split_at_mut(mid)
    }
}

impl Index<OrganismIndex> for Organisms {
    type Output = Organism;
    fn index(&self, index: OrganismIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

impl IndexMut<OrganismIndex> for Organisms {
    fn index_mut(&mut self, index: OrganismIndex) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}

impl<'a> IntoParallelRefMutIterator<'a> for Organisms {
    type Item = &'a mut Organism;
    type Iter = rayon::slice::IterMut<'a, Organism>;

    fn par_iter_mut(&'a mut self) -> Self::Iter {
        self.0.par_iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::neat::{genome::{Gene, GeneExt, GeneNumber, GeneKey, Genome}, organism::Organism, phenome::{Node, NodeNumber, NodeType}};
    use assert_approx_eq::assert_approx_eq;
    use itertools::Itertools;
    use petgraph::{data::Build, visit::{IntoNodeIdentifiers, IntoNodeReferences}, Directed};

    fn genome_sample_feed_forward_1() -> Genome{
        Genome::create(vec![
            Gene::create(0, 4, -0.1, 0, true),
            Gene::create(4, 3, 0.6, 1, true),
            Gene::create(1, 5, -0.8, 2, true),
            Gene::create(5, 3, -0.9, 3, true),
            Gene::create(0, 5, 0.6, 4, true),
            Gene::create(5, 2, 0.4, 5, true),
        ], 2, 2)
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
        ], 2, 1)
    }

    fn genome_sample_dead_ends() -> Genome {
        Genome::create(vec![
            Gene::create(0, 4, 0.0, 0, true),
            Gene::create(4, 2, 0.0, 1, true),
            Gene::create(7, 6, 0.0, 2, true),
            Gene::create(4, 6, 0.0, 3, true),
            Gene::create(4, 18, 0.0, 4, true),
            Gene::create(6, 5, 0.0, 5, true),
            Gene::create(5, 4, 0.0, 6, true),
            Gene::create(1, 5, 0.0, 7, true),
            Gene::create(5, 3, 0.0, 8, true),
        ], 2, 2)
    }

    #[test]
    fn network_creation() {
        let network =  Organism::create_from_genome(genome_sample_feed_forward_1(), 0);
        assert_eq!(network.phenome.len(), 6);
        assert_eq!(network.phenome[NodeNumber(2)].inputs.len(), 1);
        assert_eq!(network.phenome[NodeNumber(3)].inputs.len(), 2);
        assert_eq!(network.phenome[NodeNumber(4)].inputs.len(), 1);
    }

    #[test]
    fn network_init(){
        let mut rng = rand::thread_rng();
        let n_sensor_nodes = 9;
        let n_output_nodes = 10;
        let n_total = n_sensor_nodes + n_output_nodes;
        let network = Organism::init(&mut rng, n_sensor_nodes, n_output_nodes, 0);
        assert_eq!(network.genome.get_index(GeneNumber(89)).1.innovation.0, 89);
        assert_eq!(network.genome.len(), 90);
        assert_eq!(network.phenome.len(), n_total);
        assert_eq!(network.genome.n_output_nodes, n_output_nodes);
        assert_eq!(network.genome.n_sensor_nodes, n_sensor_nodes);

        for node_index in network.genome.n_sensor_nodes..network.phenome.len() {
            let node = &network.phenome[NodeNumber(node_index)];
            let l = node.inputs.len();
            assert_eq!(l, n_sensor_nodes)
        }
    }

    #[test]
    fn feed_forward() {
        let genome = genome_sample_feed_forward_1();
        let mut organism =  Organism::create_from_genome(genome, 0);

        let output = organism.activate(&vec![0.5, -0.2]);
        assert_approx_eq!(organism.phenome[NodeNumber(2)].value, 0.184);
        assert_approx_eq!(organism.phenome[NodeNumber(3)].value, 0.);

        assert_approx_eq!(output[0], 0.184);
        assert_approx_eq!(output[1], 0.);
    }

    #[test]
    fn recurrent() {
        let genome = genome_sample_recurrent_1();
        let mut organism =  Organism::create_from_genome(genome, 0);

        let inputs = vec![-0.9, 0.6];
        let mut outputs = organism.activate(&inputs);
        assert_approx_eq!(outputs[0], 0.);

        outputs = organism.activate(&inputs);
        assert_approx_eq!(outputs[0], 0.0216);

        outputs = organism.activate(&inputs);
        assert_approx_eq!(outputs[0], 0.0168);
        
    }

    fn digraph_from_organism(organism: &Organism) -> petgraph::Graph<NodeNumber, GeneNumber> {
        let mut graph = petgraph::graph::DiGraph::new();
        let distinct_nodes = organism.genome.iter().map(|(gene_key, _)| {
            vec![gene_key.in_node_id, gene_key.out_node_id]
        }).flatten().collect::<HashSet<_>>();


        for i in distinct_nodes {
            graph.add_node(i);
        }


        fn make_node_index(i: NodeNumber) -> petgraph::graph::NodeIndex {
            petgraph::graph::NodeIndex::new(i.0)
        }
        for (gene_key, _) in organism.genome.iter() {
            println!("{:?}-->{:?}", gene_key.in_node_id.0, gene_key.out_node_id.0);
            // println!("in node exists {:?}", graph(gene_key.in_node_id.0));
            // let in_index = graph.node_references()
            graph.add_edge(make_node_index(gene_key.in_node_id), make_node_index(gene_key.out_node_id), GeneNumber(0));
        }

        graph
    }

    
    fn digraph_from_genome(genome: &Genome) -> petgraph::graph::DiGraph<(), (), usize> {
        let edges = genome.iter().map(|(gene_key, _)| {
            (gene_key.in_node_id.0, gene_key.out_node_id.0)
        }).collect_vec();

        let res = petgraph::graph::DiGraph::from_edges(edges);
        res
    }

    fn my_activation_order(organism: &Organism, visited: &mut HashMap<NodeNumber, bool>, acc: &mut Vec<GeneNumber>, nodes: &Vec<NodeNumber>) -> Option<Vec<GeneNumber>> {
        let res =
            nodes.iter().map(|node_index| {
                match visited.get(node_index) {
                    Some(true) => Some(vec![]),
                    Some(false) => None,
                    None => {
                        if organism.phenome[*node_index].node_type == NodeType::Sensor {
                            visited.insert(*node_index, true);
                            Some(vec![])
                        } else {
                            if organism.phenome[*node_index].inputs.len() == 0 {
                                visited.insert(*node_index, false);
                                None
                            } else {
                                let input_nodes = 
                                    organism.phenome[*node_index].inputs.iter().map(|gene_index| {
                                        let (gene_key, _) = organism.genome.get_index(*gene_index);
                                        gene_key.in_node_id
                                    }).collect();
                                match my_activation_order(organism, visited, acc, &input_nodes) {
                                    Some(mut temp) => {
                                        // temp.extend(acc);
                                        visited.insert(*node_index, true);
                                        Some(temp)
                                    },
                                    None => {
                                        visited.insert(*node_index, false);
                                        None
                                    }
                                }
                            }
                        }
                    }
                }

            }).collect_vec();
        None
    }

    // fn petgtaph_activation_order(organism: &Organism) -> Vec<GeneIndex> {
    //     let graph = digraph_from_genome(organism);
    //     let mut dfs = petgraph::visit::DfsPostOrder::empty(&graph);
    //     let mut order = vec![];
    //     while let Some(nx) = dfs.next(&graph) {
    //         order.push(nx);
    //     }
    //     order.iter().map(|nx| graph[nx].clone()).collect()
    // }

    #[test]
    fn test_dead_ends(){
        let genome = genome_sample_dead_ends();
        let organism =  Organism::create_from_genome(genome, 0);

        for (gene_key, _) in organism.genome.iter() {
            println!("{:?}-->{:?}", gene_key.in_node_id.0, gene_key.out_node_id.0);
        }

        println!("activation order");
        for &node_index in &organism.activation_order {
            let node = &organism.phenome[node_index];
            node.inputs.iter().for_each(|gene_index| {
                let (gene_key, _) = organism.genome.get_index(*gene_index);
                println!("{:?}-->{:?}", gene_key.in_node_id.0, gene_key.out_node_id.0);
            });
        }
    }


    #[test]
    fn test_cyclic_bfs_order() {
        let genome = genome_sample_dead_ends();
        // let organism =  Organism::create_from_genome(genome, 0);
        // println!("phenome size {:?}", organism.phenome.len());
        let graph = digraph_from_genome(&genome);
        println!("genome size {:?}", genome.len());
        println!("graph size {:?}", graph.node_count());
        let tarjan_scc = petgraph::algo::tarjan_scc(&graph);

        for scc in tarjan_scc {
            println!("scc len{:?}", scc.len());
            for node in scc {
                println!("node {:?}", node);
            }
        }
    }

    #[test]
    fn test_digraph() {
        let edges = vec![
            (0, 4),
            (4, 2),
            (7, 6),
            (4, 6),
            (4, 18),
            (6, 5),
            (5, 4),
            (1, 5),
            (5, 3),
        ];

        let distinct_nodes = edges.iter().map(|(a, b)| vec![*a, *b]).flatten().collect::<HashSet<_>>();
        let mut graph = petgraph::graph::DiGraph::new();
        let mut node_map = HashMap::new();
        for &i in &distinct_nodes {
            let node = graph.add_node(i);
            node_map.insert(i, node);
        }

        for (a, b) in edges {
            graph.add_edge(node_map[&a], node_map[&b], ());
        }

        println!("nodes size {:?}", distinct_nodes.len());
        println!("graph size {:?}", graph.node_count());

        let tarjan_scc = petgraph::algo::tarjan_scc(&graph);

        for scc in tarjan_scc {
            println!("scc len{:?}", scc.len());
            for node_index in scc {
                let node = graph.node_weight(node_index).unwrap();
                println!("node {:?}", node);
            }
        }
    }
}