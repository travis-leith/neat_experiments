use std::ops::{Index, IndexMut};

use itertools::Itertools;
use petgraph::{graph::NodeIndex, visit::{EdgeRef, Walker}, Directed, Direction, Graph};
use rand::{seq::SliceRandom, RngCore};
use rustc_hash::FxHashSet;

use crate::neat::genome::Genome;

use super::genome::{GeneNumber, GeneValue};

#[derive(Clone, Copy)]
pub struct OrganismIndex(pub usize);

#[derive(PartialEq, Default, Clone, Debug)]
pub enum NodeType{
    Sensor,
    #[default]
    Hidden,
    Output,
}

#[derive(Clone)]
pub struct Node {
    pub id: usize,
    pub value: f64,
    pub node_type: NodeType,
}

#[derive(Clone)]
pub struct Organism {
    pub phenome: Graph<Node, f64, Directed>,
    pub sensor_nodes: Vec<NodeIndex>,
    pub output_nodes: Vec<NodeIndex>,
    pub genome: Genome,
    pub activation_order: Vec<NodeIndex>,
    pub fitness: usize
}

impl Organism {
    pub fn print_mermaid_graph(&self) {
        println!("graph TD");
        for &node_index in &self.activation_order {
            let node = &self.phenome[node_index];
            let node_id = node.id;
            let node_type = match node.node_type {
                NodeType::Sensor => "S",
                NodeType::Hidden => "H",
                NodeType::Output => "O",
            };
            println!("{}[{}:{}]", node_id, node_id, node_type);
        }

        for &node_index in &self.activation_order {
            let node = &self.phenome[node_index];
            if node.node_type != NodeType::Sensor {
                let incoming_edges = self.phenome.edges_directed(node_index, Direction::Incoming);
                for edge in incoming_edges {
                    let in_node_id = edge.source();
                    let in_node = &self.phenome[in_node_id];
                    let out_node_id = edge.target();
                    let out_node = &self.phenome[out_node_id];
                    println!("{} -->|{:.4}|{}", in_node.id, edge.weight(), out_node.id);
                }
            }
        }
    }

    pub fn create_from_genome(genome: Genome, initial_fitness: usize) -> Organism {
        use rustc_hash::FxHashMap;
        //TODO remove connections involving dead end nodes
        //TODO create phenome::create_from_genome

        let mut phenome = Graph::<Node, f64, Directed>::new();
        let mut node_map: FxHashMap<usize, NodeIndex> = FxHashMap::default();//TODO init with capacity

        fn get_node_index(node_map: &mut FxHashMap<usize, NodeIndex>, phenome: &mut Graph<Node, f64, Directed>, n_sensor_nodes: usize, n_output_nodes: usize, node_id: usize) -> NodeIndex {
            match node_map.get(&node_id) {
                Some(i) => *i,
                None => {
                    let node_type = 
                        if node_id < n_sensor_nodes {
                            NodeType::Sensor
                        } else if node_id < n_sensor_nodes + n_output_nodes {
                            NodeType::Output
                        } else {
                            NodeType::Hidden
                        };
                    let node = Node{id: node_id, value: 0., node_type};
                    let &node_index = &phenome.add_node(node);
                    node_map.insert(node_id, node_index);
                    node_index
                }
            }
        }

        for (gene_key, gene_val) in genome.iter() {
            if gene_val.enabled {
                let in_node_index = get_node_index(&mut node_map, &mut phenome, genome.n_sensor_nodes, genome.n_output_nodes, gene_key.in_node_id);
                let out_node_index = get_node_index(&mut node_map, &mut phenome, genome.n_sensor_nodes, genome.n_output_nodes, gene_key.out_node_id);
                phenome.add_edge(in_node_index, out_node_index, gene_val.weight);
            }
        }

        let dangling_node = get_node_index(&mut node_map, &mut phenome, genome.n_sensor_nodes, genome.n_output_nodes, genome.next_node_id);
        for i in 0..genome.n_sensor_nodes {
            let node_index = get_node_index(&mut node_map, &mut phenome, genome.n_sensor_nodes, genome.n_output_nodes, i);
            phenome.add_edge(dangling_node, node_index, 0.0);
        }

        let dfs = petgraph::visit::DfsPostOrder::new(&phenome, dangling_node);
        let mut order_forward_set:FxHashSet<_> = dfs.iter(&phenome).collect();
        order_forward_set.remove(&dangling_node);
        // order_forward.pop(); //remove dangling node

        // for &node_index in &order_forward {
        //     let node = &phenome[node_index];
        //     println!("order_forward node {:?}", node.id);
        // }

        // phenome.remove_node(dangling_node); //assuming this is safe to do as long as it is the last node

        let dangling_node2 = get_node_index(&mut node_map, &mut phenome, genome.n_sensor_nodes, genome.n_output_nodes, genome.next_node_id + 1);
        for i in genome.n_sensor_nodes..genome.n_sensor_nodes + genome.n_output_nodes {
            let node_index = get_node_index(&mut node_map, &mut phenome, genome.n_sensor_nodes, genome.n_output_nodes, i);
            phenome.add_edge(node_index, dangling_node2, 0.0);
        }

        phenome.reverse();
        let dfs = petgraph::visit::DfsPostOrder::new(&phenome, dangling_node2);
        let mut order_backward = dfs.iter(&phenome).collect_vec();
        order_backward.pop(); //remove dangling node

        // for &node_index in &order_backward {
        //     let node = &phenome[node_index];
        //     println!("order_backward node {:?}", node.id);
        // }

        // phenome.remove_node(dangling_node2); //assuming this is safe to do as long as it is the last node

        // let forward_set: std::collections::HashSet<_> = order_forward.into_iter().collect();
        let backward_intersection: Vec<_> = order_backward.into_iter().filter(|&x| order_forward_set.contains(&x)).collect();

        // for &node_index in &backward_intersection {
        //     let node = &phenome[node_index];
        //     println!("backward_intersection node {:?}", node.id);
        // }

        let sensor_nodes = 
            (0..genome.n_sensor_nodes)
            .map(|i| get_node_index(&mut node_map, &mut phenome, genome.n_sensor_nodes, genome.n_output_nodes, i))
            .collect_vec();

        let output_nodes =
            (genome.n_sensor_nodes..genome.n_sensor_nodes + genome.n_output_nodes)
            .map(|i| get_node_index(&mut node_map, &mut phenome, genome.n_sensor_nodes, genome.n_output_nodes, i))
            .collect_vec();

        phenome.reverse(); //TODO start with reveersed graph so that only 1 reversal is required

        Organism {
            phenome,
            genome,
            sensor_nodes,
            output_nodes,
            activation_order: backward_intersection,
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
            let node_index = self.sensor_nodes[i];
            self.phenome[node_index].value = input;
        }

        for &node_index in &self.activation_order {
            let node = &self.phenome[node_index];
            if node.node_type != NodeType::Sensor { //TODO remove sensor nodes from activation ordeer
                let incoming_edges = self.phenome.edges_directed(node_index, Direction::Incoming);
                // debug_assert!(node.node_type != NodeType::Sensor);
                

                let active_sum = incoming_edges.fold(0., |acc, edge| {
                    acc + *edge.weight() * self.phenome[edge.source()].value
                });

                let incoming_edges = self.phenome.edges_directed(node_index, Direction::Incoming);

                let weighted_inputs = incoming_edges.map(|edge| {
                    *edge.weight() * self.phenome[edge.source()].value
                }).collect_vec();

                self.phenome[node_index].value = relu(active_sum);
            }
        }

        let outputs = self.output_nodes.iter().map(|&node_index| self.phenome[node_index].value).collect();
        outputs
    }

    pub fn clear_values(&mut self) {
        for &node_index in &self.activation_order {
            self.phenome[node_index].value = 0.;
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

    use crate::neat::{genome::{Gene, GeneExt, GeneKey, GeneNumber, GeneValue, Genome}, innovation::InnovationNumber, organism::{NodeType, Organism}};
    use assert_approx_eq::assert_approx_eq;
    use itertools::Itertools;
    use petgraph::{data::Build, visit::{EdgeRef, IntoNodeIdentifiers, IntoNodeReferences, NodeCount, NodeRef}, Directed};

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
            Gene::create(4, 8, 0.0, 4, true),
            Gene::create(6, 5, 0.0, 5, true),
            Gene::create(5, 4, 0.0, 6, true),
            Gene::create(1, 5, 0.0, 7, true),
            Gene::create(5, 3, 0.0, 8, true),
        ], 2, 2)
    }

    #[test]
    fn network_creation() {
        let network =  Organism::create_from_genome(genome_sample_feed_forward_1(), 0);
        assert_eq!(network.phenome.node_count(), 6);
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
        assert_eq!(network.phenome.node_count(), n_total);
        assert_eq!(network.genome.n_output_nodes, n_output_nodes);
        assert_eq!(network.genome.n_sensor_nodes, n_sensor_nodes);
    }

    #[test]
    fn feed_forward() {
        let genome = genome_sample_feed_forward_1();
        let mut organism =  Organism::create_from_genome(genome, 0);
        organism.print_mermaid_graph();
        let output = organism.activate(&vec![0.5, -0.2]);

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

    
    #[test]
    fn test_cyclic_bfs_order() {
        let genome = genome_sample_dead_ends();
        println!("genome size {:?}", genome.len());
        let organism =  Organism::create_from_genome(genome, 0);
        // println!("phenome size {:?}", organism.phenome.len());
        
        println!("graph size {:?}", organism.phenome.node_count());
        let tarjan_scc = petgraph::algo::tarjan_scc(&organism.phenome);

        for scc in tarjan_scc {
            println!("scc len{:?}", scc.len());
            for node_index in scc {
                let node = &organism.phenome[node_index];
                println!("node {:?}", node.id);
            }
        }
    }

 
}