use std::ops::{Index, IndexMut};
use itertools::Itertools;
use rand::{seq::SliceRandom, RngCore};
use rustc_hash::FxHashSet;
use crate::neat::genome::Genome;
use super::{genome::GeneKey, phenome::Phenome};
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct OrganismIndex(pub usize);


#[derive(Clone, Serialize, Deserialize)]
pub struct Organism {
    pub phenome: Phenome,
    pub genome: Genome,
    pub fitness: usize
}

impl Organism {
    
    pub fn create_from_genome(genome: Genome) -> Organism {
        let phenome = Phenome::create_from_genome(&genome);

        Organism {
            phenome,
            genome,
            fitness: 0
        }
    }

    pub fn init<R: RngCore>(rng: &mut R, n_sensor_nodes: usize, n_output_nodes: usize) -> Organism {
        let genome = Genome::init(rng, n_sensor_nodes, n_output_nodes);
        Self::create_from_genome(genome)
    }
    pub fn activate(&mut self, sensor_values: &[f64]) -> Vec<f64> {
        debug_assert!(sensor_values.len() == self.genome.n_sensor_nodes);
        self.phenome.activate(sensor_values);
        self.phenome.outputs.iter().map(|node_index| self.phenome.nodes[*node_index].value).collect()
    }

    pub fn clear_values(&mut self) {
        self.phenome.clear_values();
    }

    // pub fn trim_genome(&mut self) {
    //     let distinct_gene_keys : FxHashSet<GeneKey> = 
    //         self.phenome.activation_order.iter().flat_map(|(out_node_index, inputs)| {
    //             let out_node_id = self.phenome.nodes[*out_node_index].id;
    //         inputs.iter().map(|(in_node_index, _)| {
    //             let in_node_id = self.phenome.nodes[*in_node_index].id;
    //             GeneKey{in_node_id, out_node_id}
    //         }).collect_vec()
    //     }).collect();

    //     self.genome.data.retain(|gene_key, _| distinct_gene_keys.contains(&gene_key));
    //     self.phenome = Phenome::create_from_genome(&self.genome);

    // }
}

#[derive(Clone, Serialize, Deserialize)]
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
    use crate::neat::{genome::{Gene, GeneExt, Genome, NodeId}, organism::Organism};
    use assert_approx_eq::assert_approx_eq;

    fn genome_sample_feed_forward_1() -> Genome{
        Genome::create(vec![
            Gene::create(0, 4, -0.1),
            Gene::create(4, 3, 0.6),
            Gene::create(1, 5, -0.8),
            Gene::create(5, 3, -0.9),
            Gene::create(0, 5, 0.6),
            Gene::create(5, 2, 0.4),
        ], 2, 2)
    }

    fn genome_sample_recurrent_1() -> Genome{
        Genome::create(vec![
            Gene::create(3, 2, 0.9),
            Gene::create(1, 4, -0.8),
            Gene::create(4, 3, 0.1),
            Gene::create(5, 2, -0.4),
            Gene::create(0, 4, -0.8),
            Gene::create(3, 5, 0.5),
            Gene::create(5, 4, -0.1),
        ], 2, 1)
    }

    #[test]
    fn organism_creation() {
        let organism =  Organism::create_from_genome(genome_sample_feed_forward_1());
        assert_eq!(organism.phenome.try_node_id(NodeId(2)).map(|n|n.inputs.len()), Some(1));
        assert_eq!(organism.phenome.try_node_id(NodeId(3)).map(|n|n.inputs.len()), Some(2));
        assert_eq!(organism.phenome.try_node_id(NodeId(4)).map(|n|n.inputs.len()), Some(1));
    }

    #[test]
    fn organsim_init(){
        let mut rng = rand::thread_rng();
        let n_sensor_nodes = 9;
        let n_output_nodes = 10;
        let n_total = n_sensor_nodes + n_output_nodes;
        let organism = Organism::init(&mut rng, n_sensor_nodes, n_output_nodes);
        assert_eq!(organism.genome.len(), 90);
        assert_eq!(organism.phenome.nodes.len(), n_total);
        assert_eq!(organism.genome.n_output_nodes, n_output_nodes);
        assert_eq!(organism.genome.n_sensor_nodes, n_sensor_nodes);

        for node_index in organism.genome.n_sensor_nodes..organism.phenome.nodes.len() {
            let input_length = organism.phenome.try_node_id(NodeId(node_index)).map(|n|n.inputs.len());
            assert_eq!(input_length, Some(n_sensor_nodes))
        }
    }

    #[test]
    fn feed_forward() {
        let genome = genome_sample_feed_forward_1();
        let mut organism =  Organism::create_from_genome(genome);

        organism.phenome.print_mermaid_graph();
        let output = organism.activate(&vec![0.5, -0.2]);
        println!("{:?}", output);
        assert_approx_eq!(output[0], 0.184);
        assert_approx_eq!(output[1], 0.);
    }

    #[test]
    fn recurrent() {
        let genome = genome_sample_recurrent_1();
        let mut organism =  Organism::create_from_genome(genome);
        organism.phenome.print_mermaid_graph();

        let inputs = vec![-0.9, 0.6];

        let outputs = organism.activate(&inputs);
        assert_approx_eq!(outputs[0], 0.0168);
        
    }
}