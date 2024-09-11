use itertools::Itertools;
use rand::{RngCore, Rng};
use rand_distr::{Distribution, Normal, Uniform};
use indexmap::IndexMap;
use rustc_hash::FxBuildHasher;

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;

use crate::neat::vector::{AllignedTuplePair, allign_indexmap_map, allign_indexmap_iter};

use super::common::Settings;

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct GeneIndex(pub usize);

#[derive(PartialEq, PartialOrd, Ord, Clone, Copy, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn inc(self) -> NodeId {
        NodeId(self.0 + 1)
    }
}

#[derive(Hash, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub struct GeneKey {
    pub in_node_id: NodeId,
    pub out_node_id: NodeId
}

#[derive(Clone)]
pub struct GeneValue {
    pub weight: f64,
    pub enabled: bool
}

impl GeneValue {
    pub fn create(weight: f64, enabled: bool) -> GeneValue {
        GeneValue {
            weight,
            enabled,
        }
    }
}

pub type Gene = (GeneKey, GeneValue);

pub trait GeneExt {
    fn create(in_node_id: usize, out_node_id: usize, weight: f64, enabled: bool) -> Self;
}

impl GeneExt for Gene {
    fn create(in_node_id: usize, out_node_id: usize, weight: f64, enabled: bool) -> Gene {
        (
            GeneKey {
                in_node_id: NodeId(in_node_id),
                out_node_id: NodeId(out_node_id),
            },
            GeneValue::create(weight, enabled),
        )
    }
}

#[derive(Clone)]
pub struct Genome{
    data: FxIndexMap<GeneKey, GeneValue>,
    next_node_id: NodeId,
    pub n_sensor_nodes: usize,
    pub n_output_nodes: usize,
}

impl Genome {
    pub fn iter(&self) -> indexmap::map::Iter<GeneKey, GeneValue> {
        self.data.iter()
    }

    pub fn create(genes: Vec<Gene>, n_sensor_nodes: usize, n_output_nodes: usize) -> Genome {
        let max_node_id = genes.iter().fold(NodeId(0), |acc, (gene_key, _)| {
            if gene_key.in_node_id > acc {
                gene_key.in_node_id
            } else if gene_key.out_node_id > acc {
                gene_key.out_node_id
            } else {
                acc
            }
        });
        let data = genes.into_iter().sorted_by_key(|x|x.0.clone()).collect();
        let next_node_id = max_node_id.inc();
        Genome{data, next_node_id, n_sensor_nodes, n_output_nodes}
    }

    pub fn init<R: RngCore>(rng: &mut R, n_sensor_nodes: usize, n_output_nodes: usize) -> Genome {
        let between = Uniform::from(-1.0..1.0);

        let n_connections = n_sensor_nodes * n_output_nodes;
        let mut data = IndexMap::with_capacity_and_hasher(n_connections, FxBuildHasher);

        for out_node_ind in 0..n_output_nodes {
            let out_node_id = out_node_ind + n_sensor_nodes;
            for in_node_ind in 0..n_sensor_nodes {
                let in_node_id = in_node_ind;
                let (gene_key, gene_val) = Gene::create(in_node_id, out_node_id, between.sample(rng), true);
                data.insert(gene_key, gene_val);
            }
        }

        let next_node_id = NodeId(n_sensor_nodes + n_output_nodes);

        Genome{data, next_node_id, n_sensor_nodes, n_output_nodes}
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get_index(&self, index: GeneIndex) -> (&GeneKey, &GeneValue) {
        self.data.get_index(index.0).unwrap()
    }

    fn get_index_mut(&mut self, index: GeneIndex) -> (&GeneKey, &mut GeneValue) {
        self.data.get_index_mut(index.0).unwrap()
    }

    pub fn add_connection(&mut self, in_node_id: NodeId, out_node_id: NodeId, weight: f64) {
        debug_assert!(in_node_id != out_node_id, "Tried to add a connection where input is the same node as output");
        debug_assert!(in_node_id < self.next_node_id, "Tried to add a connection with an input node that does not exist");
        debug_assert!(out_node_id < self.next_node_id, "Tried to add a connection with an output node that does not exist");
        debug_assert!(out_node_id.0 >= self.n_sensor_nodes, "Tried to add a connection with an output node that is a sensor node");

        let gene_key = GeneKey {
            in_node_id,
            out_node_id
        };

        let gene_value = GeneValue::create(weight, true);
        let insertion_result = self.data.insert(gene_key, gene_value);
        debug_assert!(insertion_result.is_none(), "Tried to add a connection that already exists");
    }

    pub fn add_node(&mut self, existing_conn_index: GeneIndex) {
        let (gene_key, gene_val) = {
            let (gene_key, gene_val) = self.get_index_mut(existing_conn_index);
            let cloned_pair = (gene_key.clone(), gene_val.clone());
            gene_val.enabled = false;
            cloned_pair
        };
    
        if gene_val.enabled {
            let new_node_id = self.next_node_id;
            self.next_node_id = self.next_node_id.inc();
            self.add_connection(gene_key.in_node_id, new_node_id, 1.);
            self.add_connection(new_node_id, gene_key.out_node_id, gene_val.weight);
        }    
    }

    pub fn distance(&self, other: &Genome, excess_coef: f64, disjoint_coef: f64, weight_diff_coef: f64) -> f64 {
        #[derive(PartialEq, PartialOrd)]
        enum ExcessSide {
            Left,
            Right,
            Neither
        }

        let mut total_weight_diff = 0.;
        let mut excess_side = ExcessSide::Neither;
        let mut excess_count = 0;
        let mut disjoint_count = 0;
        let mut n1 = 0;
        let mut n2 = 0;

        let mut increment_counters =  |pair: AllignedTuplePair<GeneKey, GeneValue>| {
            match pair {
                AllignedTuplePair::HasBoth(left, right) => {
                    let left_gene_value = left.1;
                    let right_gene_value = right.1;
                    n1 += 1;
                    n2 += 1;
                    excess_side = ExcessSide::Neither;
                    disjoint_count += excess_count;
                    excess_count = 0;
                    total_weight_diff += (left_gene_value.weight - right_gene_value.weight).abs();
                },
                AllignedTuplePair::HasLeft(_) => {
                    n1 +=1;
                    match excess_side {
                        ExcessSide::Neither => {
                            excess_side = ExcessSide::Left;
                            excess_count = 1;
                        },
                        ExcessSide::Right => {
                            excess_side = ExcessSide::Left;
                            disjoint_count += excess_count;
                            excess_count = 1;
                        },
                        ExcessSide::Left => {
                            excess_count += 1;
                        }
                    }
                },
                AllignedTuplePair::HasRight(_) => {
                    n2 += 1;
                    match excess_side {
                        ExcessSide::Neither => {
                            excess_side = ExcessSide::Right;
                            excess_count = 1;
                        },
                        ExcessSide::Right => {
                            excess_count += 1;
                        },
                        ExcessSide::Left => {
                            excess_side = ExcessSide::Right;
                            disjoint_count += excess_count;
                            excess_count = 1;
                        }
                    }
                }
            }
        };

        let get_id = |gene: (&GeneKey, &GeneValue)| gene.0.clone();
        allign_indexmap_iter(&self.data, &other.data, &get_id, &mut increment_counters);

        let n = std::cmp::max(n1, n2) as f64;
        let excess_term = excess_coef * (excess_count as f64) / n;
        let disjoint_term = disjoint_coef * (disjoint_count as f64) / n;
        let weight_term = weight_diff_coef * total_weight_diff / n;
        excess_term + disjoint_term + weight_term
    }

    pub fn mutate<R: RngCore>(&mut self, rng: &mut R, settings: &Settings) {
        let between = Uniform::from(0.0..1.0);
        self.mutate_add_connection(rng, &between, settings);
        self.mutate_add_node(rng, &between, settings);
        self.mutate_weight(rng, &between, settings);
        self.mutate_toggle_connection(rng, &between, settings);
    }

    fn mutate_add_connection<R: RngCore>(&mut self, rng: &mut R, between:&Uniform<f64>, settings: &Settings) {
        let r = between.sample(rng);
        if r < settings.mutate_add_connection_rate {
            let in_node_id = NodeId(rng.gen_range(0..self.next_node_id.0));
            let out_node_id = NodeId(rng.gen_range(self.n_sensor_nodes..self.next_node_id.0));
            if in_node_id != out_node_id {
                let gene_key = GeneKey{in_node_id, out_node_id};
                if !self.data.contains_key(&gene_key) {
                    self.add_connection(in_node_id, out_node_id, rng.gen_range(-1.0..1.0));
                }
            }
        }
    }

    fn mutate_add_node<R: RngCore>(&mut self, rng: &mut R, between:&Uniform<f64>, settings: &Settings) {
        let r = between.sample(rng);
        if r < settings.mutate_add_node_rate {
            let gene_index = GeneIndex(rng.gen_range(0..self.len()));
            self.add_node(gene_index);
        }
    }

    fn mutate_weight<R: RngCore>(&mut self, rng: &mut R, between:&Uniform<f64>, settings: &Settings) {
        let normal = Normal::new(0., settings.mutate_weight_scale).unwrap();
        for (_, gene_value) in self.data.iter_mut() {
            let r = between.sample(rng);
            if r < settings.mutate_weight_rate {
                //generate guassian random number
                gene_value.weight += normal.sample(rng);
            }
        }
    }

    fn mutate_toggle_connection<R: RngCore>(&mut self, rng: &mut R, between:&Uniform<f64>, settings: &Settings) {
        let r = between.sample(rng);
        if r < settings.mutate_toggle_connection_rate {
            let gene_index = GeneIndex(rng.gen_range(0..self.len()));
            let (_, gene_value) = self.get_index_mut(gene_index);
            gene_value.enabled = !gene_value.enabled;
        }
    }
}



pub fn cross_over<R: RngCore>(rng: &mut R, genome_1: &Genome, fitness_1: usize, genome_2: &Genome, fitness_2: usize) -> Genome {
    let between = Uniform::from(0.0..1.0);
    let mut choose_gene = |pair: AllignedTuplePair<GeneKey, GeneValue>| {
        let r = between.sample(rng);
        fn clone_gene(gene: (&GeneKey, &GeneValue)) -> Gene {
            (gene.0.clone(), gene.1.clone())
        }
        match pair{
            AllignedTuplePair::HasBoth(left, right) => {
                if r > 0.5 {
                    Some(clone_gene(left))
                } else {
                    Some(clone_gene(right))
                }
            },
            AllignedTuplePair::HasLeft(left) => {
                if fitness_1 > fitness_2 {
                    Some(clone_gene(left))
                } else if fitness_1 < fitness_2 {
                    None
                } else {
                    Some(clone_gene(left)) //prefer left. do not randomly choose as this could result in dead end nodes if not done consistently
                }
            },
            AllignedTuplePair::HasRight(right) => {
                if fitness_1 > fitness_2 {
                    None
                } else if fitness_1 < fitness_2 {
                    Some(clone_gene(right))
                } else {
                    None //prefer left. do not randomly choose as this could result in dead end nodes if not done consistently
                }
            }
        }
    };

    let get_id = |gene: (&GeneKey, &GeneValue)| gene.0.clone();
    let new_genome_data = allign_indexmap_map(&genome_1.data, &genome_2.data, &get_id, &mut choose_gene);
    let new_next_node_id = NodeId(std::cmp::max(genome_1.next_node_id.0, genome_2.next_node_id.0));
    Genome{data: new_genome_data, next_node_id: new_next_node_id, n_sensor_nodes:genome_1.n_sensor_nodes, n_output_nodes:genome_1.n_output_nodes}
}


#[cfg(test)]
mod tests {
    use super::*;
    fn genome_sample_1() -> Genome{
        Genome::create(vec![
            Gene::create(0, 3, 0.0, true),
            Gene::create(1, 3, 0.0, true),
            Gene::create(1, 4, 0.0, true),
            Gene::create(2, 4, 0.0, true),
            Gene::create(3, 4, 0.0, true),
        ], 2, 2)
    } 
    
    #[test]
    fn test_genome_max_node_id() {
        let genome = genome_sample_1();
        assert_eq!(genome.next_node_id.0, 5);
    }

    #[test]
    fn test_genome_init() {
        let mut rng = rand::thread_rng();
        let genome = Genome::init(&mut rng, 2, 2);
        assert_eq!(genome.next_node_id.0, 4);
        assert_eq!(genome.len(), 4);
    }

    #[test]
    fn test_genome_add_connection() {
        let mut genome = genome_sample_1();
        genome.add_connection(NodeId(0), NodeId(4), 0.0);
        assert_eq!(genome.len(), 6);
    }

    #[test]
    fn test_genome_add_node() {
        let mut genome = genome_sample_1();
        genome.add_node(GeneIndex(0));
        assert_eq!(genome.len(), 7);
    }

}