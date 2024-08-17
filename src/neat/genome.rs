use std::ops::Index;
use itertools::Itertools;
use rand::RngCore;
use rand_distr::{Distribution, Uniform};
use indexmap::IndexMap;
use fxhash::FxBuildHasher;

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;

use crate::neat::vector::{AllignedTuplePair, allign_indexmap_map, allign_indexmap_iter};

use super::{innovation::InnovationNumber, phenome::NodeIndex};

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct GeneIndex(pub usize);

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct GeneKey {
    pub in_node_id: NodeIndex,
    pub out_node_id: NodeIndex
}

#[derive(Clone)]
pub struct GeneValue {
    // pub in_node_id: NodeIndex,
    // pub out_node_id: NodeIndex,
    pub weight: f64,
    pub innovation: InnovationNumber,
    pub enabled: bool
}

impl GeneValue {
    pub fn create(weight: f64, innovation: usize, enabled: bool) -> GeneValue {
        GeneValue {
            // in_node_id: NodeIndex(in_node_id),
            // out_node_id: NodeIndex(out_node_id),
            weight,
            innovation: InnovationNumber(innovation),
            enabled,
        }
    }
}

pub type Gene = (GeneKey, GeneValue);

pub trait GeneExt {
    fn create(in_node_id: usize, out_node_id: usize, weight: f64, innovation: usize, enabled: bool) -> Self;
}

impl GeneExt for Gene {
    fn create(in_node_id: usize, out_node_id: usize, weight: f64, innovation: usize, enabled: bool) -> Gene {
        (
            GeneKey {
                in_node_id: NodeIndex(in_node_id),
                out_node_id: NodeIndex(out_node_id),
            },
            GeneValue::create(weight, innovation, enabled),
        )
    }
}

pub struct Genome(FxIndexMap<GeneKey, GeneValue>);
impl Genome {
    pub fn calculate_max_node_id(&self) -> usize {
        let res = 
            self.0.keys().fold(NodeIndex(0), |acc, gene_key| {
                if gene_key.in_node_id > acc {
                    gene_key.in_node_id
                } else if gene_key.out_node_id > acc {
                    gene_key.out_node_id
                } else {
                    acc
                }
            });
        res.0
    }

    pub fn iter(&self) -> indexmap::map::Iter<GeneKey, GeneValue> {
        self.0.iter()
    }

    pub fn create(data: Vec<Gene>) -> Genome {
        
        Genome(data.into_iter().collect())
    }

    pub fn push(&mut self, gene: Gene) {
        self.0.insert(gene.0, gene.1);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get_index(&self, index: GeneIndex) -> (&GeneKey, &GeneValue) {
        self.0.get_index(index.0).unwrap()
    }

    pub fn tarjan_scc(&self) -> Vec<Vec<NodeIndex>> {
        use petgraph::graph::DiGraph;
        let edges = self.0.keys().map(|gene_key| (gene_key.in_node_id.0, gene_key.out_node_id.0));
        let graph: petgraph::graph::DiGraph<(), (), usize> = DiGraph::from_edges(edges);


        let scc_order = petgraph::algo::tarjan_scc(&graph);
        scc_order.iter().rev().map(|scc| {
            scc.iter().map(|node_index| {
                NodeIndex(node_index.index())
            }).collect()
        }).collect()
    }

    
}

pub fn cross_over(rng: &mut dyn RngCore, genome_1: &Genome, fitness_1: usize, genome_2: &Genome, fitness_2: usize) -> Genome {
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

    for i in 1 .. genome_1.len() {
        // if organism_1.network.genome[i].innovation.0 <= organism_1.network.genome[i-1].innovation.0 {
        //     println!("left {}", organism_1.network.genome[i].innovation.0);
        //     println!("right {}", organism_1.network.genome[i - 1].innovation.0);
        //     println!("found one")
        // }
        // debug_assert!(genome_1[GeneIndex(i)].innovation.0 > genome_1[GeneIndex(i-1)].innovation.0)
    }
    for i in 1 .. genome_2.len() {
        // if organism_2.network.genome[i].innovation.0 <= organism_2.network.genome[i-1].innovation.0 {
        //     println!("left {}", organism_2.network.genome[i].innovation.0);
        //     println!("right {}", organism_2.network.genome[i - 1].innovation.0);
        //     println!("found one")
        // }
        // debug_assert!(genome_2[GeneIndex(i)].innovation.0 > genome_2[GeneIndex(i-1)].innovation.0)
    }

    let get_id = |gene: (&GeneKey, &GeneValue)| gene.1.innovation;
    let new_genome_vec = 
    allign_indexmap_map(&genome_1.0, &genome_2.0, &get_id, &mut choose_gene);
    let new_genome = Genome(new_genome_vec);
    new_genome
}

pub fn genome_distance(genome_1: &Genome, genome_2: &Genome, excess_coef: f64, disjoint_coef: f64, weight_diff_coef: f64) -> f64 {
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
                disjoint_count = disjoint_count + excess_count;
                excess_count = 0;
                total_weight_diff = total_weight_diff + (left_gene_value.weight - right_gene_value.weight).abs();
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
                        disjoint_count = disjoint_count + excess_count;
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
                        disjoint_count = disjoint_count + excess_count;
                        excess_count = 1;
                    }
                }
            }
        }
    };

    let get_id = |gene: (&GeneKey, &GeneValue)| gene.1.innovation;
    allign_indexmap_iter(&genome_1.0, &genome_2.0, &get_id, &mut increment_counters);

    let n = std::cmp::max(n1, n2) as f64;
    let excess_term = excess_coef * (excess_count as f64) / n;
    let disjoint_term = disjoint_coef * (disjoint_count as f64) / n;
    let weight_term = weight_diff_coef * total_weight_diff / n;
    excess_term + disjoint_term + weight_term
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(genome.calculate_max_node_id(), 4);
    }
}