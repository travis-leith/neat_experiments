use std::ops::Index;
use itertools::Itertools;
use rand::RngCore;
use rand_distr::{Distribution, Uniform};

use crate::neat::vector::{AllignedPair, allign};

use super::{innovation::InnovationNumber, phenome::NodeIndex};

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct GeneIndex(pub usize);

#[derive(Clone)]
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

    pub fn tarjan_scc(&self) -> Vec<Vec<NodeIndex>> {
        use petgraph::graph::DiGraph;
        let edges = self.iter().map(|gene| (gene.in_node_id.0, gene.out_node_id.0));
        let graph: petgraph::graph::DiGraph<(), (), usize> = DiGraph::from_edges(edges);


        let scc_order = petgraph::algo::tarjan_scc(&graph);
        scc_order.iter().rev().map(|scc| {
            scc.iter().map(|node_index| {
                NodeIndex(node_index.index())
            }).collect()
        }).collect()
    }

    
}

impl Index<GeneIndex> for Genome {
    type Output = Gene;
    fn index(&self, index: GeneIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

pub fn cross_over(rng: &mut dyn RngCore, genome_1: &Genome, fitness_1: usize, genome_2: &Genome, fitness_2: usize) -> Genome {
    let between = Uniform::from(0.0..1.0);
    let mut choose_gene = |pair: AllignedPair<Gene>| {
        let r = between.sample(rng);
        match pair{
            AllignedPair::HasBoth(left, right) => {
                if r > 0.5 {
                    Some((*left).clone())
                } else {
                    Some((*right).clone())
                }
            },
            AllignedPair::HasLeft(left) => {
                if fitness_1 > fitness_2 {
                    Some((*left).clone())
                } else if fitness_1 < fitness_2 {
                    None
                } else {
                    Some((*left).clone()) //prefer left. do not randomly choose as this could result in dead end nodes if not done consistently
                }
            },
            AllignedPair::HasRight(right) => {
                if fitness_1 > fitness_2 {
                    None
                } else if fitness_1 < fitness_2 {
                    Some((*right).clone())
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
        debug_assert!(genome_1[GeneIndex(i)].innovation.0 > genome_1[GeneIndex(i-1)].innovation.0)
    }
    for i in 1 .. genome_2.len() {
        // if organism_2.network.genome[i].innovation.0 <= organism_2.network.genome[i-1].innovation.0 {
        //     println!("left {}", organism_2.network.genome[i].innovation.0);
        //     println!("right {}", organism_2.network.genome[i - 1].innovation.0);
        //     println!("found one")
        // }
        debug_assert!(genome_2[GeneIndex(i)].innovation.0 > genome_2[GeneIndex(i-1)].innovation.0)
    }

    let get_id = |gene:&Gene| gene.innovation.0;
    let new_genome_vec = 
        allign(&genome_1.0, &genome_2.0, &get_id, &mut choose_gene)
        .into_iter().flatten().collect_vec();
    let new_genome = Genome(new_genome_vec);

    // new_genome.sort_by_key(|conn|conn.innovation.0);
        
    new_genome
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
        assert_eq!(genome.calculate_max_node_id(), 5);
    }
}