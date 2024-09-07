use rand::{RngCore, Rng};
use rand_distr::{Distribution, Normal, Uniform};
use indexmap::IndexMap;
use rustc_hash::FxBuildHasher;

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;

use crate::neat::vector::{AllignedTuplePair, allign_indexmap_map, allign_indexmap_iter};

use super::{common::Settings, innovation::{InnovationContext, InnovationNumber}};

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct GeneIndex(pub usize);

#[derive(PartialEq, PartialOrd, Clone, Copy, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn inc(self) -> NodeId {
        NodeId(self.0 + 1)
    }
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct GeneKey {
    pub in_node_id: NodeId,
    pub out_node_id: NodeId
}

#[derive(Clone)]
pub struct GeneValue {
    pub weight: f64,
    pub innovation: InnovationNumber,
    pub enabled: bool
}

impl GeneValue {
    pub fn create(weight: f64, innovation: usize, enabled: bool) -> GeneValue {
        GeneValue {
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
                in_node_id: NodeId(in_node_id),
                out_node_id: NodeId(out_node_id),
            },
            GeneValue::create(weight, innovation, enabled),
        )
    }
}

#[derive(Clone)]
pub struct Genome{
    data: FxIndexMap<GeneKey, GeneValue>,
    pub next_node_id: NodeId,
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
        let data = genes.into_iter().collect();
        let next_node_id = max_node_id.inc();
        Genome{data, next_node_id, n_sensor_nodes, n_output_nodes}
    }

    pub fn init<R: RngCore>(rng: &mut R, n_sensor_nodes: usize, n_output_nodes: usize) -> Genome {
        let between = Uniform::from(-1.0..1.0);

        let n_connections = n_sensor_nodes * n_output_nodes;
        let mut data = IndexMap::with_capacity_and_hasher(n_connections, FxBuildHasher::default());

        for out_node_ind in 0..n_output_nodes {
            let out_node_id = out_node_ind + n_sensor_nodes;
            for in_node_ind in 0..n_sensor_nodes {
                let in_node_id = in_node_ind;
                let innovation_number = out_node_ind * n_sensor_nodes + in_node_ind;
                let (gene_key, gene_val) = Gene::create(in_node_id, out_node_id, between.sample(rng), innovation_number, true);
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

    // pub fn rev_dfs_order_petgraph(&self) -> Vec<NodeIndex> {
    //     //TODO implement a version of this that ignores dead end nodes
    //     use petgraph::graph::DiGraph;
    //     let new_node_id = self.next_node_id.0;
    //     let start_edges = 
    //         (0..self.n_sensor_nodes).map(|i| (new_node_id, i));

    //     let edges = 
    //         self.data.iter()
    //         .filter(|(_, gene)| gene.enabled)
    //         .map(|(gene_key, _)| (gene_key.in_node_id.0, gene_key.out_node_id.0));

    //     let all_edges = edges.chain(start_edges);
        
    //     let graph: petgraph::graph::DiGraph<(), (), usize> = DiGraph::from_edges(all_edges);

    //     let dfs = petgraph::visit::DfsPostOrder::new(&graph, new_node_id.into());

    //     dfs.iter(&graph)
    //     .map(|node| NodeIndex(node.index()))
    //     .take_while(|node| node.0 != new_node_id)
    //     .collect_vec()
    //     .into_iter()
    //     .rev()
    //     .collect()
    // }

    pub fn add_connection(&mut self, innovation_context: &mut InnovationContext, in_node_id: NodeId, out_node_id: NodeId, weight: f64) {
        debug_assert!(in_node_id != out_node_id, "Tried to add a connection where input is the same node as output");
        debug_assert!(in_node_id < self.next_node_id, "Tried to add a connection with an input node that does not exist");
        debug_assert!(out_node_id < self.next_node_id, "Tried to add a connection with an output node that does not exist");
        debug_assert!(out_node_id.0 >= self.n_sensor_nodes, "Tried to add a connection with an output node that is a sensor node");

        let gene_key = GeneKey {
            in_node_id,
            out_node_id
        };

        let innov_number = innovation_context.get_innovation_number(gene_key.clone());
        let gene_value = GeneValue::create(weight, innov_number.0, true);
        let insertion_result = self.data.insert(gene_key, gene_value);
        debug_assert!(insertion_result.is_none(), "Tried to add a connection that already exists");
    }

    pub fn add_node(&mut self, innovation_context: &mut InnovationContext, existing_conn_index: GeneIndex) {
        let (gene_key, gene_val) = {
            let (gene_key, gene_val) = self.get_index_mut(existing_conn_index);
            let cloned_pair = (gene_key.clone(), gene_val.clone());
            gene_val.enabled = false;
            cloned_pair
        };
    
        if gene_val.enabled {
            let new_node_id = self.next_node_id;
            self.next_node_id = self.next_node_id.inc();
            self.add_connection(innovation_context, gene_key.in_node_id, new_node_id, 1.);
            self.add_connection(innovation_context, new_node_id, gene_key.out_node_id, gene_val.weight);
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
        allign_indexmap_iter(&self.data, &other.data, &get_id, &mut increment_counters);

        let n = std::cmp::max(n1, n2) as f64;
        let excess_term = excess_coef * (excess_count as f64) / n;
        let disjoint_term = disjoint_coef * (disjoint_count as f64) / n;
        let weight_term = weight_diff_coef * total_weight_diff / n;
        excess_term + disjoint_term + weight_term
    }

    pub fn mutate<R: RngCore>(&mut self, rng: &mut R, innovation_context: &mut InnovationContext, settings: &Settings) {
        let between = Uniform::from(0.0..1.0);
        self.mutate_add_connection(rng, &between, innovation_context, settings);
        self.mutate_add_node(rng, &between, innovation_context, settings);
        self.mutate_weight(rng, &between, settings);
        //TODO: enable/disable genes
    }

    fn mutate_add_connection<R: RngCore>(&mut self, rng: &mut R, between:&Uniform<f64>, innovation_context: &mut InnovationContext, settings: &Settings) {
        let r = between.sample(rng);
        if r < settings.mutate_add_connection_rate {
            let in_node_id = NodeId(rng.gen_range(0..self.next_node_id.0));
            let out_node_id = NodeId(rng.gen_range(self.n_sensor_nodes..self.next_node_id.0));
            if in_node_id != out_node_id {
                let gene_key = GeneKey{in_node_id, out_node_id};
                if !self.data.contains_key(&gene_key) {
                    self.add_connection(innovation_context, in_node_id, out_node_id, rng.gen_range(-1.0..1.0));
                }
            }
        }
    }

    fn mutate_add_node<R: RngCore>(&mut self, rng: &mut R, between:&Uniform<f64>, innovation_context: &mut InnovationContext, settings: &Settings) {
        let r = between.sample(rng);
        if r < settings.mutate_add_node_rate {
            let gene_index = GeneIndex(rng.gen_range(0..self.len()));
            self.add_node(innovation_context, gene_index);
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

    let get_id = |gene: (&GeneKey, &GeneValue)| gene.1.innovation;
    let new_genome_data = allign_indexmap_map(&genome_1.data, &genome_2.data, &get_id, &mut choose_gene);
    let new_next_node_id = NodeId(std::cmp::max(genome_1.next_node_id.0, genome_2.next_node_id.0));
    let new_genome = Genome{data: new_genome_data, next_node_id: new_next_node_id, n_sensor_nodes:genome_1.n_sensor_nodes, n_output_nodes:genome_1.n_output_nodes};
    new_genome
}


#[cfg(test)]
mod tests {
    // use fxhash::FxHashSet;
    // use rand::SeedableRng;
    // use rand_xoshiro::Xoshiro256PlusPlus;

    use super::*;
    fn genome_sample_1() -> Genome{
        Genome::create(vec![
            Gene::create(0, 3, 0.0, 0, true),
            Gene::create(1, 3, 0.0, 1, true),
            Gene::create(1, 4, 0.0, 2, true),
            Gene::create(2, 4, 0.0, 3, true),
            Gene::create(3, 4, 0.0, 4, true),
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
        let mut innovation_context = InnovationContext::init(2, 2);
        genome.add_connection(&mut innovation_context, NodeId(0), NodeId(4), 0.0);
        assert_eq!(genome.len(), 6);
    }

    #[test]
    fn test_genome_add_node() {
        let mut genome = genome_sample_1();
        let mut innovation_context = InnovationContext::init(2, 2);
        genome.add_node(&mut innovation_context, GeneIndex(0));
        assert_eq!(genome.len(), 7);
    }

    // fn generate_random_genes(n: usize, m: usize, random_seed: u64) -> Vec<Gene> {
    //     let mut rng = Xoshiro256PlusPlus::seed_from_u64(random_seed);
    //     let mut seen = FxHashSet::default();
    //     let mut iters = 0;
    //     let max_iterations = n * n;

    //     while seen.len() < n && iters < max_iterations {
    //         let input = rng.gen_range(0..m);
    //         let output = rng.gen_range(0..m);
    //         let pair = (input, output);
    //         let reverse_pair = (output, input);
    
    //         if !seen.contains(&pair) && !seen.contains(&reverse_pair) && input != output {
    //             seen.insert(pair);
    //         }
    //         iters += 1;
    //     }
    
    //     seen.iter().enumerate().map(|(i, gene)| {
    //         let weight = rng.gen_range(-1.0..1.0);
    //         let innovation = i;
    //         let enabled = true;
    //         Gene::create(gene.0, gene.1, weight, innovation, enabled)
    //     }).collect()
    // }

    // extern crate test;
    // #[bench]
    // fn bench_tarjan_scc(b: &mut test::Bencher) {
    //     let genes = generate_random_genes(1000, 30, 123);
    //     let genome = Genome::create(genes, 10, 10);
    //     b.iter(|| {
    //         genome.tarjan_scc();
    //     });
    // }

    // #[test]
    // fn test_dfs_order2(){
    //     let genome = genome_sample_recurrent_1();

    //     for (gene_key, gene_value) in genome.data.iter() {
    //         println!("{:?}---|{:.4}|{:?}", gene_key.in_node_id.0, gene_value.weight, gene_key.out_node_id.0);
    //     }

    //     println!("rev dfs order");
    //     let dfs_order = genome.rev_dfs_order_petgraph();

    //     for node_index in dfs_order {
    //         println!("{:?}", node_index.0);
    //     }

    // }

}