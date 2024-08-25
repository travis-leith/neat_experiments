use itertools::Itertools;
use rand::{RngCore, Rng};
use rand_distr::{Distribution, Normal, Uniform};
use indexmap::IndexMap;
use fxhash::FxBuildHasher;

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;

use crate::neat::vector::{AllignedTuplePair, allign_indexmap_map, allign_indexmap_iter};

use super::{common::Settings, innovation::{InnovationContext, InnovationNumber}, phenome::NodeIndex};

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct GeneIndex(pub usize);

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct GeneKey {
    pub in_node_id: NodeIndex,
    pub out_node_id: NodeIndex
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
                in_node_id: NodeIndex(in_node_id),
                out_node_id: NodeIndex(out_node_id),
            },
            GeneValue::create(weight, innovation, enabled),
        )
    }
}

#[derive(Clone)]
pub struct Genome{
    data: FxIndexMap<GeneKey, GeneValue>,
    pub next_node_id: NodeIndex,
    pub n_sensor_nodes: usize,
    pub n_output_nodes: usize,
}

impl Genome {
    pub fn iter(&self) -> indexmap::map::Iter<GeneKey, GeneValue> {
        self.data.iter()
    }

    pub fn create(genes: Vec<Gene>, n_sensor_nodes: usize, n_output_nodes: usize) -> Genome {
        let max_node_id = genes.iter().fold(NodeIndex(0), |acc, (gene_key, _)| {
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

        let next_node_id = NodeIndex(n_sensor_nodes + n_output_nodes);

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

    pub fn tarjan_scc_(&self) -> Vec<Vec<NodeIndex>> {
        use petgraph::graph::DiGraph;
        let edges = 
            self.data.iter().filter(|(_, gene)| gene.enabled).map(|(gene_key, _)| (gene_key.in_node_id.0, gene_key.out_node_id.0)).collect_vec();
        let graph: petgraph::graph::DiGraph<(), (), usize> = DiGraph::from_edges(edges);


        let scc_order = petgraph::algo::tarjan_scc(&graph);
        let mut res: Vec<Vec<NodeIndex>> =
            scc_order.iter().rev().map(|scc| {
                scc.iter().map(|node_index| {
                    NodeIndex(node_index.index())
                }).collect()
            }).collect();
        res.retain(|scc| scc.len() > 0);
        res
    }

    pub fn tarjan_scc(&self) -> Vec<Vec<NodeIndex>> {
        // Define necessary structures and types
        use rustc_hash::FxHashMap;

        // Initialize variables
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices = FxHashMap::default();
        let mut low_links = FxHashMap::default();
        let mut on_stack = FxHashMap::default();
        let mut sccs = Vec::new();

        // Define the recursive function `strong_connect`
        fn strong_connect(
            node: NodeIndex,
            index: &mut usize,
            stack: &mut Vec<NodeIndex>,
            indices: &mut FxHashMap<NodeIndex, usize>,
            low_links: &mut FxHashMap<NodeIndex, usize>,
            on_stack: &mut FxHashMap<NodeIndex, bool>,
            sccs: &mut Vec<Vec<NodeIndex>>,
            graph: &FxHashMap<NodeIndex, Vec<NodeIndex>>,
        ) {
            // Set the depth index for `node` to the smallest unused index
            indices.insert(node, *index);
            low_links.insert(node, *index);
            *index += 1;
            stack.push(node);
            on_stack.insert(node, true);

            // Consider successors of `node`
            if let Some(successors) = graph.get(&node) {
                for &successor in successors {
                    if !indices.contains_key(&successor) {
                        // Successor has not yet been visited; recurse on it
                        strong_connect(successor, index, stack, indices, low_links, on_stack, sccs, graph);
                        let low_link_node = low_links.get(&node).unwrap().clone();
                        let low_link_successor = low_links.get(&successor).unwrap().clone();
                        low_links.insert(node, low_link_node.min(low_link_successor));
                    } else if *on_stack.get(&successor).unwrap() {
                        // Successor is in stack and hence in the current SCC
                        let low_link_node = low_links.get(&node).unwrap().clone();
                        let index_successor = indices.get(&successor).unwrap().clone();
                        low_links.insert(node, low_link_node.min(index_successor));
                    }
                }
            }

            // If `node` is a root node, pop the stack and generate an SCC
            if indices.get(&node) == low_links.get(&node) {
                let mut scc = Vec::new();
                loop {
                    let w = stack.pop().unwrap();
                    on_stack.insert(w, false);
                    scc.push(w);
                    if w == node {
                        break;
                    }
                }
                sccs.push(scc);
            }
        }

        // Build the graph from the gene data
        let mut graph: FxHashMap<NodeIndex, Vec<NodeIndex>> = FxHashMap::default();
        for (gene_key, gene) in self.data.iter() {
            if gene.enabled {
                graph.entry(gene_key.in_node_id).or_default().push(gene_key.out_node_id);
            }
        }

        // Iterate over all nodes and call `strong_connect` if the node is not yet visited
        for &node in graph.keys() {
            if !indices.contains_key(&node) {
                strong_connect(node, &mut index, &mut stack, &mut indices, &mut low_links, &mut on_stack, &mut sccs, &graph);
            }
        }

        // Return the list of SCCs
        sccs.reverse();
        sccs
    }

    pub fn add_connection(&mut self, innovation_context: &mut InnovationContext, in_node_id: NodeIndex, out_node_id: NodeIndex, weight: f64) {
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
            let in_node_id = NodeIndex(rng.gen_range(0..self.next_node_id.0));
            let out_node_id = NodeIndex(rng.gen_range(self.n_sensor_nodes..self.next_node_id.0));
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
    let new_next_node_id = NodeIndex(std::cmp::max(genome_1.next_node_id.0, genome_2.next_node_id.0));
    let new_genome = Genome{data: new_genome_data, next_node_id: new_next_node_id, n_sensor_nodes:genome_1.n_sensor_nodes, n_output_nodes:genome_1.n_output_nodes};
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
        genome.add_connection(&mut innovation_context, NodeIndex(0), NodeIndex(4), 0.0);
        assert_eq!(genome.len(), 6);
    }

    #[test]
    fn test_genome_add_node() {
        let mut genome = genome_sample_1();
        let mut innovation_context = InnovationContext::init(2, 2);
        genome.add_node(&mut innovation_context, GeneIndex(0));
        assert_eq!(genome.len(), 7);
    }

    #[test]
    fn test_tarjan_scc() {
        let genome = genome_sample_1();
        let sccs = genome.tarjan_scc_();
        assert_eq!(sccs.len(), 5);

        let genome = genome_sample_recurrent_1();
        let sccs = genome.tarjan_scc_();
        for scc in sccs {
            println!("scc size: {:?}", scc.len());
            for node_index in scc.iter() {
                println!("{:?}", node_index.0);
            }
        }
        // assert_eq!(sccs.len(), 5);
    }
}