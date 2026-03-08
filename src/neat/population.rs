use super::genome::crossover::CrossoverPolicy;
use super::genome::innovation::InnovationTracker;
use super::genome::mutation::Mutation;
use super::genome::types::{ConnectionGene, Genome, NodeId, ParentFitness};
use super::species::Species;
use rand::prelude::IndexedRandom;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproductionConfig {
    pub mutation_rate_perturb_weight: f64,
    pub mutation_rate_add_connection: f64,
    pub mutation_rate_add_node: f64,
    pub mutation_rate_disable_connection: f64,
    pub weight_perturb_magnitude: f64,
    pub crossover_rate: f64,
    pub elitism_count: usize,
    pub interspecies_crossover_rate: f64,
}

impl Default for ReproductionConfig {
    fn default() -> Self {
        Self {
            mutation_rate_perturb_weight: 0.8,
            mutation_rate_add_connection: 0.05,
            mutation_rate_add_node: 0.03,
            mutation_rate_disable_connection: 0.01,
            weight_perturb_magnitude: 0.5,
            crossover_rate: 0.75,
            elitism_count: 1,
            interspecies_crossover_rate: 0.001,
        }
    }
}

fn select_parent<R: Rng>(member_indices: &[usize], fitnesses: &[f64], rng: &mut R) -> usize {
    // Tournament selection with size 3
    let tournament_size = member_indices.len().min(3);
    let candidates: Vec<usize> = member_indices
        .sample(rng, tournament_size)
        .copied()
        .collect();

    *candidates
        .iter()
        .max_by(|&&a, &&b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap())
        .unwrap()
}

fn random_mutations<R: Rng>(
    genome: &Genome,
    config: &ReproductionConfig,
    rng: &mut R,
) -> Vec<Mutation> {
    let mut mutations = Vec::new();

    if rng.random::<f64>() < config.mutation_rate_perturb_weight {
        let innovations: Vec<_> = genome.innovations().collect();
        if let Some(&innov) = innovations.choose(rng) {
            let delta =
                rng.random_range(-config.weight_perturb_magnitude..config.weight_perturb_magnitude);
            mutations.push(Mutation::PerturbWeight {
                innovation: innov,
                delta,
            });
        }
    }

    if rng.random::<f64>() < config.mutation_rate_add_connection {
        let nodes: Vec<NodeId> = genome.nodes.keys().copied().collect();
        if nodes.len() >= 2 {
            let &in_node = nodes.choose(rng).unwrap();
            let &out_node = nodes.choose(rng).unwrap();
            let weight = rng.random_range(-1.0..1.0);
            mutations.push(Mutation::AddConnection {
                in_node,
                out_node,
                weight,
            });
        }
    }

    if rng.random::<f64>() < config.mutation_rate_add_node {
        let enabled_innovations: Vec<_> = genome
            .connections_by_innovation
            .values()
            .filter(|c| c.enabled)
            .map(|c| c.innovation)
            .collect();
        if let Some(&split) = enabled_innovations.choose(rng) {
            mutations.push(Mutation::AddNode {
                split_innovation: split,
            });
        }
    }

    if rng.random::<f64>() < config.mutation_rate_disable_connection {
        let enabled_innovations: Vec<_> = genome
            .connections_by_innovation
            .values()
            .filter(|c| c.enabled)
            .map(|c| c.innovation)
            .collect();
        if let Some(&innov) = enabled_innovations.choose(rng) {
            mutations.push(Mutation::DisableConnection { innovation: innov });
        }
    }

    mutations
}

struct RandomCrossoverPolicy<'a, R: Rng> {
    rng: &'a mut R,
    disabled_reenable_probability: f64,
}

impl<'a, R: Rng> CrossoverPolicy for RandomCrossoverPolicy<'a, R> {
    fn choose_left_matching(&mut self, _left: &ConnectionGene, _right: &ConnectionGene) -> bool {
        self.rng.random::<bool>()
    }

    fn choose_left_when_equal_for_unmatched(&mut self) -> bool {
        self.rng.random::<bool>()
    }

    fn enable_if_either_parent_disabled(&mut self) -> bool {
        self.rng.random::<f64>() < self.disabled_reenable_probability
    }
}

fn produce_offspring_by_mutation<R: Rng>(
    parent: &Genome,
    tracker: &mut InnovationTracker,
    config: &ReproductionConfig,
    rng: &mut R,
) -> Genome {
    let mutations = random_mutations(parent, config, rng);
    if mutations.is_empty() {
        return parent.clone();
    }
    // Single clone + in-place mutations instead of N clones
    let mut child = parent.clone();
    for m in &mutations {
        if child.apply_mutation_in_place(tracker, m).is_err() {
            return parent.clone();
        }
    }
    child
}

fn produce_offspring_by_crossover<R: Rng>(
    parent_a: &Genome,
    parent_b: &Genome,
    fitness_a: f64,
    fitness_b: f64,
    tracker: &mut InnovationTracker,
    config: &ReproductionConfig,
    rng: &mut R,
) -> Genome {
    let fitness = if fitness_a > fitness_b {
        ParentFitness::Left
    } else if fitness_b > fitness_a {
        ParentFitness::Right
    } else {
        ParentFitness::Equal
    };

    let mut policy = RandomCrossoverPolicy {
        rng,
        disabled_reenable_probability: 0.75,
    };

    let child = Genome::crossover_with_policy(parent_a, parent_b, fitness, &mut policy);

    match child {
        Ok(c) => produce_offspring_by_mutation(&c, tracker, config, rng),
        Err(_) => produce_offspring_by_mutation(parent_a, tracker, config, rng),
    }
}

fn sorted_members_by_fitness(species: &Species, fitnesses: &[f64]) -> Vec<usize> {
    let mut members = species.member_indices.clone();
    members.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());
    members
}

pub fn reproduce_species<R: Rng>(
    species: &Species,
    all_genomes: &[Genome],
    all_fitnesses: &[f64],
    offspring_count: usize,
    tracker: &mut InnovationTracker,
    config: &ReproductionConfig,
    rng: &mut R,
) -> Vec<Genome> {
    if offspring_count == 0 || species.member_indices.is_empty() {
        return Vec::new();
    }

    let sorted = sorted_members_by_fitness(species, all_fitnesses);
    let mut offspring = Vec::with_capacity(offspring_count);

    // Elitism: carry forward top genomes unchanged
    let elites = config.elitism_count.min(sorted.len()).min(offspring_count);
    for &idx in sorted.iter().take(elites) {
        offspring.push(all_genomes[idx].clone());
    }

    while offspring.len() < offspring_count {
        let child = if rng.random::<f64>() < config.crossover_rate && sorted.len() >= 2 {
            let parent_a_idx = select_parent(&sorted, all_fitnesses, rng);
            let parent_b_idx = select_parent(&sorted, all_fitnesses, rng);
            produce_offspring_by_crossover(
                &all_genomes[parent_a_idx],
                &all_genomes[parent_b_idx],
                all_fitnesses[parent_a_idx],
                all_fitnesses[parent_b_idx],
                tracker,
                config,
                rng,
            )
        } else {
            let parent_idx = select_parent(&sorted, all_fitnesses, rng);
            produce_offspring_by_mutation(&all_genomes[parent_idx], tracker, config, rng)
        };
        offspring.push(child);
    }

    offspring
}
