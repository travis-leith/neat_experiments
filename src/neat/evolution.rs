use crate::neat::genome::pruning::prune_population;
use crate::neat::species::should_prune;

use super::genome::innovation::InnovationTracker;
use super::genome::types::Genome;
use super::phenome::{ActivationConfig, Phenome, PhenomeError};
use super::population::{reproduce_species, ReproductionConfig};
use super::species::{
    compute_offspring_counts, speciate, update_stagnation, SpeciationConfig, Species,
};
use super::stats::{build_generation_stats, EvolutionLogger, NullLogger, OrganismStats};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OrganismId(pub usize);

/// A handle the user receives during competitive evaluation.
/// Contains a pre-built phenome ready for activation.
pub struct Organism {
    pub id: OrganismId,
    phenome: Phenome,
    fitness: f64,
    raw_fitness: f64,
    stats: OrganismStats,
}

impl Organism {
    pub fn node_count(&self) -> usize {
        self.phenome.node_count()
    }

    pub fn connection_count(&self) -> usize {
        self.phenome.connection_count()
    }

    /// Nodes that actively participate in computation (excludes dead-end/disconnected nodes).
    pub fn active_node_count(&self) -> usize {
        self.phenome.active_node_count()
    }

    /// Enabled connections between active nodes (excludes disabled connections).
    pub fn active_connection_count(&self) -> usize {
        self.phenome.active_connection_count()
    }

    pub fn activate(&mut self, inputs: &[f64]) -> Result<Vec<f64>, PhenomeError> {
        self.phenome.activate(inputs)
    }

    pub fn add_fitness(&mut self, delta: f64) {
        self.fitness += delta;
    }

    pub fn set_fitness(&mut self, f: f64) {
        self.fitness = f;
    }

    pub fn fitness(&self) -> f64 {
        self.fitness
    }

    pub fn add_raw_fitness(&mut self, delta: f64) {
        self.raw_fitness += delta;
    }

    pub fn set_raw_fitness(&mut self, f: f64) {
        self.raw_fitness = f;
    }

    pub fn raw_fitness(&self) -> f64 {
        self.raw_fitness
    }

    pub fn stats(&mut self) -> &mut OrganismStats {
        &mut self.stats
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub speciation: SpeciationConfig,
    pub reproduction: ReproductionConfig,
    pub activation: ActivationConfig,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 150,
            speciation: SpeciationConfig::default(),
            reproduction: ReproductionConfig::default(),
            activation: ActivationConfig::default(),
        }
    }
}

/// Defines a group of organisms that will compete against each other.
/// The user's evaluate function receives a `&mut Match` and runs the game.
pub struct Match {
    pub organisms: Vec<Organism>,
    pub seed: u64,
}
/// Describes which organisms should be grouped together for competition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchConfig {
    /// How many organisms per match.
    pub players_per_match: usize,
    /// How many matches each organism should play.
    pub matches_per_organism: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            players_per_match: 2,
            matches_per_organism: 5,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SizeStats {
    pub nodes: usize,
    pub connections: usize,
}

#[derive(Debug, Clone)]
pub struct GenerationReport {
    pub generation: usize,
    pub best_penalised_fitness: f64,
    pub mean_penalised_fitness: f64,
    pub best_raw_fitness: f64,
    pub mean_raw_fitness: f64,
    pub best_size: SizeStats,
    pub mean_size: SizeStats,
    pub species_count: usize,
    pub population_size: usize,
    pub compatibility_threshold: f64,
}

/// Per-organism evaluation outcome returned from evaluate_population.
struct EvaluationResults {
    penalised_fitnesses: Vec<f64>,
    raw_fitnesses: Vec<f64>,
    organism_stats: Vec<BTreeMap<String, f64>>,
    active_sizes: Vec<SizeStats>,
}

fn build_matchups(
    population_size: usize,
    match_config: &MatchConfig,
    rng: &mut impl Rng,
) -> Vec<Vec<usize>> {
    use rand::seq::SliceRandom;

    let total_slots = population_size * match_config.matches_per_organism;
    let matches_needed =
        (total_slots + match_config.players_per_match - 1) / match_config.players_per_match;

    let mut all_indices: Vec<usize> = (0..population_size)
        .flat_map(|i| std::iter::repeat(i).take(match_config.matches_per_organism))
        .collect();
    all_indices.shuffle(rng);

    all_indices
        .chunks(match_config.players_per_match)
        .map(|chunk| chunk.to_vec())
        .filter(|chunk| chunk.len() == match_config.players_per_match)
        .take(matches_needed)
        .collect()
}

/// Pre-built phenomes, one per genome. `None` if the genome failed to build.
fn build_phenomes(genomes: &[Genome], activation_config: ActivationConfig) -> Vec<Option<Phenome>> {
    genomes
        .par_iter()
        .map(|g| Phenome::from_genome_with_config(g, activation_config).ok())
        .collect()
}

/// Per-match result: accumulated (fitness_delta, raw_fitness_delta, stats) keyed by organism index.
struct MatchResult {
    outcomes: Vec<(usize, f64, f64, OrganismStats)>,
}

/// Serializable snapshot of evolution state, suitable for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionCheckpoint {
    pub config: EvolutionConfig,
    pub match_config: MatchConfig,
    pub tracker: InnovationTracker,
    pub genomes: Vec<Genome>,
    pub species: Vec<Species>,
    pub next_species_id: u64,
    pub generation: usize,
    pub rng_seed_state: Vec<u8>,
}

fn find_best_index(fitnesses: &[f64]) -> usize {
    fitnesses
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn build_report(
    generation: usize,
    sizes: &[SizeStats],
    penalised_fitnesses: &[f64],
    raw_fitnesses: &[f64],
    species: &[Species],
    compatibility_threshold: f64,
) -> GenerationReport {
    let best_penalised_fitness = penalised_fitnesses
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_penalised_fitness = if penalised_fitnesses.is_empty() {
        0.0
    } else {
        penalised_fitnesses.iter().sum::<f64>() / penalised_fitnesses.len() as f64
    };

    let best_raw_fitness = raw_fitnesses
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_raw_fitness = if raw_fitnesses.is_empty() {
        0.0
    } else {
        raw_fitnesses.iter().sum::<f64>() / raw_fitnesses.len() as f64
    };

    let best_idx = find_best_index(penalised_fitnesses);
    let best_size = sizes.get(best_idx).copied().unwrap_or(SizeStats {
        nodes: 0,
        connections: 0,
    });

    let (total_nodes, total_connections) = sizes.iter().fold((0usize, 0usize), |(n, c), s| {
        (n + s.nodes, c + s.connections)
    });
    let count = sizes.len().max(1);
    let mean_size = SizeStats {
        nodes: total_nodes / count,
        connections: total_connections / count,
    };

    GenerationReport {
        generation,
        best_penalised_fitness,
        mean_penalised_fitness,
        best_raw_fitness,
        mean_raw_fitness,
        best_size,
        mean_size,
        species_count: species.len(),
        population_size: sizes.len(),
        compatibility_threshold,
    }
}

pub struct Evolution {
    config: EvolutionConfig,
    match_config: MatchConfig,
    tracker: InnovationTracker,
    genomes: Vec<Genome>,
    species: Vec<Species>,
    next_species_id: u64,
    generation: usize,
    rng: StdRng,
    logger: Box<dyn EvolutionLogger>,
}

impl Evolution {
    pub fn new(
        n_inputs: usize,
        n_outputs: usize,
        config: EvolutionConfig,
        match_config: MatchConfig,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut tracker = InnovationTracker::new();

        let genomes = Genome::random_fully_connected_population(
            config.population_size,
            n_inputs,
            n_outputs,
            &mut tracker,
            &mut rng,
        );

        Self {
            config,
            match_config,
            tracker,
            genomes,
            species: Vec::new(),
            next_species_id: 1,
            generation: 0,
            rng,
            logger: Box::new(NullLogger),
        }
    }

    /// Attach a logger to receive per-generation stats.
    pub fn with_logger(mut self, logger: Box<dyn EvolutionLogger>) -> Self {
        self.logger = logger;
        self
    }

    /// Replace the current logger.
    pub fn set_logger(&mut self, logger: Box<dyn EvolutionLogger>) {
        self.logger.flush();
        self.logger = logger;
    }

    /// Save current state to a checkpoint that can be serialized.
    pub fn save_checkpoint(&mut self) -> EvolutionCheckpoint {
        let mut seed_bytes = vec![0u8; 32];
        self.rng.fill_bytes(&mut seed_bytes);

        EvolutionCheckpoint {
            config: self.config.clone(),
            match_config: self.match_config.clone(),
            tracker: self.tracker.clone(),
            genomes: self.genomes.clone(),
            species: self.species.clone(),
            next_species_id: self.next_species_id,
            generation: self.generation,
            rng_seed_state: seed_bytes,
        }
    }

    /// Restore from a checkpoint.
    pub fn from_checkpoint(checkpoint: EvolutionCheckpoint) -> Self {
        let seed_array: [u8; 32] = checkpoint
            .rng_seed_state
            .try_into()
            .expect("rng seed state must be 32 bytes");
        let rng = StdRng::from_seed(seed_array);

        Self {
            config: checkpoint.config,
            match_config: checkpoint.match_config,
            tracker: checkpoint.tracker,
            genomes: checkpoint.genomes,
            species: checkpoint.species,
            next_species_id: checkpoint.next_species_id,
            generation: checkpoint.generation,
            rng,
            logger: Box::new(NullLogger),
        }
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn genomes(&self) -> &[Genome] {
        &self.genomes
    }

    pub fn species(&self) -> &[Species] {
        &self.species
    }

    pub fn config(&self) -> &EvolutionConfig {
        &self.config
    }

    /// Return the genome with the highest fitness from the last evaluation,
    /// or the first genome if no evaluation has been done.
    pub fn fittest_genome(&self, fitnesses: &[f64]) -> Genome {
        self.genomes[find_best_index(fitnesses)].clone()
    }

    /// Run one generation of competitive evolution.
    ///
    /// `evaluate_match` is called in parallel for each match. It receives a `&mut Match`
    /// and should run the game, calling `organism.activate()` and `organism.add_fitness()`
    /// (for penalised fitness) and `organism.add_raw_fitness()` (for unpenalised fitness)
    /// as needed. It can also call `organism.stats().increment()` to record per-organism
    /// custom stats that will be aggregated per-species in the log.
    pub fn run_generation<F>(&mut self, evaluate_match: F) -> (GenerationReport, Vec<f64>)
    where
        F: Fn(&mut Match, usize) + Send + Sync,
    {
        let eval_results = self.evaluate_population(&evaluate_match);
        let fitnesses = &eval_results.penalised_fitnesses;

        let report = build_report(
            self.generation,
            &eval_results.active_sizes,
            fitnesses,
            &eval_results.raw_fitnesses,
            &self.species,
            self.config.speciation.compatibility_threshold,
        );

        self.species = speciate(
            &self.genomes,
            &self.species,
            &mut self.config.speciation,
            &mut self.next_species_id,
            &mut self.rng,
        );

        update_stagnation(&mut self.species, fitnesses);

        // Log stats
        let generation_stats = build_generation_stats(
            self.generation,
            &self.species,
            fitnesses,
            &eval_results.organism_stats,
            self.config.speciation.compatibility_threshold,
        );
        self.logger.log_generation(&generation_stats);

        let offspring_counts = compute_offspring_counts(
            &self.species,
            fitnesses,
            self.config.population_size,
            self.config.speciation.stagnation_limit,
        );

        // Fork tracker and RNG for parallel reproduction
        let active_species: Vec<(usize, usize)> = offspring_counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(i, &count)| (i, count))
            .collect();

        let n_active = active_species.len();

        let max_offspring = active_species.iter().map(|(_, c)| *c).max().unwrap_or(1);
        let innovation_budget = (max_offspring as u64) * 8;
        let node_budget = (max_offspring as u32) * 4;

        let child_trackers = self.tracker.fork(n_active, innovation_budget, node_budget);

        let child_seeds: Vec<u64> = (0..n_active).map(|_| self.rng.next_u64()).collect();

        let species_ref = &self.species;
        let genomes_ref = &self.genomes;
        let fitnesses_ref = fitnesses;
        let reproduction_config = &self.config.reproduction;

        let results: Vec<(Vec<Genome>, InnovationTracker)> = active_species
            .into_par_iter()
            .zip(child_trackers.into_par_iter())
            .zip(child_seeds.into_par_iter())
            .map(|(((species_idx, count), mut tracker), seed)| {
                let mut rng = StdRng::seed_from_u64(seed);
                let offspring = reproduce_species(
                    &species_ref[species_idx],
                    genomes_ref,
                    fitnesses_ref,
                    count,
                    &mut tracker,
                    reproduction_config,
                    &mut rng,
                );
                (offspring, tracker)
            })
            .collect();

        let mut next_genomes = Vec::with_capacity(self.config.population_size);
        for (offspring, child_tracker) in results {
            next_genomes.extend(offspring);
            self.tracker.join(child_tracker);
        }

        while next_genomes.len() < self.config.population_size {
            let best_idx = find_best_index(fitnesses);
            next_genomes.push(self.genomes[best_idx].clone());
        }
        next_genomes.truncate(self.config.population_size);

        let penalised_fitnesses = eval_results.penalised_fitnesses;
        self.genomes = next_genomes;
        self.generation += 1;

        (report, penalised_fitnesses)
    }

    fn evaluate_population<F>(&mut self, evaluate_match: &F) -> EvaluationResults
    where
        F: Fn(&mut Match, usize) + Send + Sync,
    {
        let matchups = build_matchups(self.genomes.len(), &self.match_config, &mut self.rng);
        let phenomes = build_phenomes(&self.genomes, self.config.activation);

        // Compute active sizes from the phenomes before matches scatter them.
        let active_sizes: Vec<SizeStats> = phenomes
            .iter()
            .map(|p| match p {
                Some(phenome) => SizeStats {
                    nodes: phenome.active_node_count(),
                    connections: phenome.active_connection_count(),
                },
                None => SizeStats {
                    nodes: 0,
                    connections: 0,
                },
            })
            .collect();

        // Pre-generate one seed per match from the main RNG
        let match_seeds: Vec<u64> = (0..matchups.len()).map(|_| self.rng.next_u64()).collect();

        let generation = self.generation;

        // Each match builds its own organisms from cloned phenomes — no Mutex needed.
        let match_results: Vec<MatchResult> = matchups
            .par_iter()
            .zip(match_seeds.par_iter())
            .map(|(matchup, &seed)| {
                let organisms: Vec<Organism> = matchup
                    .iter()
                    .filter_map(|&idx| {
                        phenomes[idx].clone().map(|p| Organism {
                            id: OrganismId(idx),
                            phenome: p,
                            fitness: 0.0,
                            raw_fitness: 0.0,
                            stats: OrganismStats::new(),
                        })
                    })
                    .collect();

                if organisms.is_empty() {
                    return MatchResult {
                        outcomes: Vec::new(),
                    };
                }

                let mut m = Match { organisms, seed };
                evaluate_match(&mut m, generation);

                let outcomes = m
                    .organisms
                    .into_iter()
                    .map(|o| (o.id.0, o.fitness, o.raw_fitness, o.stats))
                    .collect();

                MatchResult { outcomes }
            })
            .collect();

        // Aggregate results sequentially — very fast, just summing floats.
        let pop_size = self.genomes.len();
        let mut penalised_fitnesses = vec![0.0f64; pop_size];
        let mut raw_fitnesses = vec![0.0f64; pop_size];
        let mut stats: Vec<BTreeMap<String, f64>> =
            (0..pop_size).map(|_| BTreeMap::new()).collect();

        for result in match_results {
            for (idx, fitness_delta, raw_fitness_delta, organism_stats) in result.outcomes {
                penalised_fitnesses[idx] += fitness_delta;
                raw_fitnesses[idx] += raw_fitness_delta;
                for (k, v) in organism_stats.into_map() {
                    *stats[idx].entry(k).or_insert(0.0) += v;
                }
            }
        }

        EvaluationResults {
            penalised_fitnesses,
            raw_fitnesses,
            organism_stats: stats,
            active_sizes,
        }
    }

    /// Run multiple generations, calling the callback after each.
    pub fn run<F, C>(
        &mut self,
        generations: usize,
        evaluate_match: F,
        mut on_generation: C,
    ) -> Vec<f64>
    where
        F: Fn(&mut Match, usize) + Send + Sync,
        C: FnMut(&GenerationReport, &Evolution),
    {
        let mut last_fitnesses = vec![0.0; self.genomes.len()];
        for _ in 0..generations {
            let (report, fitnesses) = self.run_generation(&evaluate_match);

            if should_prune(self.generation, &self.config.speciation) {
                self.genomes = prune_population(&self.genomes);
            }

            on_generation(&report, self);
            last_fitnesses = fitnesses;
        }
        self.logger.flush();
        last_fitnesses
    }
}
