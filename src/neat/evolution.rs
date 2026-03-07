use super::genome::innovation::InnovationTracker;
use super::genome::types::Genome;
use super::phenome::{ActivationConfig, Phenome, PhenomeError};
use super::population::{reproduce_species, ReproductionConfig};
use super::species::{
    compute_offspring_counts, speciate, update_stagnation, SpeciationConfig, Species,
};
use super::stats::{build_generation_stats, EvolutionLogger, NullLogger, OrganismStats};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OrganismId(pub usize);

/// A handle the user receives during competitive evaluation.
/// Contains a pre-built phenome ready for activation.
pub struct Organism {
    pub id: OrganismId,
    phenome: Phenome,
    fitness: f64,
    stats: OrganismStats,
}

impl Organism {
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

    pub fn node_count(&self) -> usize {
        self.phenome.node_count()
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

#[derive(Debug, Clone)]
pub struct GenerationReport {
    pub generation: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub species_count: usize,
    pub population_size: usize,
    pub compatibility_threshold: f64,
}

fn build_matchups(
    population_size: usize,
    match_config: &MatchConfig,
    rng: &mut impl RngCore,
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

fn build_organisms(
    genomes: &[Genome],
    activation_config: ActivationConfig,
) -> Vec<Result<Organism, PhenomeError>> {
    genomes
        .par_iter()
        .enumerate()
        .map(|(i, g)| {
            Phenome::from_genome_with_config(g, activation_config).map(|phenome| Organism {
                id: OrganismId(i),
                phenome,
                fitness: 0.0,
                stats: OrganismStats::new(),
            })
        })
        .collect()
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
    pub fn save_checkpoint(&self) -> EvolutionCheckpoint {
        let mut rng_clone = self.rng.clone();
        let mut seed_bytes = vec![0u8; 32];
        rng_clone.fill_bytes(&mut seed_bytes);

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
        let best_idx = fitnesses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.genomes[best_idx].clone()
    }

    /// Run one generation of competitive evolution.
    ///
    /// `evaluate_match` is called in parallel for each match. It receives a `&mut Match`
    /// and should run the game, calling `organism.activate()` and `organism.add_fitness()`
    /// as needed. It can also call `organism.stats().increment()` to record per-organism
    /// custom stats that will be aggregated per-species in the log.
    pub fn run_generation<F>(&mut self, evaluate_match: F) -> (GenerationReport, Vec<f64>)
    where
        F: Fn(&mut Match) + Send + Sync,
    {
        let (fitnesses, organism_stats) = self.evaluate_population(&evaluate_match);

        let report = self.build_report(&fitnesses);

        self.species = speciate(
            &self.genomes,
            &self.species,
            &mut self.config.speciation,
            &mut self.next_species_id,
        );

        update_stagnation(&mut self.species, &fitnesses);

        // Log stats
        let generation_stats = build_generation_stats(
            self.generation,
            &self.species,
            &fitnesses,
            &organism_stats,
            self.config.speciation.compatibility_threshold,
        );
        self.logger.log_generation(&generation_stats);

        let offspring_counts = compute_offspring_counts(
            &self.species,
            &fitnesses,
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

        let child_seeds: Vec<u64> = (0..n_active).map(|_| self.rng.gen::<u64>()).collect();

        let species_ref = &self.species;
        let genomes_ref = &self.genomes;
        let fitnesses_ref = &fitnesses;
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
            let best_idx = fitnesses
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            next_genomes.push(self.genomes[best_idx].clone());
        }
        next_genomes.truncate(self.config.population_size);

        self.genomes = next_genomes;
        self.generation += 1;

        (report, fitnesses)
    }

    fn evaluate_population<F>(
        &mut self,
        evaluate_match: &F,
    ) -> (Vec<f64>, Vec<BTreeMap<String, f64>>)
    where
        F: Fn(&mut Match) + Send + Sync,
    {
        let matchups = build_matchups(self.genomes.len(), &self.match_config, &mut self.rng);

        let organism_results: Vec<Result<Organism, PhenomeError>> =
            build_organisms(&self.genomes, self.config.activation);

        let organisms: Vec<Option<Organism>> =
            organism_results.into_iter().map(|r| r.ok()).collect();

        let organism_slots: Vec<Mutex<Option<Organism>>> =
            organisms.into_iter().map(|o| Mutex::new(o)).collect();

        matchups.par_iter().for_each(|matchup| {
            let mut taken: Vec<(usize, Organism)> = matchup
                .iter()
                .filter_map(|&idx| {
                    let mut slot = organism_slots[idx].lock().unwrap();
                    slot.take().map(|o| (idx, o))
                })
                .collect();

            if taken.len() < 2 {
                for (idx, org) in taken {
                    *organism_slots[idx].lock().unwrap() = Some(org);
                }
                return;
            }

            let organisms_for_match: Vec<Organism> = taken.drain(..).map(|(_, o)| o).collect();
            let indices: Vec<usize> = matchup.clone();

            let mut m = Match {
                organisms: organisms_for_match,
            };

            evaluate_match(&mut m);

            for (org, &idx) in m.organisms.drain(..).zip(indices.iter()) {
                *organism_slots[idx].lock().unwrap() = Some(org);
            }
        });

        let (fitnesses, organism_stats): (Vec<f64>, Vec<BTreeMap<String, f64>>) = organism_slots
            .into_iter()
            .map(|slot| {
                slot.into_inner()
                    .unwrap()
                    .map(|o| (o.fitness, o.stats.into_map()))
                    .unwrap_or((0.0, BTreeMap::new()))
            })
            .unzip();

        (fitnesses, organism_stats)
    }

    fn build_report(&self, fitnesses: &[f64]) -> GenerationReport {
        let best_fitness = fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_fitness = if fitnesses.is_empty() {
            0.0
        } else {
            fitnesses.iter().sum::<f64>() / fitnesses.len() as f64
        };

        GenerationReport {
            generation: self.generation,
            best_fitness,
            mean_fitness,
            species_count: self.species.len(),
            population_size: self.genomes.len(),
            compatibility_threshold: self.config.speciation.compatibility_threshold,
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
        F: Fn(&mut Match) + Send + Sync,
        C: FnMut(&GenerationReport, &Evolution),
    {
        let mut last_fitnesses = vec![0.0; self.genomes.len()];
        for _ in 0..generations {
            let (report, fitnesses) = self.run_generation(&evaluate_match);
            on_generation(&report, self);
            last_fitnesses = fitnesses;
        }
        self.logger.flush();
        last_fitnesses
    }
}
