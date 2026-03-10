use super::genome::distance::genetic_distance;
use super::genome::types::{DistanceCoefficients, Genome};
use rand::seq::IndexedRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum RepresentativeStrategy {
    /// The original founder genome is kept as representative forever.
    Permanent,
    /// A random member is chosen as representative each generation.
    RandomPerGeneration,
}

impl Default for RepresentativeStrategy {
    fn default() -> Self {
        Self::Permanent
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SpeciesId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Species {
    pub id: SpeciesId,
    pub representative: Genome,
    pub member_indices: Vec<usize>,
    pub stagnation_counter: usize,
    pub best_fitness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciationConfig {
    pub target_species_count: usize,
    pub species_count_lower_bound: usize,
    pub species_count_upper_bound: usize,
    pub compatibility_threshold: f64,
    pub compatibility_threshold_min: f64,
    pub compatibility_threshold_max: f64,
    pub threshold_adjustment_rate: f64,
    pub threshold_adjustment_max_iterations: usize,
    pub distance_coefficients: DistanceCoefficients,
    pub stagnation_limit: usize,
    pub representative_strategy: RepresentativeStrategy,
    pub pruning_interval: Option<usize>,
}

impl Default for SpeciationConfig {
    fn default() -> Self {
        Self {
            target_species_count: 10,
            species_count_lower_bound: 5,
            species_count_upper_bound: 15,
            compatibility_threshold: 3.0,
            compatibility_threshold_min: 0.001,
            compatibility_threshold_max: 50.0,
            threshold_adjustment_rate: 0.1,
            threshold_adjustment_max_iterations: 20,
            distance_coefficients: DistanceCoefficients::default(),
            stagnation_limit: 50,
            representative_strategy: RepresentativeStrategy::default(),
            pruning_interval: None,
        }
    }
}

pub fn should_prune(generation: usize, config: &SpeciationConfig) -> bool {
    match config.pruning_interval {
        Some(n) if n > 0 => generation > 0 && generation % n == 0,
        _ => false,
    }
}

fn count_species_at_threshold(
    genomes: &[Genome],
    representatives: &[Genome],
    threshold: f64,
    coefficients: DistanceCoefficients,
) -> usize {
    let mut count = representatives.len();
    for genome in genomes {
        let compatible = representatives
            .iter()
            .any(|rep| genetic_distance(rep, genome, coefficients) < threshold);
        if !compatible {
            count += 1;
        }
    }
    count
}

fn adjust_threshold_step(threshold: f64, rate: f64, increase: bool) -> f64 {
    let delta = threshold * rate;
    if increase {
        threshold + delta
    } else {
        threshold - delta
    }
}

/// Iteratively adjust the compatibility threshold until the species count
/// is on the correct side of the target. Only called when bounds are breached.
fn search_threshold_to_target(
    genomes: &[Genome],
    representatives: &[Genome],
    config: &SpeciationConfig,
    current_count: usize,
) -> f64 {
    let too_many = current_count > config.species_count_upper_bound;
    let too_few = current_count < config.species_count_lower_bound;

    if !too_many && !too_few {
        return config.compatibility_threshold;
    }

    // too_many => increase threshold to merge species
    // too_few  => decrease threshold to split species
    let increase = too_many;
    let target = config.target_species_count;

    let mut threshold = config.compatibility_threshold;

    for _ in 0..config.threshold_adjustment_max_iterations {
        threshold = adjust_threshold_step(threshold, config.threshold_adjustment_rate, increase);
        threshold = threshold.clamp(
            config.compatibility_threshold_min,
            config.compatibility_threshold_max,
        );

        let trial_count = count_species_at_threshold(
            genomes,
            representatives,
            threshold,
            config.distance_coefficients,
        );

        // If we were over the upper bound, we want to reach target or below.
        // If we were under the lower bound, we want to reach target or above.
        let satisfied = if increase {
            trial_count <= target
        } else {
            trial_count >= target
        };

        if satisfied {
            return threshold;
        }

        // Stop if we've hit the clamp limits
        if threshold <= config.compatibility_threshold_min
            || threshold >= config.compatibility_threshold_max
        {
            return threshold;
        }
    }

    threshold
}

fn find_compatible_species(
    genome: &Genome,
    species: &[Species],
    config: &SpeciationConfig,
) -> Option<usize> {
    species.iter().position(|s| {
        genetic_distance(&s.representative, genome, config.distance_coefficients)
            < config.compatibility_threshold
    })
}

fn choose_representative<R: Rng>(
    current_representative: &Genome,
    member_indices: &[usize],
    genomes: &[Genome],
    strategy: RepresentativeStrategy,
    rng: &mut R,
) -> Genome {
    match strategy {
        RepresentativeStrategy::Permanent => current_representative.clone(),
        RepresentativeStrategy::RandomPerGeneration => {
            let &idx = member_indices
                .choose(rng)
                .expect("choose_representative called on species with no members");
            genomes[idx].clone()
        }
    }
}

fn assign_genomes_to_species(
    genomes: &[Genome],
    species: &mut Vec<Species>,
    config: &SpeciationConfig,
    next_species_id: &mut u64,
) {
    for (idx, genome) in genomes.iter().enumerate() {
        match find_compatible_species(genome, species, config) {
            Some(si) => species[si].member_indices.push(idx),
            None => {
                let id = SpeciesId(*next_species_id);
                *next_species_id += 1;
                species.push(Species {
                    id,
                    representative: genome.clone(),
                    member_indices: vec![idx],
                    stagnation_counter: 0,
                    best_fitness: f64::NEG_INFINITY,
                });
            }
        }
    }
}

fn retain_and_update_representatives<R: Rng>(
    species: Vec<Species>,
    genomes: &[Genome],
    strategy: RepresentativeStrategy,
    rng: &mut R,
) -> Vec<Species> {
    species
        .into_iter()
        .filter(|s| !s.member_indices.is_empty())
        .map(|mut s| {
            s.representative =
                choose_representative(&s.representative, &s.member_indices, genomes, strategy, rng);
            s
        })
        .collect()
}

fn collect_representatives(previous_species: &[Species]) -> Vec<Genome> {
    previous_species
        .iter()
        .map(|s| s.representative.clone())
        .collect()
}

pub fn speciate<R: Rng>(
    genomes: &[Genome],
    previous_species: &[Species],
    config: &mut SpeciationConfig,
    next_species_id: &mut u64,
    rng: &mut R,
) -> Vec<Species> {
    let representatives = collect_representatives(previous_species);
    let previous_count = previous_species.len().max(1);

    // Only adjust threshold when bounds are breached
    if previous_count > config.species_count_upper_bound
        || previous_count < config.species_count_lower_bound
    {
        config.compatibility_threshold =
            search_threshold_to_target(genomes, &representatives, config, previous_count);
    }

    let mut species: Vec<Species> = previous_species
        .iter()
        .map(|s| Species {
            id: s.id,
            representative: s.representative.clone(),
            member_indices: Vec::new(),
            stagnation_counter: s.stagnation_counter,
            best_fitness: s.best_fitness,
        })
        .collect();

    assign_genomes_to_species(genomes, &mut species, config, next_species_id);

    retain_and_update_representatives(species, genomes, config.representative_strategy, rng)
}

fn adjusted_fitness(raw_fitness: f64, species_size: usize) -> f64 {
    raw_fitness / species_size as f64
}

pub fn update_stagnation(species: &mut [Species], fitnesses: &[f64]) {
    for s in species.iter_mut() {
        let species_best = s
            .member_indices
            .iter()
            .map(|&i| fitnesses[i])
            .fold(f64::NEG_INFINITY, f64::max);

        if species_best > s.best_fitness {
            s.best_fitness = species_best;
            s.stagnation_counter = 0;
        } else {
            s.stagnation_counter += 1;
        }
    }
}

pub fn compute_offspring_counts(
    species: &[Species],
    fitnesses: &[f64],
    total_population: usize,
    stagnation_limit: usize,
) -> Vec<usize> {
    let active_species: Vec<(usize, f64)> = species
        .iter()
        .enumerate()
        .filter(|(_, s)| s.stagnation_counter < stagnation_limit)
        .map(|(si, s)| {
            let sum: f64 = s
                .member_indices
                .iter()
                .map(|&i| adjusted_fitness(fitnesses[i], s.member_indices.len()))
                .sum();
            (si, sum.max(0.0))
        })
        .collect();

    let total_adjusted: f64 = active_species.iter().map(|(_, f)| f).sum();

    let mut counts = vec![0usize; species.len()];

    if total_adjusted <= 0.0 {
        // Fallback: distribute evenly among non-stagnant species
        let active_count = active_species.len().max(1);
        let per_species = total_population / active_count;
        let remainder = total_population % active_count;
        for (i, (si, _)) in active_species.iter().enumerate() {
            counts[*si] = per_species + if i < remainder { 1 } else { 0 };
        }
    } else {
        let mut assigned = 0usize;
        for (i, (si, adj)) in active_species.iter().enumerate() {
            let share = if i == active_species.len() - 1 {
                total_population - assigned
            } else {
                let raw = (adj / total_adjusted * total_population as f64).round() as usize;
                raw.min(total_population - assigned)
            };
            counts[*si] = share;
            assigned += share;
        }
    }

    counts
}
