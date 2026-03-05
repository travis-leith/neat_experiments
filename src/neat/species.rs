use super::genome::distance::genetic_distance;
use super::genome::types::{DistanceCoefficients, Genome};
use serde::{Deserialize, Serialize};

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
    pub compatibility_threshold: f64,
    pub compatibility_threshold_min: f64,
    pub compatibility_threshold_max: f64,
    pub threshold_adjustment_rate: f64,
    pub distance_coefficients: DistanceCoefficients,
    pub stagnation_limit: usize,
}

impl Default for SpeciationConfig {
    fn default() -> Self {
        Self {
            target_species_count: 10,
            compatibility_threshold: 3.0,
            compatibility_threshold_min: 0.001,
            compatibility_threshold_max: 50.0,
            threshold_adjustment_rate: 0.1,
            distance_coefficients: DistanceCoefficients::default(),
            stagnation_limit: 50,
        }
    }
}

fn adjust_compatibility_threshold(config: &SpeciationConfig, current_species_count: usize) -> f64 {
    let delta = config.compatibility_threshold * config.threshold_adjustment_rate;
    let adjusted = if current_species_count > config.target_species_count {
        config.compatibility_threshold + delta
    } else if current_species_count < config.target_species_count {
        config.compatibility_threshold - delta
    } else {
        config.compatibility_threshold
    };
    adjusted.clamp(
        config.compatibility_threshold_min,
        config.compatibility_threshold_max,
    )
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

pub fn speciate(
    genomes: &[Genome],
    previous_species: &[Species],
    config: &mut SpeciationConfig,
    next_species_id: &mut u64,
) -> Vec<Species> {
    let previous_count = previous_species.len().max(1);
    config.compatibility_threshold = adjust_compatibility_threshold(config, previous_count);

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

    for (idx, genome) in genomes.iter().enumerate() {
        match find_compatible_species(genome, &species, config) {
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

    // Remove empty species, update representatives
    species
        .into_iter()
        .filter(|s| !s.member_indices.is_empty())
        .map(|mut s| {
            let rep_genome = &genomes[s.member_indices[0]];
            s.representative = rep_genome.clone();
            s
        })
        .collect()
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
