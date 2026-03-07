use super::species::Species;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Standard per-generation stats that apply to any NEAT context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub population_size: usize,
    pub species_count: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub median_fitness: f64,
    pub fitness_std_dev: f64,
    pub compatibility_threshold: f64,
    pub species_details: Vec<SpeciesStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesStats {
    pub species_id: u64,
    pub size: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub stagnation_counter: usize,
    pub representative_nodes: usize,
    pub representative_connections: usize,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub aggregated_custom_stats: BTreeMap<String, f64>,
}

pub fn build_generation_stats(
    generation: usize,
    species: &[Species],
    fitnesses: &[f64],
    organism_stats: &[BTreeMap<String, f64>],
    compatibility_threshold: f64,
) -> GenerationStats {
    let population_size = fitnesses.len();

    let best_fitness = fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mean_fitness = if fitnesses.is_empty() {
        0.0
    } else {
        fitnesses.iter().sum::<f64>() / fitnesses.len() as f64
    };

    let median_fitness = compute_median(fitnesses);
    let fitness_std_dev = compute_std_dev(fitnesses, mean_fitness);

    let species_details = species
        .iter()
        .map(|s| {
            let member_fitnesses: Vec<f64> =
                s.member_indices.iter().map(|&i| fitnesses[i]).collect();
            let species_best = member_fitnesses
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let species_mean = if member_fitnesses.is_empty() {
                0.0
            } else {
                member_fitnesses.iter().sum::<f64>() / member_fitnesses.len() as f64
            };

            let aggregated_custom_stats =
                aggregate_organism_stats(&s.member_indices, organism_stats);

            SpeciesStats {
                species_id: s.id.0,
                size: s.member_indices.len(),
                best_fitness: species_best,
                mean_fitness: species_mean,
                stagnation_counter: s.stagnation_counter,
                representative_nodes: s.representative.nodes.len(),
                representative_connections: s.representative.connection_count(),
                aggregated_custom_stats,
            }
        })
        .collect();

    GenerationStats {
        generation,
        population_size,
        species_count: species.len(),
        best_fitness,
        mean_fitness,
        median_fitness,
        fitness_std_dev,
        compatibility_threshold,
        species_details,
    }
}

fn aggregate_organism_stats(
    member_indices: &[usize],
    organism_stats: &[BTreeMap<String, f64>],
) -> BTreeMap<String, f64> {
    let mut aggregated = BTreeMap::new();
    for &idx in member_indices {
        for (key, value) in &organism_stats[idx] {
            *aggregated.entry(key.clone()).or_insert(0.0) += value;
        }
    }
    aggregated
}

fn compute_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("NaN in fitness values"));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn compute_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Trait for receiving generation stats. Implement this to customize logging behavior.
pub trait EvolutionLogger: Send {
    /// Called after each generation with the stats for that generation.
    fn log_generation(&mut self, stats: &GenerationStats);

    /// Called when evolution is complete or the logger is being dropped.
    /// Use this to flush any buffered output.
    fn flush(&mut self) {}
}

/// A no-op logger for when no logging is desired.
pub struct NullLogger;

impl EvolutionLogger for NullLogger {
    fn log_generation(&mut self, _stats: &GenerationStats) {}
}

/// Logs each generation as a JSON line to a file.
pub struct JsonFileLogger {
    entries: Vec<GenerationStats>,
    path: String,
    flush_interval: usize,
}

impl JsonFileLogger {
    pub fn new(path: String) -> Self {
        Self {
            entries: Vec::new(),
            path,
            flush_interval: 10,
        }
    }

    pub fn with_flush_interval(mut self, interval: usize) -> Self {
        self.flush_interval = interval;
        self
    }

    fn write_to_disk(&self) {
        let json =
            serde_json::to_string_pretty(&self.entries).expect("failed to serialize evolution log");
        std::fs::write(&self.path, json).expect("failed to write evolution log file");
    }
}

impl EvolutionLogger for JsonFileLogger {
    fn log_generation(&mut self, stats: &GenerationStats) {
        self.entries.push(stats.clone());
        if self.entries.len() % self.flush_interval == 0 {
            self.write_to_disk();
        }
    }

    fn flush(&mut self) {
        self.write_to_disk();
    }
}

impl Drop for JsonFileLogger {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Per-organism stats accumulator. Each organism gets its own instance
/// so that stats can later be aggregated per-species.
#[derive(Debug, Clone)]
pub struct OrganismStats {
    stats: BTreeMap<String, f64>,
}

impl OrganismStats {
    pub fn new() -> Self {
        Self {
            stats: BTreeMap::new(),
        }
    }

    pub fn increment(&mut self, key: &str, delta: f64) {
        *self.stats.entry(key.to_string()).or_insert(0.0) += delta;
    }

    pub fn set(&mut self, key: &str, value: f64) {
        self.stats.insert(key.to_string(), value);
    }

    pub fn into_map(self) -> BTreeMap<String, f64> {
        self.stats
    }

    pub fn as_map(&self) -> &BTreeMap<String, f64> {
        &self.stats
    }
}

impl Default for OrganismStats {
    fn default() -> Self {
        Self::new()
    }
}
