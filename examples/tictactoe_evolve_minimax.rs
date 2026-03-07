#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

mod tictactoe;

use neat_experiments::neat::evolution::{
    Evolution, EvolutionCheckpoint, EvolutionConfig, MatchConfig,
};
use neat_experiments::neat::phenome::Phenome;
use neat_experiments::neat::species::SpeciationConfig;
use neat_experiments::neat::stats::JsonFileLogger;
use std::fs;
use std::path::Path;
use tictactoe::cli::{game_loop_minimax, play_against_neat};
use tictactoe::evaluate_minimax::evaluate_minimax_fitness;

const CHECKPOINT_PATH: &str = "tictactoe_minimax_checkpoint.json";
const STATS_LOG_PATH: &str = "tictactoe_minimax_evolution_log.json";

// 10 inputs: board cells from the agent's perspective + 1 bias input
// 9 outputs: one per cell, highest output is the chosen move
const N_INPUTS: usize = 10;
const N_OUTPUTS: usize = 9;

fn default_evolution_config() -> EvolutionConfig {
    let speciation_config = SpeciationConfig {
        compatibility_threshold: 0.3,
        stagnation_limit: 30,
        ..Default::default()
    };
    EvolutionConfig {
        population_size: 300,
        speciation: speciation_config,
        ..Default::default()
    }
}

fn default_match_config() -> MatchConfig {
    // Each "match" is just 1 organism evaluated independently against minimax.
    // We still set players_per_match to 1 so the framework hands us one organism at a time,
    // but evaluate_minimax_fitness handles any number.
    MatchConfig {
        players_per_match: 1,
        matches_per_organism: 10,
    }
}

fn default_logger() -> Box<JsonFileLogger> {
    Box::new(JsonFileLogger::new(STATS_LOG_PATH.to_string()).with_flush_interval(5))
}

fn save_checkpoint(checkpoint: &EvolutionCheckpoint, path: &str) {
    let json = serde_json::to_string(checkpoint).expect("failed to serialize checkpoint");
    fs::write(path, json).expect("failed to write checkpoint file");
}

fn load_checkpoint(path: &str) -> EvolutionCheckpoint {
    let json = fs::read_to_string(path).expect("failed to read checkpoint file");
    serde_json::from_str(&json).expect("failed to deserialize checkpoint")
}

fn run_and_report(evo: &mut Evolution, generations: usize) -> Vec<f64> {
    evo.run(generations, evaluate_minimax_fitness, |report, _evo| {
        println!(
            "Gen {:4} | best: {:6.4} | mean: {:6.4} | species: {:3} | pop: {} | compat thresh: {:3.5}",
            report.generation,
            report.best_fitness,
            report.mean_fitness,
            report.species_count,
            report.population_size,
            report.compatibility_threshold,
        );
    })
}

fn train(generations: usize, seed: u64) {
    let config = default_evolution_config();
    let match_config = default_match_config();
    let mut evo = Evolution::new(N_INPUTS, N_OUTPUTS, config, match_config, seed)
        .with_logger(default_logger());

    let last_fitnesses = run_and_report(&mut evo, generations);

    let checkpoint = evo.save_checkpoint();
    save_checkpoint(&checkpoint, CHECKPOINT_PATH);

    let best = evo.fittest_genome(&last_fitnesses);
    println!(
        "\nTraining complete. {} generations. Best genome has {} connections.",
        evo.generation(),
        best.connection_count()
    );
    println!(
        "Best fitness: {:.4}",
        last_fitnesses
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!("Checkpoint saved to {CHECKPOINT_PATH}");
    println!("Evolution log saved to {STATS_LOG_PATH}");
}

fn resume(generations: usize) {
    if !Path::new(CHECKPOINT_PATH).exists() {
        eprintln!("No checkpoint found at {CHECKPOINT_PATH}. Run 'train' first.");
        return;
    }

    let checkpoint = load_checkpoint(CHECKPOINT_PATH);
    let starting_gen = checkpoint.generation;
    let mut evo = Evolution::from_checkpoint(checkpoint);
    evo.set_logger(default_logger());

    println!("Resuming from generation {starting_gen}...");

    let last_fitnesses = run_and_report(&mut evo, generations);

    let checkpoint = evo.save_checkpoint();
    save_checkpoint(&checkpoint, CHECKPOINT_PATH);

    let best = evo.fittest_genome(&last_fitnesses);
    println!(
        "\nTraining complete. Now at generation {}. Best genome has {} connections.",
        evo.generation(),
        best.connection_count()
    );
    println!("Checkpoint saved to {CHECKPOINT_PATH}");
    println!("Evolution log saved to {STATS_LOG_PATH}");
}

fn play() {
    if !Path::new(CHECKPOINT_PATH).exists() {
        eprintln!("No checkpoint found at {CHECKPOINT_PATH}. Run 'train' first.");
        return;
    }

    let checkpoint = load_checkpoint(CHECKPOINT_PATH);
    let mut evo = Evolution::from_checkpoint(checkpoint);

    println!("Evaluating population to find the fittest agent...");
    let (_, fitnesses) = evo.run_generation(evaluate_minimax_fitness);

    let best_genome = evo.fittest_genome(&fitnesses);
    let best_fitness = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "Best agent: {} nodes, {} connections, fitness: {:.4}",
        best_genome.nodes.len(),
        best_genome.connection_count(),
        best_fitness,
    );

    let phenome =
        Phenome::from_genome(&best_genome).expect("failed to build phenome from best genome");

    play_against_neat(phenome);
}

fn play_minimax() {
    println!("Playing against the perfect minimax agent...");
    game_loop_minimax();
}

fn print_usage() {
    eprintln!("Usage: tictactoe_evolve_minimax <command> [args]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  train <generations> [seed]  - Train from scratch using minimax fitness");
    eprintln!("  resume <generations>        - Resume training from checkpoint");
    eprintln!("  play                        - Play against the best evolved agent");
    eprintln!("  play-minimax                - Play against the perfect minimax agent");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => {
            let generations = args
                .get(2)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100);
            let seed = args
                .get(3)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(42);
            train(generations, seed);
        }
        "resume" => {
            let generations = args
                .get(2)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100);
            resume(generations);
        }
        "play" => {
            play();
        }
        "play-minimax" => {
            play_minimax();
        }
        _ => {
            print_usage();
        }
    }
}

// # Train for 200 generations using minimax fitness
// cargo +nightly run --example tictactoe_evolve_minimax -- train 200 42

// # Resume for another 100 generations
// cargo +nightly run --example tictactoe_evolve_minimax -- resume 100

// # Play against the best evolved agent
// cargo +nightly run --example tictactoe_evolve_minimax -- play

// # Play against the perfect minimax agent directly
// cargo +nightly run --example tictactoe_evolve_minimax -- play-minimax
