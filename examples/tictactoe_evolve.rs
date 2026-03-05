#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

mod tictactoe;

use neat_experiments::neat::evolution::{
    Evolution, EvolutionCheckpoint, EvolutionConfig, MatchConfig,
};
use neat_experiments::neat::phenome::Phenome;
use neat_experiments::neat::species::SpeciationConfig;
use std::fs;
use std::path::Path;
use tictactoe::cli::play_against_neat;
use tictactoe::evaluate::evaluate_tictactoe_match;

const CHECKPOINT_PATH: &str = "tictactoe_checkpoint.json";

// 10 inputs: board cells from the agent's perspective + 1 bias input
// 9 outputs: one per cell, highest output is the chosen move
const N_INPUTS: usize = 10;
const N_OUTPUTS: usize = 9;

fn default_evolution_config() -> EvolutionConfig {
    let speciation_config = SpeciationConfig {
        compatibility_threshold: 0.1,
        ..Default::default()
    };
    EvolutionConfig {
        population_size: 300,
        speciation: speciation_config,
        ..Default::default()
    }
}

fn default_match_config() -> MatchConfig {
    MatchConfig {
        players_per_match: 2,
        matches_per_organism: 10,
    }
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
    evo.run(generations, evaluate_tictactoe_match, |report, _evo| {
        println!(
            "Gen {:4} | best: {:6.2} | mean: {:6.2} | species: {:3} | pop: {}",
            report.generation,
            report.best_fitness,
            report.mean_fitness,
            report.species_count,
            report.population_size,
        );
    })
}

fn train(generations: usize, seed: u64) {
    let config = default_evolution_config();
    let match_config = default_match_config();
    let mut evo = Evolution::new(N_INPUTS, N_OUTPUTS, config, match_config, seed);

    let last_fitnesses = run_and_report(&mut evo, generations);

    let checkpoint = evo.save_checkpoint();
    save_checkpoint(&checkpoint, CHECKPOINT_PATH);

    let best = evo.fittest_genome(&last_fitnesses);
    println!(
        "\nTraining complete. {} generations. Best genome has {} connections.",
        evo.generation(),
        best.connection_count()
    );
    println!("Checkpoint saved to {CHECKPOINT_PATH}");
}

fn resume(generations: usize) {
    if !Path::new(CHECKPOINT_PATH).exists() {
        eprintln!("No checkpoint found at {CHECKPOINT_PATH}. Run 'train' first.");
        return;
    }

    let checkpoint = load_checkpoint(CHECKPOINT_PATH);
    let starting_gen = checkpoint.generation;
    let mut evo = Evolution::from_checkpoint(checkpoint);

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
}

fn play() {
    if !Path::new(CHECKPOINT_PATH).exists() {
        eprintln!("No checkpoint found at {CHECKPOINT_PATH}. Run 'train' first.");
        return;
    }

    let checkpoint = load_checkpoint(CHECKPOINT_PATH);
    let mut evo = Evolution::from_checkpoint(checkpoint);

    println!("Evaluating population to find the fittest agent...");
    let (_, fitnesses) = evo.run_generation(evaluate_tictactoe_match);
    let best_genome = evo.fittest_genome(&fitnesses);

    println!(
        "Best agent: {} nodes, {} connections",
        best_genome.nodes.len(),
        best_genome.connection_count()
    );

    let phenome =
        Phenome::from_genome(&best_genome).expect("failed to build phenome from best genome");

    play_against_neat(phenome);
}

fn print_usage() {
    eprintln!("Usage: tictactoe_evolve <command> [args]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  train <generations> [seed]  - Train from scratch");
    eprintln!("  resume <generations>        - Resume training from checkpoint");
    eprintln!("  play                        - Play against the best evolved agent");
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
        _ => {
            print_usage();
        }
    }
}

// # Train for 200 generations from scratch
// cargo run --example tictactoe_evolve -- train 200 42

// # Resume for another 100 generations
// cargo run --example tictactoe_evolve -- resume 100

// # Play against the best evolved agent
// cargo run --example tictactoe_evolve -- play
