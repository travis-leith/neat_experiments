use neat_experiments::neat::evolution::{Evolution, EvolutionConfig, Match, MatchConfig};

fn main() {
    let config = EvolutionConfig {
        population_size: 150,
        ..Default::default()
    };

    let match_config = MatchConfig {
        players_per_match: 2,
        matches_per_organism: 10,
    };

    // 3 inputs, 2 outputs
    let mut evo = Evolution::new(3, 2, config, match_config, 42);

    evo.run(
        100,
        |m: &mut Match| {
            // Run a competitive game between the organisms in this match.
            // Example: a simple number-guessing duel.
            let inputs = vec![0.5, 0.3, 0.8];

            let outputs: Vec<Vec<f64>> = m
                .organisms
                .iter_mut()
                .map(|org| org.activate(&inputs).unwrap_or_default())
                .collect();

            // Score: whoever outputs closer to 1.0 on output[0] wins
            let scores: Vec<f64> = outputs
                .iter()
                .map(|out| 1.0 - (1.0 - out.get(0).copied().unwrap_or(0.0)).abs())
                .collect();

            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            for (org, &score) in m.organisms.iter_mut().zip(scores.iter()) {
                if (score - max_score).abs() < 1e-12 {
                    org.add_fitness(1.0); // winner
                } else {
                    org.add_fitness(0.0); // loser
                }
            }
        },
        |report, _evo| {
            println!(
                "Gen {:4} | best: {:.4} | mean: {:.4} | species: {} | pop: {}",
                report.generation,
                report.best_fitness,
                report.mean_fitness,
                report.species_count,
                report.population_size,
            );
        },
    );
}
