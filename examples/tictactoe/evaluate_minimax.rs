use crate::tictactoe::game::*;
use crate::tictactoe::minimax::score_against_perfect_play;
use crate::tictactoe::neat_agent::{board_to_inputs, outputs_to_move};
use crate::tictactoe::size_penalty;
use neat_experiments::neat::evolution::Match;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Evaluate a single organism by playing one game as Cross and one as Circle
/// against a perfect opponent that randomizes among optimal moves.
///
/// Fitness = fraction of moves that matched perfect play across both games.
///
/// Each organism gets its own RNG derived from the match seed and its index
/// within the match, ensuring deterministic but high-quality randomisation.
fn evaluate_organism(m: &mut Match, organism_idx: usize) -> (f64, usize) {
    let mut rng = StdRng::seed_from_u64(m.seed.wrapping_add(organism_idx as u64));

    let mut total_correct: u32 = 0;
    let mut total_moves: u32 = 0;

    for &agent_player in &[Player::Cross, Player::Circle] {
        let initial = new_game(Player::Cross);

        let (correct, moves) = score_against_perfect_play(
            initial,
            agent_player,
            |state| {
                let inputs = board_to_inputs(&state.gameboard, agent_player);
                let outputs = m.organisms[organism_idx]
                    .activate(&inputs)
                    .expect("organism activation failed during minimax evaluation");
                outputs_to_move(&outputs)
            },
            &mut rng,
        );

        total_correct += correct;
        total_moves += moves;
    }

    m.organisms[organism_idx]
        .stats()
        .increment("optimal_moves", total_correct as f64);
    m.organisms[organism_idx]
        .stats()
        .increment("total_moves", total_moves as f64);

    let raw_fitness = if total_moves == 0 {
        0.0
    } else {
        (total_correct as f64) / (total_moves as f64)
    };

    let node_count = m.organisms[organism_idx].node_count();
    (raw_fitness, node_count)
}

/// Evaluation function compatible with the Evolution framework.
///
/// Each organism in the match is independently scored against perfect play.
/// Called once per match; the framework handles running multiple matches per
/// organism via `MatchConfig::matches_per_organism`.
pub fn make_evaluate_minimax<F>(penalty_fn: F) -> impl Fn(&mut Match, usize) + Send + Sync
where
    F: Fn(usize, usize) -> f64 + Send + Sync,
{
    move |m: &mut Match, generation: usize| {
        for i in 0..m.organisms.len() {
            let (raw_fitness, node_count) = evaluate_organism(m, i);
            let penalized = size_penalty::apply(raw_fitness, node_count, generation, &penalty_fn);

            m.organisms[i].stats().increment("raw_fitness", raw_fitness);
            m.organisms[i]
                .stats()
                .increment("size_penalty_factor", penalty_fn(node_count, generation));

            m.organisms[i].add_fitness(penalized);
        }
    }
}

/// Evaluation function with no size penalty — backwards compatible.
pub fn evaluate_minimax_fitness(m: &mut Match, generation: usize) {
    let no_pen = size_penalty::no_penalty;
    for i in 0..m.organisms.len() {
        let (raw_fitness, node_count) = evaluate_organism(m, i);
        let fitness = size_penalty::apply(raw_fitness, node_count, generation, &no_pen);
        m.organisms[i].add_fitness(fitness);
    }
}
