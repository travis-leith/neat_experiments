use crate::tictactoe::game::*;
use crate::tictactoe::minimax::score_against_perfect_play;
use crate::tictactoe::neat_agent::{board_to_inputs, outputs_to_move};
use neat_experiments::neat::evolution::Match;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Evaluate a single organism by playing one game as Cross and one as Circle
/// against a perfect opponent that randomizes among optimal moves.
///
/// Fitness = fraction of moves that matched perfect play across both games.
///
/// The RNG seed is derived from the organism's id, the current generation,
/// and the match index (approximated via accumulated fitness progression).
/// Using the generation ensures organisms face different opponent lines each
/// generation, preventing memorization of specific opponent sequences.
fn evaluate_organism(m: &mut Match, organism_idx: usize, generation: usize) -> f64 {
    let org_id = m.organisms[organism_idx].id.0 as u64;
    let generation = generation as u64;
    let match_index = m.organisms[organism_idx].fitness().to_bits();
    let seed = org_id
        .wrapping_mul(2654435761)
        .wrapping_add(generation.wrapping_mul(1099511628211))
        .wrapping_add(match_index);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rng = rand::rng();

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

    if total_moves == 0 {
        0.0
    } else {
        (total_correct as f64) / (total_moves as f64)
    }
}

/// Evaluation function compatible with the Evolution framework.
///
/// Each organism in the match is independently scored against perfect play.
/// Called once per match; the framework handles running multiple matches per
/// organism via `MatchConfig::matches_per_organism`.
pub fn evaluate_minimax_fitness(m: &mut Match, generation: usize) {
    for i in 0..m.organisms.len() {
        let fitness = evaluate_organism(m, i, generation);
        m.organisms[i].add_fitness(fitness);
    }
}
