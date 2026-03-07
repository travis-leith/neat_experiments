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
/// The framework calls this multiple times per organism according to
/// `matches_per_organism`, each time with a fresh Match. We derive a unique
/// RNG seed from the organism's id and its current accumulated fitness
/// (which changes across matches), giving each match a distinct opponent line.
fn evaluate_organism(m: &mut Match, organism_idx: usize) -> f64 {
    let org_id = m.organisms[organism_idx].id.0 as u64;
    let salt = m.organisms[organism_idx].fitness().to_bits();
    let mut rng = StdRng::seed_from_u64(org_id.wrapping_mul(2654435761).wrapping_add(salt));

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
pub fn evaluate_minimax_fitness(m: &mut Match) {
    for i in 0..m.organisms.len() {
        let fitness = evaluate_organism(m, i);
        m.organisms[i].add_fitness(fitness);
    }
}
