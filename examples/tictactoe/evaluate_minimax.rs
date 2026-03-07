use crate::tictactoe::game::*;
use crate::tictactoe::minimax::score_against_perfect_play;
use crate::tictactoe::neat_agent::{board_to_inputs, outputs_to_move};
use neat_experiments::neat::evolution::Match;

/// Evaluate a single organism by playing it through several games against
/// a perfect opponent, scoring each move on whether it was in the optimal set.
///
/// The organism plays as Cross in one game and Circle in another.
/// Fitness = fraction of moves that matched perfect play, summed across games.
fn evaluate_organism_against_perfect(m: &mut Match, organism_idx: usize) -> f64 {
    let mut total_correct: u32 = 0;
    let mut total_moves: u32 = 0;

    for &agent_player in &[Player::Cross, Player::Circle] {
        let initial = new_game(Player::Cross);

        let (correct, moves) = score_against_perfect_play(initial, agent_player, |state| {
            let perspective = agent_player;
            let inputs = board_to_inputs(&state.gameboard, perspective);
            let outputs = m.organisms[organism_idx]
                .activate(&inputs)
                .expect("organism activation failed during minimax evaluation");

            outputs_to_move(&outputs)
        });

        total_correct += correct;
        total_moves += moves;

        m.organisms[organism_idx]
            .stats()
            .increment("optimal_moves", correct as f64);
        m.organisms[organism_idx]
            .stats()
            .increment("total_moves", moves as f64);
    }

    let fitness = if total_moves == 0 {
        0.0
    } else {
        (total_correct as f64) / (total_moves as f64)
    };

    fitness
}

/// Evaluation function compatible with the Evolution framework.
///
/// Each organism in the match is independently scored against perfect play.
/// This does not require organisms to play against each other.
pub fn evaluate_minimax_fitness(m: &mut Match) {
    for i in 0..m.organisms.len() {
        let fitness = evaluate_organism_against_perfect(m, i);
        m.organisms[i].add_fitness(fitness);
    }
}
