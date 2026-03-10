use crate::tictactoe::game::*;
use crate::tictactoe::minimax::score_against_perfect_play;
use crate::tictactoe::neat_agent::{board_to_inputs, outputs_to_move};
use crate::tictactoe::size_penalty::{self, NetworkSize};
use neat_experiments::neat::evolution::Match;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Evaluate a single organism by playing one game as Cross and one as Circle
/// against a perfect opponent that randomizes among optimal moves.
///
/// Returns (raw_fitness, network_size).
fn evaluate_organism(m: &mut Match, organism_idx: usize) -> (f64, NetworkSize) {
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

    let raw_fitness = total_correct as f64;

    let network_size = NetworkSize {
        nodes: m.organisms[organism_idx].active_node_count(),
        connections: m.organisms[organism_idx].active_connection_count(),
    };

    (raw_fitness, network_size)
}

/// Build an evaluation function with the given size penalty strategy.
///
/// # Example
///
/// ```ignore
/// use tictactoe::size_penalty;
///
/// // No penalty
/// let eval = make_evaluate_minimax(size_penalty::no_penalty);
///
/// // Penalise both nodes and connections
/// let eval = make_evaluate_minimax(size_penalty::compose(
///     size_penalty::threshold_nodes(25, 0.02),
///     size_penalty::threshold_connections(40, 0.01),
/// ));
///
/// // Seasonal on connections with threshold
/// let eval = make_evaluate_minimax(
///     size_penalty::seasonal_connections_with_threshold(30, 100, 0.0, 0.03),
/// );
/// ```
pub fn make_evaluate_minimax<F>(penalty_fn: F) -> impl Fn(&mut Match, usize) + Send + Sync
where
    F: Fn(NetworkSize, usize) -> f64 + Send + Sync,
{
    move |m: &mut Match, generation: usize| {
        for i in 0..m.organisms.len() {
            let (raw_fitness, network_size) = evaluate_organism(m, i);
            let penalized = size_penalty::apply(raw_fitness, network_size, generation, &penalty_fn);

            m.organisms[i].stats().increment("raw_fitness", raw_fitness);
            m.organisms[i]
                .stats()
                .increment("size_penalty_factor", penalty_fn(network_size, generation));
            m.organisms[i]
                .stats()
                .increment("node_count", network_size.nodes as f64);
            m.organisms[i]
                .stats()
                .increment("connection_count", network_size.connections as f64);

            m.organisms[i].add_raw_fitness(raw_fitness);
            m.organisms[i].add_fitness(penalized);
        }
    }
}

/// Evaluation function with no size penalty — backwards compatible.
pub fn evaluate_minimax_fitness(m: &mut Match, generation: usize) {
    for i in 0..m.organisms.len() {
        let (raw_fitness, network_size) = evaluate_organism(m, i);
        let fitness = size_penalty::apply(
            raw_fitness,
            network_size,
            generation,
            &size_penalty::no_penalty,
        );
        m.organisms[i].add_raw_fitness(raw_fitness);
        m.organisms[i].add_fitness(fitness);
    }
}
