use crate::tictactoe::game::*;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

/// The outcome of perfect play from a given position.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The current player wins in `depth` moves.
    Win(u8),
    /// The game is a draw with perfect play.
    Draw,
    /// The current player loses in `depth` moves.
    Loss(u8),
}

impl Outcome {
    pub fn negate(self) -> Self {
        match self {
            Outcome::Win(d) => Outcome::Loss(d),
            Outcome::Loss(d) => Outcome::Win(d),
            Outcome::Draw => Outcome::Draw,
        }
    }

    pub fn score(self) -> i8 {
        match self {
            Outcome::Win(_) => 1,
            Outcome::Draw => 0,
            Outcome::Loss(_) => -1,
        }
    }

    pub fn is_better_than(self, other: Self) -> bool {
        match (self, other) {
            (Outcome::Win(d1), Outcome::Win(d2)) => d1 < d2,
            (Outcome::Win(_), _) => true,
            (Outcome::Draw, Outcome::Loss(_)) => true,
            (Outcome::Draw, _) => false,
            (Outcome::Loss(d1), Outcome::Loss(d2)) => d1 > d2,
            (Outcome::Loss(_), _) => false,
        }
    }
}

type Cache = HashMap<([Cell; 9], Player), Outcome>;

fn minimax_cached(state: &PlayingGameState, cache: &mut Cache) -> Outcome {
    let key = (state.gameboard.cells, state.player_turn);
    if let Some(&cached) = cache.get(&key) {
        return cached;
    }

    let moves: Vec<CellLocation> = state.gameboard.available_moves().collect();

    let result = moves
        .into_iter()
        .map(|loc| {
            match state
                .apply_move(loc)
                .expect("available_moves should only yield legal moves")
            {
                GameState::GameOver(_, GameOverState::Won(winner)) => {
                    if winner == state.player_turn {
                        Outcome::Win(1)
                    } else {
                        Outcome::Loss(1)
                    }
                }
                GameState::GameOver(_, GameOverState::Tied) => Outcome::Draw,
                GameState::GameOver(_, GameOverState::Disqualified(_, _)) => {
                    panic!("disqualification should not occur from legal moves")
                }
                GameState::Playing(next) => {
                    let child_outcome = minimax_cached(&next, cache);
                    match child_outcome.negate() {
                        Outcome::Win(d) => Outcome::Win(d + 1),
                        Outcome::Loss(d) => Outcome::Loss(d + 1),
                        Outcome::Draw => Outcome::Draw,
                    }
                }
            }
        })
        .fold(None, |best: Option<Outcome>, outcome| match best {
            None => Some(outcome),
            Some(b) if outcome.is_better_than(b) => Some(outcome),
            _ => best,
        })
        .expect("no moves available in a playing state");

    cache.insert(key, result);
    result
}

/// For a given playing state, returns the set of moves that are consistent
/// with perfect (minimax-optimal) play, along with the outcome of the position.
pub fn optimal_moves(state: &PlayingGameState) -> (Outcome, Vec<CellLocation>) {
    optimal_moves_with_cache(state, &mut Cache::new())
}

fn outcome_for_move(state: &PlayingGameState, loc: CellLocation, cache: &mut Cache) -> Outcome {
    match state
        .apply_move(loc)
        .expect("available_moves should only yield legal moves")
    {
        GameState::GameOver(_, GameOverState::Won(winner)) => {
            if winner == state.player_turn {
                Outcome::Win(1)
            } else {
                Outcome::Loss(1)
            }
        }
        GameState::GameOver(_, GameOverState::Tied) => Outcome::Draw,
        GameState::GameOver(_, GameOverState::Disqualified(_, _)) => {
            panic!("disqualification should not occur from legal moves")
        }
        GameState::Playing(next) => {
            let child_outcome = minimax_cached(&next, cache);
            match child_outcome.negate() {
                Outcome::Win(d) => Outcome::Win(d + 1),
                Outcome::Loss(d) => Outcome::Loss(d + 1),
                Outcome::Draw => Outcome::Draw,
            }
        }
    }
}

fn optimal_moves_with_cache(
    state: &PlayingGameState,
    cache: &mut Cache,
) -> (Outcome, Vec<CellLocation>) {
    let moves: Vec<CellLocation> = state.gameboard.available_moves().collect();

    let scored: Vec<(CellLocation, Outcome)> = moves
        .into_iter()
        .map(|loc| (loc, outcome_for_move(state, loc, cache)))
        .collect();

    let best_outcome = scored
        .iter()
        .map(|(_, o)| *o)
        .fold(None, |best: Option<Outcome>, outcome| match best {
            None => Some(outcome),
            Some(b) if outcome.is_better_than(b) => Some(outcome),
            _ => best,
        })
        .expect("no moves available in a playing state");

    let optimal: Vec<CellLocation> = scored
        .into_iter()
        .filter(|(_, o)| o.score() == best_outcome.score())
        .map(|(loc, _)| loc)
        .collect();

    (best_outcome, optimal)
}

/// A minimax agent that always plays optimally.
pub struct MinimaxAgent {
    cache: Cache,
}

impl MinimaxAgent {
    pub fn new() -> Self {
        Self {
            cache: Cache::new(),
        }
    }
}

impl Agent for MinimaxAgent {
    fn select_move(&mut self, state: &PlayingGameState) -> CellLocation {
        let (_, moves) = optimal_moves_with_cache(state, &mut self.cache);
        moves[0]
    }
}

/// Evaluate a sequence of board positions and agent outputs, returning
/// a score based on how many moves matched perfect play.
///
/// Walks through an entire game from the given state, using the provided
/// function to get the agent's chosen move at each step where it's the
/// agent's turn. The minimax opponent chooses randomly among optimal moves.
/// Returns (correct_moves, total_moves).
pub fn score_against_perfect_play<F, R>(
    initial_state: PlayingGameState,
    agent_player: Player,
    mut get_agent_move: F,
    rng: &mut R,
) -> (u32, u32)
where
    F: FnMut(&PlayingGameState) -> CellLocation,
    R: Rng,
{
    fn walk<F, R>(
        state: PlayingGameState,
        agent_player: Player,
        get_agent_move: &mut F,
        cache: &mut Cache,
        rng: &mut R,
        correct: u32,
        total: u32,
    ) -> (u32, u32)
    where
        F: FnMut(&PlayingGameState) -> CellLocation,
        R: Rng,
    {
        if state.player_turn != agent_player {
            // Opponent plays a random optimal move
            let (_, optimal) = optimal_moves_with_cache(&state, cache);
            let opponent_move = *optimal
                .choose(rng)
                .expect("optimal_moves returned empty list");
            match state.apply_move_or_disqualify(opponent_move) {
                GameState::Playing(next) => {
                    become walk(
                        next,
                        agent_player,
                        get_agent_move,
                        cache,
                        rng,
                        correct,
                        total,
                    )
                }
                GameState::GameOver(_, _) => (correct, total),
            }
        } else {
            let (_, optimal) = optimal_moves_with_cache(&state, cache);
            let chosen = get_agent_move(&state);
            let is_correct = optimal.contains(&chosen);
            let new_correct = correct + if is_correct { 1 } else { 0 };
            let new_total = total + 1;
            match state.apply_move_or_disqualify(chosen) {
                GameState::Playing(next) => {
                    become walk(
                        next,
                        agent_player,
                        get_agent_move,
                        cache,
                        rng,
                        new_correct,
                        new_total,
                    )
                }
                GameState::GameOver(_, _) => (new_correct, new_total),
            }
        }
    }

    let mut cache = Cache::new();
    walk(
        initial_state,
        agent_player,
        &mut get_agent_move,
        &mut cache,
        rng,
        0,
        0,
    )
}
