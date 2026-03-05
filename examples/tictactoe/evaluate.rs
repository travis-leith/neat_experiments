use crate::tictactoe::game::*;
use crate::tictactoe::neat_agent::{board_to_inputs, outputs_to_move};
use neat_experiments::neat::evolution::Match;

fn play_one_game(m: &mut Match, cross_idx: usize, circle_idx: usize) -> GameOverState {
    fn play_turn(
        m: &mut Match,
        cross_idx: usize,
        circle_idx: usize,
        state: PlayingGameState,
    ) -> (GameBoard, GameOverState) {
        let (organism_idx, perspective) = match state.player_turn {
            Player::Cross => (cross_idx, Player::Cross),
            Player::Circle => (circle_idx, Player::Circle),
        };

        let inputs = board_to_inputs(&state.gameboard, perspective);
        let outputs = m.organisms[organism_idx]
            .activate(&inputs)
            .expect("organism activation failed during evaluation");

        let chosen = outputs_to_move(&outputs);

        match state.apply_move_or_disqualify(chosen) {
            GameState::Playing(next) => become play_turn(m, cross_idx, circle_idx, next),
            GameState::GameOver(board, result) => (board, result),
        }
    }

    let initial = new_game(Player::Cross);
    let (_, result) = play_turn(m, cross_idx, circle_idx, initial);
    result
}

fn score_game(result: GameOverState, organism_idx: usize, cross_idx: usize) -> (f64, f64) {
    let organism_player = if organism_idx == cross_idx {
        Player::Cross
    } else {
        Player::Circle
    };

    match result {
        GameOverState::Won(winner) if winner == organism_player => (3.0, 0.0),
        GameOverState::Won(_) => (0.0, 3.0),
        GameOverState::Tied => (1.0, 1.0),
        GameOverState::Disqualified(disqualified, _) if disqualified == organism_player => {
            (-1.0, 2.0)
        }
        GameOverState::Disqualified(_, _) => (2.0, -1.0),
    }
}

pub fn evaluate_tictactoe_match(m: &mut Match) {
    if m.organisms.len() < 2 {
        panic!(
            "evaluate_tictactoe_match requires at least 2 organisms, got {}",
            m.organisms.len()
        );
    }

    let result_1 = play_one_game(m, 0, 1);
    let (score_0a, score_1a) = score_game(result_1, 0, 0);

    let result_2 = play_one_game(m, 1, 0);
    let (score_1b, score_0b) = score_game(result_2, 1, 1);

    m.organisms[0].add_fitness(score_0a + score_0b);
    m.organisms[1].add_fitness(score_1a + score_1b);
}
