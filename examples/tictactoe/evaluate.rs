use crate::tictactoe::game::*;
use neat_experiments::neat::evolution::Match;

fn board_to_inputs_for_organism(board: &GameBoard, organism_is_cross: bool) -> Vec<f64> {
    board
        .cells
        .iter()
        .map(|cell| match cell.0 {
            None => 0.0,
            Some(Player::Cross) if organism_is_cross => 1.0,
            Some(Player::Circle) if !organism_is_cross => 1.0,
            Some(_) => -1.0,
        })
        .collect()
}

fn pick_move_from_outputs(outputs: &[f64], board: &GameBoard) -> CellLocation {
    let mut ranked: Vec<(usize, f64)> = outputs.iter().copied().enumerate().collect();
    ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    ranked
        .iter()
        .filter_map(|(i, _)| CellLocation::from_usize(*i).filter(|loc| board.is_cell_empty(*loc)))
        .next()
        .unwrap_or_else(|| {
            board
                .available_moves()
                .next()
                .unwrap_or(CellLocation::MidMid)
        })
}

fn play_one_game(m: &mut Match, cross_idx: usize, circle_idx: usize) -> GameOverState {
    fn play_turn(
        m: &mut Match,
        cross_idx: usize,
        circle_idx: usize,
        state: PlayingGameState,
    ) -> (GameBoard, GameOverState) {
        let (organism_idx, is_cross) = match state.player_turn {
            Player::Cross => (cross_idx, true),
            Player::Circle => (circle_idx, false),
        };

        let inputs = board_to_inputs_for_organism(&state.gameboard, is_cross);
        let outputs = m.organisms[organism_idx]
            .activate(&inputs)
            .unwrap_or_else(|_| vec![0.0; 9]);

        let chosen = pick_move_from_outputs(&outputs, &state.gameboard);

        match state.apply_move_or_disqualify(chosen) {
            GameState::Playing(next) => play_turn(m, cross_idx, circle_idx, next),
            GameState::GameOver(board, result) => (board, result),
        }
    }

    let initial = new_game(Player::Cross);
    let (_, result) = play_turn(m, cross_idx, circle_idx, initial);
    result
}

fn score_game(
    result: GameOverState,
    organism_idx: usize,
    cross_idx: usize,
    circle_idx: usize,
) -> (f64, f64) {
    let organism_is_cross = organism_idx == cross_idx;
    let organism_player = if organism_is_cross {
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
        return;
    }

    // Play two games: each organism plays as Cross once and Circle once
    let result_1 = play_one_game(m, 0, 1);
    let (score_0a, score_1a) = score_game(result_1, 0, 0, 1);

    let result_2 = play_one_game(m, 1, 0);
    let (score_1b, score_0b) = score_game(result_2, 1, 1, 0);

    m.organisms[0].add_fitness(score_0a + score_0b);
    m.organisms[1].add_fitness(score_1a + score_1b);
}
