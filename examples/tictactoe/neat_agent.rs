use crate::tictactoe::game::*;
use neat_experiments::neat::phenome::Phenome;

/// Encode the board from the given player's perspective:
/// own pieces = 1.0, opponent pieces = -1.0, empty = 0.0
pub fn board_to_inputs(board: &GameBoard, perspective: Player) -> Vec<f64> {
    board
        .cells
        .iter()
        .map(|cell| match cell.0 {
            None => 0.0,
            Some(p) if p == perspective => 1.0,
            Some(_) => -1.0,
        })
        .chain(std::iter::once(1.0))
        .collect()
}

/// Pick the cell with the highest output activation.
/// Does NOT filter for legality — the agent must learn to pick legal moves.
///
/// Panics if outputs is empty or contains NaN, or if the index doesn't map
/// to a valid cell location (which would indicate a network output size mismatch).
pub fn outputs_to_move(outputs: &[f64]) -> CellLocation {
    let (index, _) = outputs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in network outputs"))
        .expect("outputs must not be empty");

    CellLocation::from_usize(index)
        .unwrap_or_else(|| panic!("output index {index} does not map to a valid cell location"))
}

pub struct NeatAgent {
    phenome: Phenome,
    player: Player,
}

impl NeatAgent {
    pub fn new(phenome: Phenome, player: Player) -> Self {
        Self { phenome, player }
    }
}

impl Agent for NeatAgent {
    fn select_move(&mut self, state: &PlayingGameState) -> CellLocation {
        let inputs = board_to_inputs(&state.gameboard, self.player);
        let outputs = self
            .phenome
            .activate(&inputs)
            .expect("phenome activation failed during move selection");
        outputs_to_move(&outputs)
    }
}
