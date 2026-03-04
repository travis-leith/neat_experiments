use crate::tictactoe::game::*;
use neat_experiments::neat::phenome::{Phenome, PhenomeError};

fn board_to_inputs(board: &GameBoard, perspective: Player) -> Vec<f64> {
    board
        .cells
        .iter()
        .map(|cell| match cell.0 {
            None => 0.0,
            Some(p) if p == perspective => 1.0,
            Some(_) => -1.0,
        })
        .collect()
}

fn outputs_to_move(outputs: &[f64], board: &GameBoard) -> CellLocation {
    let mut ranked: Vec<(usize, f64)> = outputs.iter().copied().enumerate().collect();
    ranked.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    ranked
        .iter()
        .filter_map(|(i, _)| CellLocation::from_usize(*i).filter(|loc| board.is_cell_empty(*loc)))
        .next()
        .unwrap_or_else(|| {
            // Fallback: pick first available
            board
                .available_moves()
                .next()
                .unwrap_or(CellLocation::MidMid)
        })
}

pub struct NeatAgent {
    phenome: Phenome,
    player: Player,
}

impl NeatAgent {
    pub fn new(phenome: Phenome, player: Player) -> Self {
        Self { phenome, player }
    }

    pub fn from_phenome_result(
        phenome_result: Result<Phenome, PhenomeError>,
        player: Player,
    ) -> Option<Self> {
        phenome_result.ok().map(|p| Self::new(p, player))
    }
}

impl Agent for NeatAgent {
    fn select_move(&mut self, state: &PlayingGameState) -> CellLocation {
        let inputs = board_to_inputs(&state.gameboard, self.player);
        match self.phenome.activate(&inputs) {
            Ok(outputs) => outputs_to_move(&outputs, &state.gameboard),
            Err(_) => state
                .gameboard
                .available_moves()
                .next()
                .unwrap_or(CellLocation::MidMid),
        }
    }
}
