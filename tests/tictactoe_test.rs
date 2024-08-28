extern crate neat_experiments;
mod tictactoe;

#[cfg(test)]
mod test {
    use neat_experiments::neat::organism::Organism;
    use crate::tictactoe::game::*;
    struct TicTacToeEvaluator;

    impl Cell {
        fn as_f64(self) -> f64 {
            match self.0 {
                Some(Player::Cross) => -1.,
                Some(Player::Circle) => -1.,
                None => 0.
            }
        }
    }

    impl GameBoard {
        fn as_sensor_values(&self) -> Vec<f64> {
            self.cells.into_iter().map(|c|c.as_f64()).collect()
        }
    }
    

    fn neat_move(organism: &mut Organism, gameboard: &GameBoard) -> CellLocation {
        let sensor_values = gameboard.as_sensor_values();
        let outputs = organism.activate(&sensor_values);
        let index_of_max = 
            outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap_or(0)
            ;
    
        CellLocation::from_usize(index_of_max).unwrap_or(CellLocation::BotLft)
    }

    struct NeatVsNeat<'a> {
        pub cross: &'a mut Organism,
        pub circle: &'a mut Organism
    }
    
    impl Controller for NeatVsNeat<'_> {
        fn retry_allowed(&mut self) -> bool {
            false
        }
    
        fn circle_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
            let res = neat_move(&mut self.circle, gameboard);
            // println!("circle AI has selected move: {:?}", res);
            res
        }
    
        fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
            let res = neat_move(&mut self.cross, gameboard);
            // println!("cross AI has selected move: {:?}", res);
            res
        }
    }

    fn single_match_up(org1: &mut Organism, org2: &mut Organism) {
        let mut ctrl = NeatVsNeat{cross: org1, circle: org2};
        match play_game(&mut ctrl, new_game(Player::Cross)) {
            Ok((_, gameover_state)) => {
                match gameover_state {
                    GameOverState::Tied => {
                        //fitness isnot changed for a tie
                    },
                    GameOverState::Won(player) => {
                        match player {
                            Player::Circle => {
                                ctrl.circle.fitness += 1;
                                // ctrl.cross.fitness -= 1;
                            },
                            Player::Cross => {
                                // ctrl.circle.fitness -= 1;
                                ctrl.cross.fitness += 1;
                            }
                        }
                    },
                    GameOverState::Disqualified(player,_) => {
                        match player {
                            Player::Circle => {
                                // ctrl.circle.fitness -= 1;
                                ctrl.cross.fitness += 1;
                            },
                            Player::Cross => {
                                ctrl.circle.fitness += 1;
                                // ctrl.cross.fitness -= 1;
                            }
                        }
                    }
                }
            },
            Err(_) => unreachable!("not retryable")
        }
    }
    
    // impl TurnBasedArena for TicTacToeEvaluator {

    // }
}