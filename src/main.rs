mod tictactoe;
mod neat;
use std::rc::Rc;

use crate::tictactoe::cli::game_loop;
use crate::tictactoe::cli::get_user_move;
use crate::neat::common::*;
use crate::tictactoe::game::*;


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

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

fn neat_move(network: &mut Network, gameboard: &GameBoard) -> CellLocation {
    network.activate(gameboard.as_sensor_values());
    let network_output = network.get_output();
    let index_of_max = 
        network_output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap_or(0)
            ;

    CellLocation::from_usize(index_of_max).unwrap_or(CellLocation::BotLft)
}

struct InitNetworkAiVsUser{
    network: Network
}
impl Controller for InitNetworkAiVsUser {
    fn retry_allowed(&mut self) -> bool {
        false
    }

    fn circle_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        let res = neat_move(&mut self.network, gameboard);
        println!("AI has selected move: {:?}", res);
        res
    }

    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        get_user_move(gameboard)
    }
}

struct RandomAiVsUser;
impl Controller for RandomAiVsUser {
    fn retry_allowed(&mut self) -> bool {
        false
    }

    fn circle_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        tictactoe::cli::get_random_move(gameboard)
    }

    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        get_user_move(gameboard)
    }
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
        let res = neat_move(&mut self.circle.network, gameboard);
        // println!("circle AI has selected move: {:?}", res);
        res
    }

    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        let res = neat_move(&mut self.cross.network, gameboard);
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
                            ctrl.cross.fitness -= 1;
                        },
                        Player::Cross => {
                            ctrl.circle.fitness -= 1;
                            ctrl.cross.fitness += 1;
                        }
                    }
                },
                GameOverState::Disqualified(player,_) => {
                    match player {
                        Player::Circle => {
                            ctrl.circle.fitness -= 1;
                            ctrl.cross.fitness += 1;
                        },
                        Player::Cross => {
                            ctrl.circle.fitness += 1;
                            ctrl.cross.fitness -= 1;
                        }
                    }
                }
            }
        },
        Err(_) => unreachable!("not retryable")
    }
}
fn main() {
    let mut rng = rand::thread_rng();
    let mut all_orgs:Vec<Organism> = (1 .. 1001).map(|_| Organism::init(&mut rng, 10, 9, true)).collect();

    println!("finding best ai");
    for i in 0 .. 1000 {
        let (left, others) = all_orgs.split_at_mut(i);
        let (middle, right) = others.split_at_mut(1);
        let org1 = &mut middle[0];
        //process left
        for org2 in left {
            single_match_up(org1, org2);
        }
        //process right
        for org2 in right {
            single_match_up(org1, org2);
        }
    }

    let best_ai = all_orgs.iter().max_by_key(|o|o.fitness).unwrap();
    println!("best fitness: {}", best_ai.fitness);
    // let simple_ai = Network::init(&mut rng, 9, 9);
    let mut ai_controller = InitNetworkAiVsUser {network:best_ai.network.clone()};

    game_loop(&mut ai_controller);
}