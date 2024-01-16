mod tictactoe;
mod neat;

use crate::tictactoe::cli::game_loop;
use crate::tictactoe::cli::get_user_move;
use crate::neat::common::*;
use crate::tictactoe::game::*;
use itertools::Itertools;

use rayon::prelude::*;
use tailcall::tailcall;

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

struct Species<'a> {
    members: Vec<&'a Organism>,
    representatve: &'a Organism
}



#[tailcall]
fn species_loop<'a>(mut acc:Vec<Species<'a>>, species_ix: usize, org:&'a Organism, delta_t: f64) -> Vec<Species<'a>> {
    if species_ix < acc.len() {
        let this_species = &mut acc[species_ix];
        let distance = genome_distance(org, this_species.representatve, 1., 1., 1.);
        if distance < delta_t {
            this_species.members.push(org);
            acc
        } else {
            species_loop(acc, species_ix + 1, org, delta_t)
        }
    } else {
        let new_species = Species{
            members: vec![org],
            representatve: org
        };
        acc.push(new_species);
        acc
    }
}

#[tailcall]
fn genome_loop<'a>(mut acc:Vec<Species<'a>>, orgs:Vec<&'a Organism>, orgs_ix: usize, delta_t: f64) -> Vec<Species<'a>> {
    if orgs_ix < orgs.len() {
        acc = species_loop(acc, 0, orgs[orgs_ix], delta_t);
        genome_loop(acc, orgs, orgs_ix + 1, delta_t)
    } else {
        acc
    }
}

struct DistSummary{
    min: f64,
    max: f64,
    mean: f64,
    std_dev: f64
}
fn get_distance_stats(orgs: &Vec<Organism>) -> (f64, f64, f64) {
    let n = orgs.len();
    let get_min = |a:f64, b:f64| if a < b {a} else {b};
    let get_max = |a:f64, b:f64| if a > b {a} else {b};
    let (min, max, mean_numer) = 
        orgs.iter().tuple_combinations().par_bridge().map(|(org1, org2)|{
            genome_distance(org1, org2, 1., 1., 1.)
        }).fold(||(0., 0., 0.), |(mn, mx, num), d| {
            (get_min(mn, d), get_max(mx, d), num + d)
        }).reduce(|| (0., 0., 0.), |(a1, a2, a3),(b1, b2, b3)|{
            (get_min(a1, b1), get_max(a2, b2), a3 + b3)
        })
        ;
    // orgs.iter().tuple_combinations().par_bridge().for_each(|(org1, org2)|{
    //     let d = genome_distance(org1, org2, 1., 1., 1.);
    //     if d < min {
    //         min = d;
    //     }

    //     if d > max {
    //         max = d;
    //     }

    //     mean_numer += d;
    // });
    // for (org1, org2) in orgs.iter().tuple_combinations().par_bridge().into_par_iter() {
    //     let d = genome_distance(org1, org2, 1., 1., 1.);
    //     if d < min {
    //         min = d;
    //     }

    //     if d > max {
    //         max = d;
    //     }

    //     mean_numer += d;
    // }
    let m = (n * (n - 1) / 2) as f64;
    (min, max, mean_numer / m)
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut all_orgs:Vec<Organism> = (0 .. 24000).map(|_| Organism::init(&mut rng, 10, 9, true)).collect();
    
    let (min, max, mean) = get_distance_stats(&all_orgs);
    println!("min: {min}; max: {max}; mean: {mean}");

    // all_orgs.par_chunks_mut(240).for_each(|chunk: &mut [Organism]|{
    //     for i in 0 .. chunk.len() {
    //         let (left, others) = chunk.split_at_mut(i);
    //         let (middle, right) = others.split_at_mut(1);
    //         let org1 = &mut middle[0];
    //         //process left
    //         for org2 in left {
    //             single_match_up(org1, org2);
    //         }
    //         //process right
    //         for org2 in right {
    //             single_match_up(org1, org2);
    //         }
    //     }
    // });

    // let best_ai = all_orgs.iter().max_by_key(|o|o.fitness).unwrap();
    // println!("best fitness: {}", best_ai.fitness);
    // // let simple_ai = Network::init(&mut rng, 9, 9);
    // let mut ai_controller = InitNetworkAiVsUser {network:best_ai.network.clone()};

    // game_loop(&mut ai_controller);
}