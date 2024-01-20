#![feature(get_many_mut)]
mod tictactoe;
mod neat;

use crate::tictactoe::cli::game_loop;
use crate::tictactoe::cli::get_user_move;
use crate::neat::common::*;
use crate::tictactoe::game::*;
use itertools::Itertools;
use rand_distr::{Normal, Distribution, Uniform};

use rayon::prelude::*;
use tailcall::tailcall;
use rand::seq::SliceRandom;

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
                            ctrl.circle.fitness += 1.;
                            ctrl.cross.fitness -= 1.;
                        },
                        Player::Cross => {
                            ctrl.circle.fitness -= 1.;
                            ctrl.cross.fitness += 1.;
                        }
                    }
                },
                GameOverState::Disqualified(player,_) => {
                    match player {
                        Player::Circle => {
                            ctrl.circle.fitness -= 1.;
                            ctrl.cross.fitness += 1.;
                        },
                        Player::Cross => {
                            ctrl.circle.fitness += 1.;
                            ctrl.cross.fitness -= 1.;
                        }
                    }
                }
            }
        },
        Err(_) => unreachable!("not retryable")
    }
}

struct Species {
    members: Vec<usize>,
    representatve: Organism //TODO change to Vec<Connection>
}

struct EvaluatedSpecies<'a> {
    species: &'a Species,
    champion: usize,
    avg_fitness: f64
}

#[tailcall]
fn species_loop<'a>(mut acc:Vec<Species>, all_orgs: &Vec<Organism>, species_ix: usize, org_ix:usize, delta_t: f64) -> Vec<Species> {
    if species_ix < acc.len() {
        let this_species = &mut acc[species_ix];
        let this_org = &all_orgs[org_ix];

        let distance = genome_distance(this_org, &this_species.representatve, 1., 1., 1.);
        if distance < delta_t {
            this_species.members.push(org_ix);
            acc
        } else {
            species_loop(acc, all_orgs, species_ix + 1, org_ix, delta_t)
        }
    } else {
        let new_species = Species{
            members: vec![org_ix],
            representatve: all_orgs[org_ix].clone()
        };
        acc.push(new_species);
        acc
    }
}

#[tailcall]
fn genome_loop(mut acc:Vec<Species>, orgs:&Vec<Organism>, orgs_ix: usize, delta_t: f64) -> Vec<Species> {
    if orgs_ix < orgs.len() {
        acc = species_loop(acc, orgs, 0, orgs_ix, delta_t);
        genome_loop(acc, orgs, orgs_ix + 1, delta_t)
    } else {
        acc
    }
}

fn get_distance_stats(orgs: &Vec<Organism>) -> (f64, f64, f64) {
    let n = orgs.len();
    let get_min = |a:f64, b:f64| if a < b {a} else {b};
    let get_max = |a:f64, b:f64| if a > b {a} else {b};
    let (min, max, mean_numer) = 
        orgs.iter().tuple_combinations().par_bridge().map(|(org1, org2)|{
            genome_distance(org1, org2, 1., 1., 1.)
        }).fold(||(1., 0., 0.), |(mn, mx, num), d| {
            (get_min(mn, d), get_max(mx, d), num + d)
        }).reduce(|| (1., 0., 0.), |(a1, a2, a3),(b1, b2, b3)|{
            (get_min(a1, b1), get_max(a2, b2), a3 + b3)
        })
        ;

    let m = (n * (n - 1) / 2) as f64;
    (min, max, mean_numer / m)
}

fn main() {
    let pop_size = 2400;
    let mut rng = rand::thread_rng();
    let mut all_orgs = (0 .. pop_size).map(|_| Organism::init(&mut rng, 10, 9, true)).collect_vec();
    all_orgs.shuffle(&mut rng);
    // // let (min, max, mean) = get_distance_stats(&all_orgs);
    // // println!("min: {min}; max: {max}; mean: {mean}");

    let mut species = genome_loop(Vec::new(), &mut all_orgs, 0, 0.7);

    for (i, s) in species.iter().enumerate() {
        let n = s.members.len();
        println!("round 1 species {i} has {n} members");
    }

    all_orgs.par_chunks_mut(48).for_each(|chunk| {
        for i in 0 .. chunk.len() {
            let (left, others) = chunk.split_at_mut(i);
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
    });

    let mut total_fitness = 0.;

    let mut evaluated_species: Vec<EvaluatedSpecies> = Vec::with_capacity(species.len());

    for s in species.iter() {
        let mut champ_fitness = 0.;
        let mut champion = 0;
        let mut avg_fitness = 0.;

        for i in s.members.iter() {
            let org = &mut all_orgs[*i];
            avg_fitness += org.fitness;
            if org.fitness > champ_fitness {
                champ_fitness = org.fitness;
                champion = *i;
            }
        }
        let n = s.members.len() as f64;
        avg_fitness = avg_fitness / n;
        total_fitness += avg_fitness;

        let es = EvaluatedSpecies{
            species: s,
            champion,
            avg_fitness
        };
        evaluated_species.push(es);
    };

    let mut new_organisms:Vec<Organism> = Vec::with_capacity(pop_size + evaluated_species.len());
    for s in evaluated_species {
        let n_offspring = (s.avg_fitness / total_fitness * (pop_size as f64)) as usize;
        // choose parents
        let n_parents = std::cmp::min(n_offspring * 2, s.species.members.len());
        let parents = &s.species.members[0 .. n_parents];

        //generate offspring by cross_over and mutation
        for _ in (0 .. n_offspring) {
            let p1 = parents.choose(&mut rng).unwrap_or(&s.champion);
            let p2 = parents.choose(&mut rng).unwrap_or(&s.champion);

            let mut new_org = cross_over(&mut rng, &all_orgs[*p1], &all_orgs[*p2]);
            let between = Uniform::from(0.0..1.0);
            let normal = Normal::new(1., 0.05).unwrap();
            for i_conn in (0 .. new_org.network.genome.len()) {
                let r_unif = between.sample(&mut rng);
                if r_unif > 0.5 {
                    let r_normal = normal.sample(&mut rng);
                    new_org.network.genome[i_conn].weight *= r_normal;
                }
            }
            new_organisms.push(new_org);
        }
        let champion = all_orgs[s.champion].clone();
        new_organisms.push(champion)
    }

    new_organisms.shuffle(&mut rng);

    for s in species.iter_mut() {
        s.members.clear()
    }

    let new_species = genome_loop(species, &mut all_orgs, 0, 0.7);

    for (i, s) in new_species.iter().enumerate() {
        let n = s.members.len();
        println!("round 2 species {i} has {n} members");
    }

    new_organisms.par_chunks_mut(48).for_each(|chunk| {
        for i in 0 .. chunk.len() {
            let (left, others) = chunk.split_at_mut(i);
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
    });

    let best_ai = new_organisms.iter().max_by(|a, b| a.fitness.total_cmp(&b.fitness)).unwrap();
    println!("best fitness: {}", best_ai.fitness);
    // let simple_ai = Network::init(&mut rng, 9, 9);
    let mut ai_controller = InitNetworkAiVsUser {network:best_ai.network.clone()};

    game_loop(&mut ai_controller);

   
}