

extern crate neat_experiments;
use std::{cmp, fs::File, io::Write};

use itertools::Itertools;
use neat_experiments::neat::{common::Settings, organism::Organism, phenome::Phenome, population::TurnBasedArena};
use rand_xoshiro::Xoshiro256PlusPlus;
mod tictactoe;

use serde::{Serialize, Deserialize};
use tictactoe::{cli::get_user_move, game::*};
use rand::SeedableRng;
use neat_experiments::neat::population::Population;

impl Player {
    fn as_f64(self, perspective_of: &Player) -> f64 {
        if self == *perspective_of {
            1.0
        } else {
            -1.0
        }
    }
}

impl Cell {
    fn as_f64(self, perspective_of: &Player) -> f64 {
        self.0.map(|p| p.as_f64(perspective_of)).unwrap_or(0.)
    }
}

impl GameBoard {
    fn as_sensor_values(&self, perspective_of: &Player) -> Vec<f64> {
        self.cells.into_iter().map(|c|c.as_f64(perspective_of)).collect_vec()
    }

    pub fn move_count(&self, player: Player) -> usize {
        self.cells.iter().filter(|cell| cell.0 == Some(player)).count()
    }
}


fn neat_move(organism: &mut Organism, player: Player, game: &GameBoard) -> CellLocation {
    let mut sensor_values = game.as_sensor_values(&player);
    sensor_values.push(1.0); //bias
    
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
        let res = neat_move(&mut self.circle, Player::Circle, gameboard);
        // println!("circle AI has selected move: {:?}", res);
        res
    }

    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        let res = neat_move(&mut self.cross, Player::Cross, gameboard);
        // println!("cross AI has selected move: {:?}", res);
        res
    }

    fn reset(&mut self) {
        self.cross.clear_values();
        self.circle.clear_values();
    }
}

fn single_match_up(org1: &mut Organism, org2: &mut Organism) {
    let mut ctrl = NeatVsNeat{cross: org1, circle: org2};
    match play_game(&mut ctrl, new_game(Player::Cross)) {
        Ok((game_board, gameover_state)) => {
            match gameover_state {
                GameOverState::Tied => {
                    ctrl.circle.fitness += 3;
                    ctrl.cross.fitness += 3;
                },
                GameOverState::Won(player) => {
                    match player {
                        Player::Circle => {
                            let cross_count = game_board.move_count(Player::Cross);
                            ctrl.circle.fitness += cross_count;
                            if ctrl.cross.fitness > 0 {
                                ctrl.cross.fitness -= 1;
                            }
                        },
                        Player::Cross => {
                            let circle_count = game_board.move_count(Player::Circle);
                            ctrl.cross.fitness += circle_count;
                            if ctrl.circle.fitness > 0 {
                                ctrl.circle.fitness -= 1;
                            }
                        }
                    }
                },
                GameOverState::Disqualified(player,_) => {
                    match player {
                        Player::Circle => {
                            ctrl.cross.fitness += 1;
                            let circle_penalty = cmp::min(3, ctrl.circle.fitness);
                            ctrl.circle.fitness -= circle_penalty;
                        },
                        Player::Cross => {
                            ctrl.circle.fitness += 1;
                            let cross_penalty = cmp::min(3, ctrl.cross.fitness);
                            ctrl.cross.fitness -= cross_penalty;
                        }
                    }
                }
            }
        },
        Err(_) => unreachable!("not retryable")
    }
}

struct TicTacToeEvaluator;

impl TurnBasedArena for TicTacToeEvaluator {
    fn evaluate_organisms(&self, org1: &mut Organism, org2: &mut Organism) {
        single_match_up(org1, org2);
    }

}

struct InitNetworkAiVsUser{
    org: Organism
}
impl Controller for InitNetworkAiVsUser {
    fn retry_allowed(&mut self) -> bool {
        false
    }

    fn circle_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        let res = neat_move(&mut self.org, Player::Circle, gameboard);
        println!("AI has selected move: {:?}", res);
        res
    }

    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        get_user_move(gameboard)
    }

    fn reset(&mut self) {
        self.org.clear_values();
    }
}


fn describe_population_demographics(population: &Population) {
    println!("generation: {:?}; n_species: {:?}, species_distance: {:.8}", population.generation, population.species.len(), population.species_distance_threshold);
    // for (i, s) in population.species.iter().enumerate() {
    //     println!("\tspecies: {:?}", i);
    //     println!("\tnumber of members: {:?}", s.members.len());
    //     println!("");
    // }
}

fn describe_population_fitness(population: &Population) {
    // println!("generation: {:?}; n_species: {:?}", population.generation, population.species.len());
    let avg_fitness = population.species.iter().map(|s| s.avg_fitness).sum::<f64>() / population.species.len() as f64;
    let best_species = population.species.iter().max_by(|a, b| a.avg_fitness.partial_cmp(&b.avg_fitness).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let champion = &population.organisms[best_species.champion];
    let champ_size: usize = champion.phenome.activation_order.iter().map(|(_, x)| x.len()).sum();
    let champ_full_size = champion.genome.data.len();
    let max_fitness = population.organisms.iter().map(|o| o.fitness).fold(0, |acc, x| acc.max(x));
    let min_fitness = population.organisms.iter().map(|o| o.fitness).fold(1000, |acc, x| acc.min(x));

    // for (i, s) in population.species.iter().enumerate() {y
    //     println!("\tspecies: {:?}", i);
    //     println!("\tchampion fitness: {:?}", population.organisms[s.champion].fitness);
    //     println!("\taverage fitness: {:?}", s.avg_fitness);
    //     println!("");
    // }
    println!("avg fitness: {:.4}; max fitness: {}; best org size: {}({}); min fitness: {}, best_species: {} with size {} and fitness {:.4}", avg_fitness, max_fitness, champ_size, champ_full_size, min_fitness, best_species.id, best_species.members.len(), best_species.avg_fitness);
}

fn print_best_genome(population: &Population) {
    let best_org = 
        population.species.iter()
        .map(|s| &population.organisms[s.champion])
        .max_by_key(|o| o.fitness).unwrap().clone();

    best_org.phenome.print_mermaid_graph();
}

fn get_species_stats(population: &Population) -> Vec<(usize, usize, usize)> {
    population.species.iter().map(|s| (population.generation, s.members.len(), s.id)).collect()
}

use rmp_serde::encode::to_vec;
#[derive(Serialize, Deserialize)]
struct ApplicationState {
    population: Population,
    rng: Xoshiro256PlusPlus,
    settings: Settings
}
fn test_tictactoe() {
    let mut settings = Settings::standard(10, 9);
    settings.n_organisms = 200 * 16;
    settings.n_species_max = 200;
    settings.n_species_min = 5;
    settings.mutate_weight_rate = 0.1;
    settings.mutate_weight_scale = 0.1;
    settings.mutate_add_connection_rate = 0.03;
    settings.mutate_add_node_rate = 0.05;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
    
    let mut population = Population::init(&mut rng, &settings);
    

    let mut evaluator = TicTacToeEvaluator;

    describe_population_demographics(&population);
    println!("evaluating initial population");
    population.evaluate_two_player(&mut evaluator);        
    describe_population_fitness(&population);

    println!("starting evolution");
    let n_iterations = 1000;
    let mut species_stats = Vec::with_capacity(n_iterations + 1);
    species_stats.push(get_species_stats(&population));
    for _ in 0..n_iterations {
        population.next_generation_par(&mut rng, &settings);
        if population.generation % 20 == 0 {
            describe_population_demographics(&population);
        }
        
        population.evaluate_two_player(&mut evaluator);
        if population.generation % 20 == 0 {
            describe_population_fitness(&population);
        }

        // fn is_sorted<T: Ord>(vec: &Vec<T>) -> bool {
        //     vec.windows(2).all(|w| w[0] <= w[1])
        // }
    
        // for org in population.organisms.iter() {
        //     let distinct_nodes = &org.genome.distinct_node_ids;
        //     assert!(is_sorted(distinct_nodes));
        // }

        // if population.generation == 253 {
        //     let application_state = ApplicationState{population: population.clone(), rng: rng.clone()};
        //     let app_msgpack = to_vec(&application_state).unwrap();
        //     let mut file = File::create("app_253.mpk").unwrap();
        //     file.write_all(&app_msgpack).unwrap();
        // }
        species_stats.push(get_species_stats(&population));
        // if population.generation % 1000 == 0 {
        //     population.trim_genomes();
        // }
        
    }

    let app_state = ApplicationState{population: population.clone(), rng, settings};
    let app_state_msgpack = to_vec(&app_state).unwrap();
    let mut file = File::create("app.mpk").unwrap();
    file.write_all(&app_state_msgpack).unwrap();

    let species_stats_msgpack = to_vec(&species_stats).unwrap();
    let mut file = File::create("species_stats.mpk").unwrap();
    file.write_all(&species_stats_msgpack).unwrap();

    // print_best_genome(&population);
    let best_genome = 
        population.species.iter()
        .map(|s| &population.organisms[s.champion])
        .max_by_key(|o| o.fitness).unwrap().genome.clone();

    let best_ai = Organism::create_from_genome(best_genome);
    let mut ai_controller = InitNetworkAiVsUser {org:best_ai};

    tictactoe::cli::game_loop(&mut ai_controller);


}

fn resume_test_from_file() {
    let file = File::open("app.mpk").unwrap();
    let app_state: ApplicationState = rmp_serde::from_read(file).unwrap();

    let mut settings = app_state.settings;
    settings.excess_coefficient = 0.5;
    settings.disjoint_coefficient = 0.5;

    let mut evaluator = TicTacToeEvaluator;

    let mut population = app_state.population;

    let mut species_stats = Vec::with_capacity(1001);

    let mut rng = app_state.rng;
    
    // population.organisms.iter_mut().for_each(|o| o.trim_genome());

    for _ in 0..3000 {
        population.next_generation_par(&mut rng, &settings);
        if population.generation % 20 == 0 {
            println!("generation: {:?}", population.generation);
            describe_population_demographics(&population);
        }
        
        population.evaluate_two_player(&mut evaluator);

        if population.generation % 20 == 0 {
            describe_population_fitness(&population);
        }

        species_stats.push(get_species_stats(&population));
        
    }

    let app_state = ApplicationState{population: population.clone(), rng, settings};
    let app_state_msgpack = to_vec(&app_state).unwrap();
    let mut file = File::create("app.mpk").unwrap();
    file.write_all(&app_state_msgpack).unwrap();

    let species_stats_msgpack = to_vec(&species_stats).unwrap();
    let mut file = File::create("species_stats.mpk").unwrap();
    file.write_all(&species_stats_msgpack).unwrap();

    print_best_genome(&population);
    let best_genome = 
        population.species.iter()
        .map(|s| &population.organisms[s.champion])
        .max_by_key(|o| o.fitness).unwrap().genome.clone();

    let best_ai = Organism::create_from_genome(best_genome);
    let mut ai_controller = InitNetworkAiVsUser {org:best_ai};

    tictactoe::cli::game_loop(&mut ai_controller);
}

fn main() {
    //cargo run --release --example tictactoe_demo
    test_tictactoe();
    // resume_test_from_file();
}

