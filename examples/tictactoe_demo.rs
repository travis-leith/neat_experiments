

extern crate neat_experiments;
use itertools::Itertools;
use neat_experiments::neat::{common::Settings, organism::Organism, population::TurnBasedArena};
use rand_xoshiro::Xoshiro256PlusPlus;
mod tictactoe;

use tictactoe::{cli::get_user_move, game::*};
use rand::SeedableRng;
// use rand::seq::SliceRandom;
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
}

fn single_match_up(org1: &mut Organism, org2: &mut Organism) {
    let mut ctrl = NeatVsNeat{cross: org1, circle: org2};
    match play_game(&mut ctrl, new_game(Player::Cross)) {
        Ok((_, gameover_state)) => {
            match gameover_state {
                GameOverState::Tied => {
                    ctrl.circle.fitness += 1;
                    ctrl.cross.fitness += 1;
                },
                GameOverState::Won(player) => {
                    match player {
                        Player::Circle => {
                            ctrl.circle.fitness += 2;
                            // ctrl.cross.fitness -= 1;
                        },
                        Player::Cross => {
                            // ctrl.circle.fitness -= 1;
                            ctrl.cross.fitness += 2;
                        }
                    }
                },
                GameOverState::Disqualified(player,_) => {
                    match player {
                        Player::Circle => {
                            if ctrl.circle.fitness > 0 {
                                ctrl.circle.fitness -= 1;
                            }
                            // ctrl.cross.fitness += 1;
                            // ctrl.circle.fitness -= 1;
                        },
                        Player::Cross => {
                            // ctrl.circle.fitness += 1;
                            // ctrl.cross.fitness -= 1;
                            if ctrl.cross.fitness > 0 {
                                ctrl.cross.fitness -= 1;
                            }
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
}


fn describe_population_demographics(population: &Population) {
    println!("generation: {:?}; n_species: {:?}", population.generation, population.species.len());
    // for (i, s) in population.species.iter().enumerate() {
    //     println!("\tspecies: {:?}", i);
    //     println!("\tnumber of members: {:?}", s.members.len());
    //     println!("");
    // }
}

fn describe_population_fitness(population: &Population) {
    // println!("generation: {:?}; n_species: {:?}", population.generation, population.species.len());
    let avg_fitness = population.species.iter().map(|s| s.avg_fitness).sum::<f64>() / population.species.len() as f64;
    let max_fitness = population.organisms.iter().map(|o| o.fitness).fold(0, |acc, x| acc.max(x));
    let min_fitness = population.organisms.iter().map(|o| o.fitness).fold(1000, |acc, x| acc.min(x));

    // for (i, s) in population.species.iter().enumerate() {
    //     println!("\tspecies: {:?}", i);
    //     println!("\tchampion fitness: {:?}", population.organisms[s.champion].fitness);
    //     println!("\taverage fitness: {:?}", s.avg_fitness);
    //     println!("");
    // }
    println!("avg fitness: {:?}; max fitness: {:?}; min fitness: {:?}", avg_fitness, max_fitness, min_fitness);
}

fn print_best_genome(population: &Population) {
    let best_org = 
        population.species.iter()
        .map(|s| &population.organisms[s.champion])
        .max_by_key(|o| o.fitness).unwrap().clone();

    best_org.phenome.print_mermaid_graph();
}

fn test_tictactoe() {
    let mut settings = Settings::standard(10, 9);
    settings.n_organisms = 1000;
    settings.n_species_max = 60;
    settings.mutate_weight_rate = 0.1;
    settings.mutate_weight_scale = 0.1;
    settings.mutate_add_connection_rate = 0.03;
    settings.mutate_add_node_rate = 0.05;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(12);
    let mut population = Population::init(&mut rng, &settings);

    let mut evaluator = TicTacToeEvaluator;

    describe_population_demographics(&population);
    population.evaluate_all(&mut evaluator);        
    describe_population_fitness(&population);


    for _ in 0..100000 {
        population.next_generation(&mut rng, &settings);
        if population.generation % 20 == 0 {
            println!("generation: {:?}", population.generation);
            describe_population_demographics(&population);
        }
        
        population.evaluate_two_player(&mut evaluator);
        if population.generation % 20 == 0 {
            describe_population_fitness(&population);
        }
        
    }

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
}

