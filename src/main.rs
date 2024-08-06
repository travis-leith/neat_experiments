#![feature(get_many_mut)]
#![feature(map_try_insert)]
mod tictactoe;
mod neat;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashMap;
// use rustc_hash::FxHashMap;

// use crate::tictactoe::cli::game_loop;
use crate::tictactoe::cli::get_user_move;
use crate::neat::common::*;
use crate::tictactoe::game::*;
use itertools::Itertools;
use rand_distr::{Normal, Distribution, Uniform};
use rand::prelude::*;
use rayon::prelude::*;
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
        self.circle.fitness += 1;
        res
    }

    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        let res = neat_move(&mut self.cross.network, gameboard);
        // println!("cross AI has selected move: {:?}", res);
        self.cross.fitness +=1;
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
                    ctrl.circle.fitness +=5;
                    ctrl.cross.fitness +=5;
                },
                GameOverState::Won(player) => {
                    match player {
                        Player::Circle => {
                            ctrl.circle.fitness += 10;
                            // ctrl.cross.fitness -= 1;
                        },
                        Player::Cross => {
                            // ctrl.circle.fitness -= 1;
                            ctrl.cross.fitness += 10;
                        }
                    }
                },
                GameOverState::Disqualified(player,_) => {
                    match player {
                        Player::Circle => {
                            ctrl.circle.fitness -= 1;
                            // ctrl.cross.fitness += 1;
                        },
                        Player::Cross => {
                            // ctrl.circle.fitness += 1;
                            ctrl.cross.fitness -= 1;
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
    representative: Vec<Connection>,
    // champion: usize,
    // avg_fitness: f64,
    // champ_fitness: usize
}

struct EvaluatedSpecies<'a> {
    species: &'a Species,
    champion: usize,
    avg_fitness: f64,
    champ_fitness: usize,
    champ_clone: Organism
}

fn species_loop(mut acc:Vec<Species>, orgs: &Vec<Organism>, org_ix: usize, delta_t: f64) -> Vec<Species> {
    let org = &orgs[org_ix];
    for this_species in acc.iter_mut() {
        let distance = genome_distance(&org.network.genome, &this_species.representative, 1., 1., 1.);
        if distance < delta_t {
            this_species.members.push(org_ix);
            return acc
        }
    }

    let new_species = Species{
        members: vec![org_ix],
        representative: org.network.genome.clone(),
        // champion: 0,
        // avg_fitness: 0.,
        // champ_fitness: 0
    };
    acc.push(new_species);
    acc
}

fn genome_loop(mut acc:Vec<Species>, orgs:&Vec<Organism>, delta_t: f64) -> Vec<Species> {
    for org_ix in 0 .. orgs.len() {
        acc = species_loop(acc, orgs, org_ix, delta_t);
    }
    acc
}

fn evaluate_population<'a>(species: &'a Vec<Species>, mut orgs: Vec<Organism>) -> (Vec<EvaluatedSpecies<'a>>, f64, Vec<Organism>){
    orgs.par_chunks_mut(48).for_each(|chunk| {
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
        if s.members.len() > 0 {
            let mut champ_fitness = 0;
            let mut champion = s.members[0];
            let mut avg_fitness = 0.;

            let n = s.members.len() as f64;
            for org_ix in s.members.iter() {
                let org = &mut orgs[*org_ix];
                avg_fitness += org.fitness as f64;
                if org.fitness > champ_fitness {
                    champ_fitness = org.fitness;
                    champion = *org_ix;
                }
            }
            
            avg_fitness = avg_fitness / n;
            total_fitness += avg_fitness;

            let es = EvaluatedSpecies {
                species: &s,
                champion,
                champ_fitness,
                avg_fitness,
                champ_clone: orgs[champion].clone()
            };
            // s.champion = champion;
            // s.champ_fitness = champ_fitness;
            // s.avg_fitness = avg_fitness;
            evaluated_species.push(es);
        }// else {
        //     s.avg_fitness = 0.;
        // }
        
        
        // evaluated_species.push(es);
    }
    (evaluated_species, total_fitness, orgs)
}

fn breed_population(population_size: usize, evaluated_species: &mut Vec<EvaluatedSpecies>, orgs: &Vec<Organism>, total_fitness: f64, global_innovation: &mut InnovationNumber) -> Vec<Organism> {
    // let mut rng = rand::thread_rng();
    let mut rng =  Xoshiro256PlusPlus::seed_from_u64(2);
    let mut innovation_record = HashMap::default();
    let mut new_organisms:Vec<Organism> = Vec::with_capacity(population_size + evaluated_species.len()); //TODO: unneccesary allocation.
    // let mut total_n_offspring = 0;
    // let mut total_avg_fitness = 0.;
    for s in evaluated_species.iter_mut() {
        let n_offspring = (s.avg_fitness / total_fitness * (population_size as f64)) as usize;
        // total_n_offspring += n_offspring;
        // total_avg_fitness += s.avg_fitness;
        // choose parents
        let n_parents = std::cmp::min(n_offspring * 2, s.species.members.len());
        for memb_ix in 1 .. s.species.members.len() {
            debug_assert!(orgs[s.species.members[memb_ix]].fitness >= orgs[s.species.members[memb_ix]].fitness);
        }
        let parents = &s.species.members[0 .. n_parents];
        // println!("n_offspring: {n_offspring}; n_parents: {n_parents}");

        
        //generate offspring by cross_over and mutation
        for _ in 0 .. n_offspring {
            let i1 = *parents.choose(&mut rng).unwrap_or(&s.champion);
            let i2 = *parents.choose(&mut rng).unwrap_or(&s.champion);

            let p1 = &orgs[i1];
            let p2 = &orgs[i2];

            let mut new_net = cross_over(&mut rng, p1, p2);

            let uniform_prob = Uniform::from(0.0..1.0);
            let uniform_weight = Uniform::from(-1.0..1.0);
            let uniform_conn_ix = Uniform::from(0 .. new_net.genome.len());
            
            let normal = Normal::new(1., 0.05).unwrap();
            for i_conn in 0 .. new_net.genome.len() {
                let r_unif = uniform_prob.sample(&mut rng);
                if r_unif < 0.5 {
                    let r_normal = normal.sample(&mut rng);
                    let mut new_weight = new_net.genome[i_conn].weight * r_normal;
                    if new_weight > 1. {
                        new_weight = 1.
                    } else if new_weight < -1. {
                        new_weight = -1.
                    }
                    new_net.genome[i_conn].weight = new_weight;
                }
            }
            let r_unif = uniform_prob.sample(&mut rng);
            if r_unif < 0.005 {
                let conn_ix = uniform_conn_ix.sample(&mut rng);
                // println!("adding node between {} and {} when global inno: {}", new_net.genome[conn_ix].in_node_id, new_net.genome[conn_ix].out_node_id, global_innovation.0);
                let new_net2 = add_node(new_net, conn_ix, global_innovation, &mut innovation_record);
                // println!("next available global inno: {}", global_innovation.0);
                new_net = new_net2;
                //TODO try to refactor this so that these intermediate variables are not needed. Maybe don't return anything from "add_node", just mutate the inputs
            }

            let r_unif = uniform_prob.sample(&mut rng);
            let uniform_node_ix = Uniform::from(0 .. new_net.nodes.len());
            let hidden_node_start_ix = new_net.n_sensor_nodes + new_net.n_output_nodes;
            if r_unif < 0.005 {
                let in_node_ix = uniform_node_ix.sample(&mut rng);
                if in_node_ix < new_net.n_sensor_nodes || in_node_ix >= hidden_node_start_ix {
                    let out_node_ix = uniform_node_ix.sample(&mut rng);
                    if in_node_ix != out_node_ix && out_node_ix > new_net.n_sensor_nodes {
                        // println!("adding connection from {in_node_ix} to {out_node_ix} when global inno: {}", global_innovation.0);
                        let new_net2 =     
                            add_connection(new_net, in_node_ix, out_node_ix, uniform_weight.sample(&mut rng), global_innovation, &mut innovation_record);
                        // println!("next available global inno: {}", global_innovation.0);

                        new_net = new_net2;
                    }
                } 
            }

            if new_net.out_of_order {
                new_net.genome.sort_by_key(|conn|conn.innovation.0);
                new_net = Network::create_from_genome(new_net.n_sensor_nodes, new_net.n_output_nodes, new_net.genome, new_net.has_bias_node);
            }

            let new_org = Organism{
                network: new_net,
                fitness: 0
            };
            new_organisms.push(new_org);
        }

        let mut champion = orgs[s.champion].clone();
        // champion.fitness = 0;
        s.champion = new_organisms.len();
        new_organisms.push(champion);
    }

    // println!("total_n_offspring {total_n_offspring}; total_avg_fitness: {total_avg_fitness}; total_fitness: {total_fitness}");
    // new_organisms.shuffle(&mut rng);

    new_organisms
}

fn single_run(i_gen: usize, population_size: usize, mut species:Vec<Species>, orgs:Vec<Organism>, global_innovation: &mut InnovationNumber, delta_t: &mut f64) -> (Vec<Species>, Vec<Organism>) {
    let n_orgs = orgs.len();

    let (mut eval_species, total_fitness, orgs) = evaluate_population(&species, orgs);

    let mut new_organisms= breed_population(population_size, &mut eval_species, &orgs, total_fitness, global_innovation);

    let n_active_species = species.iter().filter(|x|x.members.len()>0).count();
    if n_active_species > 10 {
        *delta_t = *delta_t * 1.05;
    }

    
    let best_species = eval_species.iter().max_by(|s1, s2|s1.avg_fitness.total_cmp(&s2.avg_fitness)).unwrap();
    let best_ai = new_organisms[best_species.champion].clone();
    let best_fitness = best_species.avg_fitness.round();
    let best_size = best_ai.network.genome.len();
    let biggest_ai = new_organisms.iter().max_by_key(|a| a.network.genome.len()).unwrap().network.genome.len();
    let n_all_species = species.len();
    let n_active_species = species.iter().filter(|x|x.members.len()>0).count();
    let delta_t_printable = ((*delta_t*100.).round() as f64)/100.;

    if i_gen % 10 == 0 {
        println!("gen: {i_gen}; n_orgs: {n_orgs}; n_all_species: {n_all_species}; active_species: {n_active_species}; delta_t: {delta_t_printable}; best fitness: {best_fitness}, global_innov: {}, best ai size: {best_size}; biggest: {biggest_ai}", global_innovation.0);
    }

    for s in species.iter_mut() {
        s.members.clear()
    }

    let new_species = genome_loop(species, &mut new_organisms, *delta_t);

    (new_species, new_organisms)
}

fn run_tictactoe() {
    let pop_size = 2400;
    // let mut rng = rand::thread_rng();
    let mut rng =  Xoshiro256PlusPlus::seed_from_u64(2);
    let mut all_orgs = (0 .. pop_size).map(|_| Organism::init(&mut rng, 10, 9, true)).collect_vec();
    let mut global_innovation = InnovationNumber(90);

    // for (conn_ix, conn) in all_orgs[0].network.genome.iter().enumerate(){
    //     let new_key = (conn.in_node_id, conn.out_node_id);
    //     let new_innnov = InnovationNumber(conn_ix);
    //     global_innovation.insert(new_key, new_innnov);
    // }
    all_orgs.shuffle(&mut rng);

    let mut delta_t = 0.7;
    let mut species = genome_loop(Vec::new(), &all_orgs,  delta_t);
  
    for i_gen in 0 .. 2000 {
        let (species2, all_orgs2) = single_run(i_gen, pop_size, species, all_orgs, &mut global_innovation, &mut delta_t);
        species = species2;
        all_orgs = all_orgs2;

        for org_ix in 0 .. all_orgs.len() {
            for conn_ix in 1 .. all_orgs[org_ix].network.genome.len() {
                if all_orgs[org_ix].network.genome[conn_ix].innovation.0 <= all_orgs[org_ix].network.genome[conn_ix - 1].innovation.0 {
                    println!("found one")
                }
            }
        }
        
    }

    let best_ai = all_orgs.iter().max_by_key(|a| a.fitness).unwrap();
    let mut ai_controller = InitNetworkAiVsUser {network:best_ai.network.clone()};

    tictactoe::cli::game_loop(&mut ai_controller);
}

fn evaluate_xor_organism(org: &mut Organism){
    org.fitness = 0;
    org.activate(vec![0., 0.]);
    let out = org.network.get_output();
    if out[0].round() == 0. {
        org.fitness += 1;
    }

    org.activate(vec![0., 1.]);
    let out = org.network.get_output();
    if out[0].round() == 1. {
        org.fitness += 1;
    }

    org.activate(vec![1., 0.]);
    let out = org.network.get_output();
    if out[0].round() == 1. {
        org.fitness += 1;
    }

    org.activate(vec![1., 1.]);
    let out = org.network.get_output();
    if out[0].round() == 0. {
        org.fitness += 1;
    }
}

fn evaluate_population_xor<'a>(i_gen: usize, species: &'a Vec<Species>, mut orgs: Vec<Organism>) -> (Vec<EvaluatedSpecies<'a>>, f64, Vec<Organism>){
    if i_gen == 84 {
        // println!("stop");
        for (i_conn, conn) in orgs[11].network.genome.iter().enumerate(){
            if conn.enabled {
                println!("{i_conn}: {} -> ({}) -> {}", conn.in_node_id, conn.weight, conn.out_node_id)
            }
        }
        evaluate_xor_organism(&mut orgs[11]);
        evaluate_xor_organism(&mut orgs[11]);
    }
    orgs.iter_mut().for_each(|org|evaluate_xor_organism(org));
    orgs.iter_mut().for_each(|org|evaluate_xor_organism(org));
    let mut total_fitness = 0.;

    let mut evaluated_species: Vec<EvaluatedSpecies> = Vec::with_capacity(species.len());

    for s in species.iter() {
        if s.members.len() > 0 {
            let mut champ_fitness = 0;
            let mut champion = s.members[0];
            let mut avg_fitness = 0.;
            let mut champ_clone = orgs[s.members[0]].clone();

            let n = s.members.len() as f64;
            for org_ix in s.members.iter() {
                let org = &mut orgs[*org_ix];
                let stated_fitness = org.fitness;
                evaluate_xor_organism(org);
                if org.fitness != stated_fitness {
                    println!("found 1");
                }
                avg_fitness += org.fitness as f64;
                if org.fitness > champ_fitness {
                    champ_fitness = org.fitness;
                    champion = *org_ix;
                    champ_clone = org.clone()
                }
            }
            avg_fitness = avg_fitness / n;
            total_fitness += avg_fitness;

            let es = EvaluatedSpecies {
                species: &s,
                champion,
                champ_fitness,
                avg_fitness,
                champ_clone//: orgs[champion].clone()
            };
            // s.champion = champion;
            // s.champ_fitness = champ_fitness;
            // s.avg_fitness = avg_fitness;
            evaluated_species.push(es);
        }
        
    }
    (evaluated_species, total_fitness, orgs)
}

fn single_run_xor(i_gen: usize, population_size: usize, mut species:Vec<Species>, orgs:Vec<Organism>, global_innovation: &mut InnovationNumber, delta_t: &mut f64) -> (Vec<Species>, Vec<Organism>, Organism) {
    
    let (mut evaluated_species, total_fitness, orgs) = evaluate_population_xor(i_gen, &species, orgs);

    let mut new_organisms= breed_population(population_size, &mut evaluated_species, &orgs, total_fitness, global_innovation);

    let n_active_species = species.iter().filter(|x|x.members.len()>0).count();
    if n_active_species > 10 {
        *delta_t = *delta_t * 1.05;
    }

    let n_orgs = new_organisms.len();
    let best_species = evaluated_species.iter().max_by(|s1, s2|s1.avg_fitness.total_cmp(&s2.avg_fitness)).unwrap();
    let mut best_ai = new_organisms[best_species.champion].clone();
    let mut best_ai_clone = best_species.champ_clone.clone();
    let best_species_fitness = best_species.avg_fitness.round();
    let best_org_fitness = best_species.champ_fitness;
    let best_size = best_ai.network.genome.len();
    let biggest_ai = new_organisms.iter().max_by_key(|a| a.network.genome.len()).unwrap().network.genome.len();
    let n_all_species = species.len();
    let n_active_species = species.iter().filter(|x|x.members.len()>0).count();
    let delta_t_printable = ((*delta_t*100.).round() as f64)/100.;

    if best_species_fitness == 4. {
        println!("optimal species found i_gen {i_gen} with fitness {}, champ fitness: {} with size: {}",best_species.avg_fitness, best_species.champ_fitness, best_ai.network.genome.len());
        
    }
    if i_gen % 10 == 0 {
        println!("gen: {i_gen}; n_orgs: {n_orgs}; n_all_species: {n_all_species}; active_species: {n_active_species}; delta_t: {delta_t_printable}; best fitness: {best_org_fitness}, global_innov: {}, best ai size: {best_size}; biggest: {biggest_ai}", global_innovation.0);
        println!("interim best ai fitness: {}", best_ai.fitness);
        best_ai.fitness = 0;
        evaluate_xor_organism(&mut best_ai);
        evaluate_xor_organism(&mut best_ai_clone);
        println!("verified best ai fitness: {}", best_ai.fitness);
        println!("verified best ai clone fitness: {}", best_ai_clone.fitness);
    }
    for s in species.iter_mut() {
        s.members.clear()
    }

    let new_species = genome_loop(species, &mut new_organisms, *delta_t);

    (new_species, new_organisms, best_ai)
}
fn run_xor() {
    let pop_size = 11;
    // let mut rng = rand::thread_rng();
    let mut rng =  Xoshiro256PlusPlus::seed_from_u64(3);
    let mut all_orgs = (0 .. pop_size).map(|_| Organism::init(&mut rng, 3, 1, true)).collect_vec();
    let mut global_innovation = InnovationNumber(3);

    // for (conn_ix, conn) in all_orgs[0].network.genome.iter().enumerate(){
    //     let new_key = (conn.in_node_id, conn.out_node_id);
    //     let new_innnov = InnovationNumber(conn_ix);
    //     global_innovation.insert(new_key, new_innnov);
    // }
    all_orgs.shuffle(&mut rng);

    let mut delta_t = 0.7;
    let mut species = genome_loop(Vec::new(), &all_orgs,  delta_t);
    let mut best_ai = all_orgs[0].clone();
    for i_gen in 0 .. 100 {
        let (species2, all_orgs2, best_ai2) = single_run_xor(i_gen, pop_size, species, all_orgs, &mut global_innovation, &mut delta_t);
        species = species2;
        all_orgs = all_orgs2;
        best_ai = best_ai2;
    }

    // let mut best_ai = all_orgs.iter().max_by_key(|a| a.fitness).unwrap().clone();
    println!("final best ai fitness: {}", best_ai.fitness);
    best_ai.fitness = 0;
    evaluate_xor_organism(&mut best_ai);
    println!("verified best ai fitness: {}", best_ai.fitness);

}

fn main() {
    
    run_xor()
   
}