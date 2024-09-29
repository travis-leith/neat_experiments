use std::sync::Mutex;

use super::{common::Settings, genome::{self, Genome}, organism::{Organism, OrganismIndex, Organisms}};
use itertools::Itertools;
use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Species {
    pub id: usize,
    pub members: Vec<OrganismIndex>,
    pub representative: Genome,
    pub champion: OrganismIndex,
    pub avg_fitness: f64
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Population {
    pub species: Vec<Species>,
    pub organisms: Organisms,
    pub species_distance_threshold: f64,
    pub generation: usize,
    pub next_species_id: usize
}

pub trait SinglePlayerArena {
    fn generate_inputs(&self) -> Vec<Vec<f64>>;
    fn evaluate_organisms(&self, outputs: Vec<Vec<f64>>) -> usize;
}

pub trait TurnBasedArena {
    fn evaluate_organisms(&self, org1: &mut Organism, org2: &mut Organism);
}

impl Population {
    pub fn add_organism(&mut self, organism: Organism) {
        self.organisms.push(organism);        
    }

    fn assign_species(&mut self, settings: &Settings, organism_index: OrganismIndex) {
        let organism = &self.organisms[organism_index];

        let species_index =
            self.species.iter().position(|species| organism.genome.distance(&species.representative, settings.excess_coefficient, settings.disjoint_coefficient, settings.weight_coefficient) < self.species_distance_threshold);

        match species_index {
            Some(index) => {
                self.species[index].members.push(organism_index);
            }
            None => {
                let new_species = Species {
                    members: vec![organism_index],
                    representative: organism.genome.clone(),
                    champion: organism_index,
                    avg_fitness: 0.0,
                    id: self.next_species_id
                };
                self.next_species_id += 1;
                self.species.push(new_species);
            }
        }
    }

    fn speciate<R: RngCore>(&mut self, rng: &mut R, settings: &Settings) {
        let max_loops = 15;
        self.organisms.shuffle(rng);
        for _ in 0..max_loops {
            for s in self.species.iter_mut() {
                s.members.clear();
            }

            for i in 0..self.organisms.len() {
                self.assign_species(settings, OrganismIndex(i));
            }

            let n_non_empty_species = self.species.iter().filter(|s| !s.members.is_empty()).count();
            if n_non_empty_species < settings.n_species_min {
                self.species_distance_threshold *= 0.94;
                println!("gen: {}; n: {} - reducing species_distance_threshold to: {:.4}", self.generation, n_non_empty_species, self.species_distance_threshold);
            } else if n_non_empty_species > settings.n_species_max {
                self.species_distance_threshold *= 1.05;
                println!("gen: {}; n: {} - increasing species_distance_threshold to: {:.4}", self.generation, n_non_empty_species, self.species_distance_threshold);
            } else {
                // println!("species_distance_threshold: {}", self.species_distance_threshold);
                break;
            }
            
        }

        //remove empty species
        self.species.retain(|s| !s.members.is_empty());
    }

    pub fn init<R: RngCore>(rng: &mut R, settings: &Settings) -> Population {
        let mut organisms = Vec::new();

        println!("initializing population");
        for _ in 0..settings.n_organisms {
            let organism = Organism::init(rng, settings.n_sensor_nodes, settings.n_output_nodes);
            organisms.push(organism);
        }

        let mut res = Population {
            species: Vec::new(),
            organisms: Organisms::new(organisms),
            species_distance_threshold: 1.5,
            generation: 0,
            next_species_id: 0
        };

        println!("speciating");
        res.speciate(rng, settings);
        println!("population initialized");
        res
    }

    pub fn evaluate<E: SinglePlayerArena>(&mut self, evaluator: &mut E, clear_state: bool) {
        for s in self.species.iter_mut() {
            let mut total_species_fitness = 0;
            let mut champion = OrganismIndex(0);
            let mut champion_fitness = 0;

            for &org_index in &s.members {
                let org = &mut self.organisms[org_index];
                let inputs = evaluator.generate_inputs();
                let outputs = inputs.iter().map(|input| {
                    if clear_state {
                        org.clear_values();
                    }
                    org.activate(input)
                }).collect();
                org.fitness = evaluator.evaluate_organisms(outputs);
                total_species_fitness += org.fitness;
                if org.fitness > champion_fitness {
                    champion_fitness = org.fitness;
                    champion = org_index;
                }
            }

            s.champion = champion;
            s.avg_fitness = (total_species_fitness as f64) / (s.members.len() as f64);
        }
    }

    fn set_champions(&mut self) {
        for s in self.species.iter_mut() {
            let mut total_species_fitness = 0;
            let mut champion = OrganismIndex(0);
            let mut champion_fitness = 0;
            for &org_index in &s.members {
                let org = &self.organisms[org_index];
                total_species_fitness += org.fitness;
                if org.fitness > champion_fitness {
                    champion_fitness = org.fitness;
                    champion = org_index;
                }
            }

            s.champion = champion;
            s.avg_fitness = (total_species_fitness as f64) / (s.members.len() as f64);
        }
    }
    pub fn evaluate_single_player<A: SinglePlayerArena + Send + Sync>(&mut self, evaluator: &A, clear_state: bool) {
        use rayon::prelude::*;

        self.organisms.par_iter_mut().for_each(|org|{
            let inputs = evaluator.generate_inputs();
            let outputs = inputs.iter().map(|input| {
                if clear_state {
                    org.clear_values();
                }
                org.activate(input)
            }).collect();
            org.fitness = evaluator.evaluate_organisms(outputs);
        });

        self.set_champions();
    }

    pub fn evaluate_two_player<A: TurnBasedArena + Send + Sync>(&mut self, evaluator: &A) {
        use rayon::prelude::*;
        self.organisms.par_chunks_mut(200).for_each(|chunk| {
            for i in 0 .. chunk.len() {
                let (left, others) = chunk.split_at_mut(i);
                let (middle, right) = others.split_at_mut(1);
                let org1 = &mut middle[0];
                //process left
                for org2 in left {
                    evaluator.evaluate_organisms(org1, org2);
                }
                //process right
                for org2 in right {
                    evaluator.evaluate_organisms(org1, org2);
                }
            }
        });

        self.set_champions();
    }

    pub fn evaluate_all<A: TurnBasedArena + Send + Sync>(&mut self, evaluator: &A) {
        use rayon::prelude::*;
        use std::sync::Mutex;
        
        let mut organisms: Vec<_> = self.organisms.iter_mut().map(Mutex::new).collect();
        for i in 0 .. organisms.len() {
            let (left, others) = organisms.split_at_mut(i);
            let (middle, right) = others.split_at_mut(1);
            let org1 = &mut middle[0];
            //process left
            left.par_iter().for_each(|org2| {
                let mut org1_lock = org1.lock().unwrap();
                let mut org2_lock = org2.lock().unwrap();
                evaluator.evaluate_organisms(&mut org1_lock, &mut org2_lock);
            });
            
            //process right
            right.par_iter().for_each(|org2| {
                let mut org1_lock = org1.lock().unwrap();
                let mut org2_lock = org2.lock().unwrap();
                evaluator.evaluate_organisms(&mut org1_lock, &mut org2_lock);
            });
        }

        self.set_champions();
    }

    pub fn next_generation_par<R: RngCore>(&mut self, rng: &mut R, settings: &Settings) {
        self.generation += 1;
        
        let total_avg_fitness: f64 = self.species.iter().map(|s| s.avg_fitness).sum(); //TODO: move inside loop

        let offspring_per_species = 
            if total_avg_fitness > 0.0 {
                self.species.iter()
                .map(|s| (s.avg_fitness / total_avg_fitness * settings.n_organisms as f64).round() as usize)
                .collect_vec()
            } else {
                let n = settings.n_organisms / self.species.len();
                vec![n; self.species.len()]
            };
        
    
        // Initialize a vector of seeds from the incoming R
        let seeds: Vec<_> = (0..self.species.len()).map(|_| rng.gen()).collect();

        let new_population = Mutex::new(Vec::with_capacity(settings.n_organisms + settings.n_species_max));
    
        // Zip the seeds with the species iterator
        self.species
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, s)| {
                
                let mut local_rng = Xoshiro256PlusPlus::seed_from_u64(seeds[i]);
                let mut local_new_population = Vec::new();
    
                let n_offspring = offspring_per_species[i];
                s.members.sort_by_key(|&org_index| -(self.organisms[org_index].fitness as i64));
                let n_members = s.members.len();
                let n_breeders = (n_members as f64 * 0.3).ceil() as usize;
                for _ in 0..n_offspring {
                    let parent_1_index = s.members[local_rng.gen_range(0..n_breeders)];
                    let parent_2_index = s.members[local_rng.gen_range(0..n_breeders)];
                    let parent_1 = &self.organisms[parent_1_index];
                    let parent_2 = &self.organisms[parent_2_index];
                    let mut child_genome = genome::cross_over(&mut local_rng, &parent_1.genome, parent_1.fitness, &parent_2.genome, parent_2.fitness);
                    child_genome.mutate(&mut local_rng, settings);
                    let child = Organism::create_from_genome(child_genome);
                    local_new_population.push(child);
                }

                if n_offspring > 0 {
                    let champion = Organism::create_from_genome(self.organisms[s.champion].genome.clone());
                    local_new_population.push(champion);
                }
    
                let mut new_population = new_population.lock().unwrap();
                new_population.extend(local_new_population);
            });
    
        self.organisms = Organisms::new(new_population.into_inner().unwrap());
        self.speciate(rng, settings);
    }

    pub fn next_generation_dep<R: RngCore>(&mut self, rng: &mut R, settings: &Settings) {
        self.generation += 1;
        let mut new_population = Vec::new();
        
        let total_avg_fitness: f64 = self.species.iter().map(|s| s.avg_fitness).sum(); //TODO: move inside loop

        let offspring_per_species = 
            if total_avg_fitness > 0.0 {
                self.species.iter()
                .map(|s| (s.avg_fitness / total_avg_fitness * settings.n_organisms as f64).round() as usize)
                .collect_vec()
            } else {
                let n = settings.n_organisms / self.species.len();
                vec![n; self.species.len()]
            };

        for (i, s) in self.species.iter_mut().enumerate() {
            let n_offspring = offspring_per_species[i];
            s.members.sort_by_key(|&org_index| -(self.organisms[org_index].fitness as i64));
            let n_members = s.members.len();
            let n_breeders = (n_members as f64 * 0.4).ceil() as usize;
            for _ in 0..n_offspring {
                let parent_1_index = s.members[rng.gen_range(0..n_breeders)];
                let parent_2_index = s.members[rng.gen_range(0..n_breeders)];
                let parent_1 = &self.organisms[parent_1_index];
                let parent_2 = &self.organisms[parent_2_index];
                let mut child_genome = genome::cross_over(rng, &parent_1.genome, parent_1.fitness, &parent_2.genome, parent_2.fitness);
                child_genome.mutate(rng, settings);
                let child = Organism::create_from_genome(child_genome);
                new_population.push(child);
            }

            if n_offspring > 0 {
                let champion = Organism::create_from_genome(self.organisms[s.champion].genome.clone());
                new_population.push(champion);
            }
        }

        self.organisms = Organisms::new(new_population);
        self.speciate(rng, settings)
    }

    // pub fn trim_genomes(&mut self) {
    //     for s in self.species.iter_mut() {
    //         for &org_index in &s.members {
    //             let org = &mut self.organisms[org_index];
    //             org.trim_genome();
    //         }
    //     }
    // }

}

#[cfg(test)]
mod tests {
    use crate::neat::{common::Settings, population::Population};

    #[test]
    fn test_init() {
        let settings = Settings::standard(3, 1);
        let mut rng = rand::thread_rng();
        let population = Population::init(&mut rng, &settings);
        assert_eq!(population.organisms.len(), settings.n_organisms);
    }
}