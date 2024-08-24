use super::{genome::{self, Genome}, organism::{self, Organism}, species::Species};
use rand::RngCore;
pub struct Settings {
    pub excess_coefficient: f64,
    pub disjoint_coefficient: f64,
    pub weight_coefficient: f64,
    pub n_organisms: usize,
    pub n_sensor_nodes: usize,
    pub n_output_nodes: usize
}
pub struct Population {
    pub species: Vec<Species>,
    species_distance_threshold: f64,
    pub generation: usize
}

//trait to evaluate the fitness of an organism
pub trait Evaluator {
    fn evaluate_single_organism(&self, organism: &mut Organism); //TODO make this more structured by modelling inputs and outputs
}

impl Population {
    pub fn add_organism(&mut self, settings: &Settings, organism: Organism) {
        let species_index =
            self.species.iter().position(|species| organism.network.genome.distance(&species.representative, settings.excess_coefficient, settings.disjoint_coefficient, settings.weight_coefficient) < self.species_distance_threshold);

        match species_index {
            Some(index) => {
                self.species[index].members.push(organism);
            }
            None => {
                let new_species = Species {
                    members: vec![organism.clone()],
                    representative: organism.network.genome,
                    champion: 0,
                    avg_fitness: 0.0
                };
                self.species.push(new_species);
            }
        }
    }

    pub fn init<R: RngCore>(rng: &mut R, settings: &Settings) -> Population {
        let mut population = Population {
            species: Vec::new(),
            species_distance_threshold: 0.3,
            generation: 0
        };

        for _ in 0..settings.n_organisms {
            let organism = Organism::init(rng, settings.n_sensor_nodes, settings.n_output_nodes);
            population.add_organism(&settings, organism);
        }

        population
    }

    pub fn evaluate<E: Evaluator>(&mut self, evaluator: &E) {
        for s in self.species.iter_mut() {
            let mut total_species_fitness = 0.0;
            let mut champion = 0;
            let mut champion_fitness = 0.0;

            for (i, org) in s.members.iter_mut().enumerate() {
                evaluator.evaluate_single_organism(org);
                total_species_fitness += org.fitness;
                if org.fitness > champion_fitness {
                    champion_fitness = org.fitness;
                    champion = i;
                }
            }

            s.champion = champion;
            s.avg_fitness = total_species_fitness / (s.members.len() as f64);
        }
    }

    pub fn next_generation<R: RngCore>(&mut self, rng: &mut R, settings: &Settings) {
        self.generation += 1;
        let mut new_population = Vec::new();
        
        let total_avg_fitness: f64 = self.species.iter().map(|s| s.avg_fitness).sum(); //TODO: move inside loop

        for s in self.species.iter() {
            let n_offspring = (s.avg_fitness / total_avg_fitness * settings.n_organisms as f64).round() as usize;
            for _ in 0..n_offspring {
                let parent_1 = &s.members[s.champion];
                let parent_2 = &s.members[s.champion];
                let child_genome = genome::cross_over(rng, &parent_1.network.genome, parent_1.fitness, &parent_2.network.genome, parent_2.fitness);
                let child = Organism::create_from_genome(settings.n_sensor_nodes, settings.n_output_nodes, child_genome);
                new_population.push(child);
            }
        }

        //clear each species
        for s in self.species.iter_mut() {
            //retain the champion
            let champion = s.members[s.champion].clone();
            new_population.push(champion);            
            s.members.clear();
        }        

        for organism in new_population {
            self.add_organism(settings, organism);
        }
    }
}