use rand::RngCore;

use super::{genome::Genome, organism::Organism};

struct Species {
    members: Vec<Genome>,
    representative: Genome,
    champion: usize,
    avg_fitness: f64
}

struct Population {
    species: Vec<Species>,
    generation: usize
}

struct Settings {
    excess_coefficient: f64,
    disjoint_coefficient: f64,
    weight_coefficient: f64,
}

enum SamplingMode {
    FreeForAll,
    InterSpecies
}

pub enum NeatPhase<T> {
    Sampling (SamplingMode, Vec<T>),
    Activation (Vec<f64>),
    Evaluation
}



pub trait Evaluator {
    fn get_input(&self) -> Vec<f64>;
    fn process_output(&self, outputs: Vec<f64>);
    fn get_fitness(&self) -> Vec<f64>;
}

impl Population {
    pub fn add_genome_to_species(&mut self, settings: &Settings, distance_threshold: f64, genome: Genome) {
        let species_index =
            self.species.iter().position(|species| genome.distance(&species.representative, settings.excess_coefficient, settings.disjoint_coefficient, settings.weight_coefficient) < distance_threshold);

        match species_index {
            Some(index) => {
                self.species[index].members.push(genome);
            }
            None => {
                self.species.push(Species {
                    members: vec![genome.clone()],
                    representative: genome,
                    champion: 0,
                    avg_fitness: 0.0
                });
            }
        }
    }

    pub fn init(rng: &mut dyn RngCore, n_sensor_nodes: usize, n_output_nodes: usize) -> Population {
        let mut population = Population {
            species: Vec::new(),
            generation: 0
        };

        let settings = Settings {
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_coefficient: 0.4
        };

        for _ in 0..100 {
            let genome = Genome::init(rng, n_sensor_nodes, n_output_nodes);
            population.add_genome_to_species(&settings, 3.0, genome);
        }

        population
    }

    pub fn evaluate<E: Evaluator>(&mut self, evaluator: &E) {

    }
}
