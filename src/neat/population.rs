use super::{common::Settings, genome, innovation::InnovationContext, organism::{Organism, Organisms}, genome::Genome};
use rand::{RngCore, Rng};

#[derive(Clone, Copy)]
pub struct OrganismIndex(pub usize);

pub struct Species {
    pub members: Vec<OrganismIndex>,
    pub representative: Genome,
    pub champion: OrganismIndex,
    pub avg_fitness: f64
}

pub struct Population {
    pub species: Vec<Species>,
    pub organisms: Organisms,
    species_distance_threshold: f64,
    pub generation: usize,
    innovation_context: InnovationContext
}

pub trait Evaluator {
    fn evaluate_single_organism(&self, organism: &mut Organism); //TODO make this more structured by modelling inputs and outputs
}

impl Population {
    pub fn add_organism(&mut self, organism: Organism) {
        self.organisms.push(organism);        
    }

    fn assign_species(&mut self, settings: &Settings, organism_index: OrganismIndex) {
        let organism = &self.organisms[organism_index];

        let species_index =
            self.species.iter().position(|species| organism.network.genome.distance(&species.representative, settings.excess_coefficient, settings.disjoint_coefficient, settings.weight_coefficient) < self.species_distance_threshold);

        match species_index {
            Some(index) => {
                self.species[index].members.push(organism_index);
            }
            None => {
                let new_species = Species {
                    members: vec![organism_index],
                    representative: organism.network.genome.clone(),
                    champion: organism_index,
                    avg_fitness: 0.0
                };
                self.species.push(new_species);
            }
        }
    }

    fn speciate(&mut self, settings: &Settings) {
        let max_loops = 10;
        for _ in 0..max_loops {
            for s in self.species.iter_mut() {
                s.members.clear();
            }

            for i in 0..self.organisms.len() {
                self.assign_species(settings, OrganismIndex(i));
            }

            let n_non_empty_species = self.species.iter().filter(|s| !s.members.is_empty()).count();
            if n_non_empty_species < settings.n_species_min {
                self.species_distance_threshold *= 0.9;
            } else if n_non_empty_species > settings.n_species_max {
                self.species_distance_threshold *= 1.1;
            } else {
                break;
            }
            
        }

        //remove empty species
        self.species.retain(|s| !s.members.is_empty());
    }

    pub fn init<R: RngCore>(rng: &mut R, settings: &Settings) -> Population {
        let innovation_context = InnovationContext::init(settings.n_sensor_nodes, settings.n_output_nodes);
        let mut organisms = Vec::new();

        for _ in 0..settings.n_organisms {
            let organism = Organism::init(rng, settings.n_sensor_nodes, settings.n_output_nodes);
            organisms.push(organism);
        }

        let mut res = Population {
            species: Vec::new(),
            organisms: Organisms::new(organisms),
            species_distance_threshold: 0.3,
            generation: 0,
            innovation_context
        };

        res.speciate(settings);
        res
    }

    pub fn evaluate<E: Evaluator>(&mut self, evaluator: &E) {
        for s in self.species.iter_mut() {
            let mut total_species_fitness = 0.0;
            let mut champion = OrganismIndex(0);
            let mut champion_fitness = 0.0;

            for &org_index in &s.members {
                let org = &mut self.organisms[org_index];
                evaluator.evaluate_single_organism(org);
                total_species_fitness += org.fitness;
                if org.fitness > champion_fitness {
                    champion_fitness = org.fitness;
                    champion = org_index;
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
                let parent_1_index = s.members[rng.gen_range(0..s.members.len())];
                let parent_2_index = s.members[rng.gen_range(0..s.members.len())];
                let parent_1 = &self.organisms[parent_1_index];
                let parent_2 = &self.organisms[parent_2_index];
                let mut child_genome = genome::cross_over(rng, &parent_1.network.genome, parent_1.fitness, &parent_2.network.genome, parent_2.fitness);
                child_genome.mutate(rng, &mut self.innovation_context, settings);
                let child = Organism::create_from_genome(child_genome);
                new_population.push(child);
            }

            if n_offspring > 0 {
                let champion = self.organisms[s.champion].clone();
                new_population.push(champion);
            }
        }

        self.organisms = Organisms::new(new_population);
        self.speciate(settings)
    }
}

#[cfg(test)]
mod tests {
    use crate::neat::{common::Settings, population::Population};

    #[test]
    fn test_init() {
        let settings = Settings::standard();
        let mut rng = rand::thread_rng();
        let population = Population::init(&mut rng, &settings);
        assert_eq!(population.innovation_context.innovation_map.len(), settings.n_sensor_nodes * settings.n_output_nodes);
        assert_eq!(population.innovation_context.next_innovation_number.0, settings.n_sensor_nodes * settings.n_output_nodes);
    }
}