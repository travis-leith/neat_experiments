use std::ops::{Index, IndexMut};

use rand::RngCore;

use crate::neat::genome::Genome;

use super::{network::Network, population::OrganismIndex};

#[derive(Clone)]
pub struct Organism {
    pub network: Network,
    pub fitness: f64
}

impl Organism {
    pub fn init<R: RngCore>(rng: &mut R, n_sensor_nodes: usize, n_output_nodes: usize) -> Organism {
        Organism { 
            network: Network::init(rng, n_sensor_nodes, n_output_nodes),
            fitness: 0.
         }
    }

    pub fn create_from_genome(genome: Genome) -> Organism {
        Organism {
            network: Network::create_from_genome(genome),
            fitness: 0.
        }
    }
    pub fn activate(&mut self, sensor_values: &Vec<f64>) -> Vec<f64> {
        self.network.activate(sensor_values)
    }
}

pub struct Organisms(Vec<Organism>);

impl Organisms {
    pub fn push(&mut self, organism: Organism) {
        self.0.push(organism);
    }

    pub fn new(data: Vec<Organism>) -> Organisms {
        Organisms(data)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Index<OrganismIndex> for Organisms {
    type Output = Organism;
    fn index(&self, index: OrganismIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

impl IndexMut<OrganismIndex> for Organisms {
    fn index_mut(&mut self, index: OrganismIndex) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}