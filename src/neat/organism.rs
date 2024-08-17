use rand::RngCore;
use rand_distr::{Distribution, Uniform};

use crate::neat::genome::{self, Gene, GeneIndex};

use super::network::Network;

pub struct Organism {
    pub network: Network,
    pub fitness: usize
}

impl Organism {
    pub fn init(rng: &mut dyn RngCore , n_sensor_nodes: usize, n_output_nodes: usize) -> Organism {
        Organism { 
            network: Network::init(rng, n_sensor_nodes, n_output_nodes),
            fitness: 0
         }
    }

    pub fn activate(&mut self, sensor_values: Vec<f64>) {
        self.network.activate(sensor_values)
    }
}


use super::vector::{AllignedPair, allign};

pub fn cross_over(rng: &mut dyn RngCore, organism_1: &Organism, organism_2: &Organism) -> Organism {
   
    let new_network = super::network::cross_over(rng, &organism_1.network, organism_1.fitness, &organism_2.network, organism_2.fitness);
    Organism {
        network: new_network,
        fitness: 0
    }
}