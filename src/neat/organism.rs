use rand::RngCore;
use rand_distr::{Distribution, Uniform};

use crate::neat::genome::{self, Gene, GeneIndex};

use super::network::Network;

pub struct Organism {
    pub network: Network,
    pub fitness: f64
}

impl Organism {
    pub fn init(rng: &mut dyn RngCore , n_sensor_nodes: usize, n_output_nodes: usize) -> Organism {
        Organism { 
            network: Network::init(rng, n_sensor_nodes, n_output_nodes),
            fitness: 0.
         }
    }

    pub fn activate(&mut self, sensor_values: Vec<f64>) {
        self.network.activate(sensor_values)
    }
}
