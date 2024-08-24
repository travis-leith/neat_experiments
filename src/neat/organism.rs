use rand::RngCore;
use rand_distr::{Distribution, Uniform};

use crate::neat::genome::Genome;

use super::network::Network;

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

    pub fn create_from_genome(n_sensor_nodes: usize, n_output_nodes: usize, genome: Genome) -> Organism {
        Organism {
            network: Network::create_from_genome(n_sensor_nodes, n_output_nodes, genome),
            fitness: 0.
        }
    }
    pub fn activate(&mut self, sensor_values: &Vec<f64>) -> Vec<f64> {
        self.network.activate(sensor_values)
    }
}
