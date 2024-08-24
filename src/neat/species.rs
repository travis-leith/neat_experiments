

use super::{genome::Genome, organism::Organism};

pub struct Species {
    pub members: Vec<Organism>,
    pub representative: Genome,
    pub champion: usize,
    pub avg_fitness: f64
}





// enum SamplingMode {
//     FreeForAll,
//     InterSpecies
// }

// pub enum NeatPhase<T> {
//     Sampling (SamplingMode, Vec<T>),
//     Activation (Vec<f64>),
//     Evaluation
// }



// pub trait Evaluator {
//     fn get_input(&self) -> Vec<f64>;
//     fn process_output(&self, outputs: Vec<f64>);
//     fn get_fitness(&self) -> Vec<f64>;
// }


