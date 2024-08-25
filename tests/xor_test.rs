extern crate neat_experiments;

#[cfg(test)]
mod test {
    use neat_experiments::neat::{common::Settings, organism::Organism, population::Evaluator};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use neat_experiments::neat::population::Population;

    //create xor evaluator
    struct XorEvaluator;
    impl Evaluator for XorEvaluator {
        fn evaluate_single_organism(&self, organism: &mut Organism) {
            let inputs = vec![vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0], vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 1.0]];
            let expected_outputs = vec![0.0, 1.0, 1.0, 0.0];
            let mut acc = 0.0;
            for i in 0..inputs.len() {
                let output = organism.activate(&inputs[i]);
                acc += (output[0] - expected_outputs[i]).powi(2);
            }
            organism.fitness = 4.0 - acc
        }
    }


    #[test]
    fn test_xor() {
        let settings = Settings {
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_coefficient: 0.4,
            n_organisms: 100,
            n_sensor_nodes: 3,
            n_output_nodes: 1,
            mutate_weight_rate: 0.2,
            mutate_add_connection_rate: 0.2,
            mutate_add_node_rate: 0.2,
            mutate_weight_scale: 0.1,
        };
    
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let mut population = Population::init(&mut rng, &settings);

        for i in 0..5 {
            println!("number of species: {:?}", population.species.len());
            population.evaluate(&XorEvaluator);
            println!("generation: {:?}", i);

            for s in population.species.iter_mut() {
                println!("number of members: {:?}", s.members.len());
                println!("champion fitness: {:?}", s.members[s.champion].fitness);
                println!("average fitness: {:?}", s.avg_fitness);
            }

            population.next_generation(&mut rng, &settings);
            println!("");
        }
        
        population.evaluate(&XorEvaluator);
        println!("generation: {:?}", 5);

        for s in population.species.iter_mut() {
            println!("number of members: {:?}", s.members.len());
            println!("champion fitness: {:?}", s.members[s.champion].fitness);
            println!("average fitness: {:?}", s.avg_fitness);
        }

    }
}

