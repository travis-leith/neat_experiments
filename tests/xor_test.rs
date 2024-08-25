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

    fn describe_population(population: &Population) {
        println!("generation: {:?}; n_species: {:?}", population.generation, population.species.len());
        for (i, s) in population.species.iter().enumerate() {
            println!("\tspecies: {:?}", i);
            println!("\tnumber of members: {:?}", s.members.len());
            println!("\tchampion fitness: {:?}", population.organisms[s.champion].fitness);
            println!("\taverage fitness: {:?}", s.avg_fitness);
            println!("");
        }
        println!("");
        println!("");
    }

    #[test]
    fn test_xor() {
        let mut settings = Settings::standard();
        settings.n_organisms = 1000;
    
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let mut population = Population::init(&mut rng, &settings);

        population.evaluate(&XorEvaluator);        
        describe_population(&population);

        for _ in 0..100 {
            population.next_generation(&mut rng, &settings);

            population.evaluate(&XorEvaluator);
            describe_population(&population);
        }

    }
}

