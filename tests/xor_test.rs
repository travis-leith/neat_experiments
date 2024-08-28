extern crate neat_experiments;

#[cfg(test)]
mod test {
    use neat_experiments::neat::{common::Settings, organism::Organism, population::SinglePlayerArena};
    use rand::SeedableRng;
    // use rand::seq::SliceRandom;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use neat_experiments::neat::population::Population;

    //create xor evaluator
    struct XorEvaluator;

    impl SinglePlayerArena for XorEvaluator {
        fn generate_inputs(&self) -> Vec<Vec<f64>> {
            vec![vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0], vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 1.0]]
        }

        fn evaluate_outputs(&self, outputs: Vec<Vec<f64>>) -> usize {
            let expected_outputs = vec![0.0, 1.0, 1.0, 0.0];
            let mut acc = 0.0;
            for (i, output) in outputs.iter().enumerate() {
                acc += (output[0] - expected_outputs[i]).powi(2);
            }
            let fitness = (4.0 - acc) * 10000.0;
            fitness.max(0.0) as usize
        }
    }

    fn check_evaluation(organism: &mut Organism) {
        let inputs = vec![vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0], vec![1.0, 0.0, 1.0], vec![1.0, 1.0, 1.0]];
        for i in 0..inputs.len() {
            // organism.clear_values();
            let output = organism.activate(&inputs[i]);
            println!("in: {:?}; out: {:?}", inputs[i], output);
        }
    }
    fn describe_population_demographics(population: &Population) {
        println!("generation: {:?}; n_species: {:?}", population.generation, population.species.len());
        // for (i, s) in population.species.iter().enumerate() {
        //     println!("\tspecies: {:?}", i);
        //     println!("\tnumber of members: {:?}", s.members.len());
        //     println!("");
        // }
    }

    fn describe_population_fitness(population: &Population) {
        // println!("generation: {:?}; n_species: {:?}", population.generation, population.species.len());
        let avg_fitness = population.species.iter().map(|s| s.avg_fitness).sum::<f64>() / population.species.len() as f64;
        let max_fitness = population.organisms.iter().map(|o| o.fitness).fold(0, |acc, x| acc.max(x));
        // for (i, s) in population.species.iter().enumerate() {
        //     println!("\tspecies: {:?}", i);
        //     println!("\tchampion fitness: {:?}", population.organisms[s.champion].fitness);
        //     println!("\taverage fitness: {:?}", s.avg_fitness);
        //     println!("");
        // }
        println!("avg fitness: {:?}; max fitness: {:?}", avg_fitness, max_fitness);
    }

    fn get_solution_organism(population: &Population) -> Option<&Organism> {
        for s in population.species.iter() {
            let champion = &population.organisms[s.champion];
            if champion.fitness > 39900 {
                return Some(champion);
            }
        }
        None
    }

    #[test]
    fn test_xor() {
        let mut settings = Settings::standard();
        settings.n_organisms = 1000;
    
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);
        let mut population = Population::init(&mut rng, &settings);

        let mut evaluator = XorEvaluator;

        describe_population_demographics(&population);
        population.evaluate(&mut evaluator, true);        
        describe_population_fitness(&population);

        for _ in 0..500 {
            population.next_generation(&mut rng, &settings);
            describe_population_demographics(&population);
            // population.evaluate(&mut evaluator);
            population.evaluate_single_player(&mut evaluator, true);
            describe_population_fitness(&population);

            if let Some(solution_org) = get_solution_organism(&population) {
                println!("solution found");
                for (gene_key, gene_value) in solution_org.genome.iter() {
                    if gene_value.enabled {
                        println!("{:?}---|{:.4}|{:?}", gene_key.in_node_id.0, gene_value.weight, gene_key.out_node_id.0);
                    }
                }
                println!("activation order");
                for &node_index in &solution_org.activation_order {
                    let node = &solution_org.phenome[node_index];
                    node.inputs.iter().for_each(|gene_index| {
                        let (gene_key, gene_value) = solution_org.genome.get_index(*gene_index);
                        println!("{:?}---|{:.4}|{:?}", gene_key.in_node_id.0, gene_value.weight, gene_key.out_node_id.0);
                    });
                }

                let mut cloned_org = Organism::create_from_genome(solution_org.genome.clone());
                check_evaluation(&mut cloned_org);

                
                break;
            }
        }
        // println!("random seed: {:?}", random_seed);
    }
}

