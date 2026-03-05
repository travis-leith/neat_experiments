#[cfg(test)]
mod tests {
    use crate::neat::evolution::{EvolutionCheckpoint, EvolutionConfig, MatchConfig};
    use crate::neat::genome::innovation::InnovationTracker;
    use crate::neat::genome::mutation::Mutation;
    use crate::neat::genome::types::{
        ConnectionGene, ConnectionKey, DistanceCoefficients, Genome, GenomeError, Innovation,
        NodeId, NodeKind, ParentFitness,
    };
    use crate::neat::phenome::ActivationConfig;
    use crate::neat::population::ReproductionConfig;
    use crate::neat::species::{SpeciationConfig, Species, SpeciesId};

    fn round_trip_json<T: serde::Serialize + serde::de::DeserializeOwned>(value: &T) -> T {
        let json = serde_json::to_string(value).expect("serialization failed");
        serde_json::from_str(&json).expect("deserialization failed")
    }

    fn assert_round_trip_json<
        T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug,
    >(
        value: &T,
    ) -> T {
        let json = serde_json::to_string_pretty(value).expect("serialization failed");
        let restored: T = serde_json::from_str(&json).expect("deserialization failed");
        // Re-serialize to verify structural equivalence
        let json2 = serde_json::to_string_pretty(&restored).expect("re-serialization failed");
        assert_eq!(json, json2, "round-trip produced different JSON");
        restored
    }

    fn sample_tracker_and_genome() -> (InnovationTracker, Genome) {
        let mut tracker = InnovationTracker::new();
        let genome = Genome::minimal_fully_connected(2, 1, &mut tracker, |_i, _o| 0.5);
        (tracker, genome)
    }

    fn sample_genome_with_hidden_node() -> (InnovationTracker, Genome) {
        let (mut tracker, genome) = sample_tracker_and_genome();
        let split = genome.innovations().next().unwrap();
        let genome = genome.with_added_node(&mut tracker, split).unwrap();
        (tracker, genome)
    }

    // -- NodeId --

    #[test]
    fn node_id_round_trip() {
        assert_round_trip_json(&NodeId(0));
        assert_round_trip_json(&NodeId(42));
        assert_round_trip_json(&NodeId(u32::MAX));
    }

    // -- Innovation --

    #[test]
    fn innovation_round_trip() {
        assert_round_trip_json(&Innovation(0));
        assert_round_trip_json(&Innovation(999));
    }

    // -- NodeKind --

    #[test]
    fn node_kind_round_trip() {
        assert_round_trip_json(&NodeKind::Sensor);
        assert_round_trip_json(&NodeKind::Hidden);
        assert_round_trip_json(&NodeKind::Output);
    }

    // -- ConnectionKey --

    #[test]
    fn connection_key_round_trip() {
        let key = ConnectionKey {
            in_node: NodeId(0),
            out_node: NodeId(3),
        };
        assert_round_trip_json(&key);
    }

    #[test]
    fn connection_key_string_round_trip() {
        use std::str::FromStr;
        let key = ConnectionKey {
            in_node: NodeId(7),
            out_node: NodeId(42),
        };
        let s = key.to_string();
        let restored = ConnectionKey::from_str(&s).unwrap();
        assert_eq!(key, restored);
    }

    // -- ConnectionGene --

    #[test]
    fn connection_gene_round_trip() {
        let gene = ConnectionGene {
            key: ConnectionKey {
                in_node: NodeId(0),
                out_node: NodeId(2),
            },
            innovation: Innovation(1),
            weight: -0.375,
            enabled: true,
        };
        assert_round_trip_json(&gene);
    }

    // -- DistanceCoefficients --

    #[test]
    fn distance_coefficients_round_trip() {
        assert_round_trip_json(&DistanceCoefficients::default());
        assert_round_trip_json(&DistanceCoefficients {
            excess: 2.0,
            disjoint: 3.0,
            weight: 0.1,
            small_genome_threshold: 5,
        });
    }

    // -- GenomeError --

    #[test]
    fn genome_error_round_trip() {
        let errors: Vec<GenomeError> = vec![
            GenomeError::MissingNode(NodeId(5)),
            GenomeError::InvalidOutputNode(NodeId(0)),
            GenomeError::SelfLoop(NodeId(3)),
            GenomeError::DuplicateConnection(ConnectionKey {
                in_node: NodeId(0),
                out_node: NodeId(2),
            }),
            GenomeError::UnknownInnovation(Innovation(99)),
            GenomeError::ConnectionAlreadyDisabled(Innovation(1)),
            GenomeError::MismatchedIo {
                left_inputs: 2,
                left_outputs: 1,
                right_inputs: 3,
                right_outputs: 2,
            },
            GenomeError::MismatchedNodeKind {
                node: NodeId(0),
                left: NodeKind::Sensor,
                right: NodeKind::Hidden,
            },
        ];
        for err in &errors {
            assert_round_trip_json(err);
        }
    }

    // -- ParentFitness --

    #[test]
    fn parent_fitness_round_trip() {
        assert_round_trip_json(&ParentFitness::Left);
        assert_round_trip_json(&ParentFitness::Right);
        assert_round_trip_json(&ParentFitness::Equal);
    }

    // -- Genome (minimal) --

    #[test]
    fn genome_minimal_round_trip() {
        let (_tracker, genome) = sample_tracker_and_genome();
        assert_round_trip_json(&genome);
    }

    // -- Genome (with hidden node and extra connections) --

    #[test]
    fn genome_with_hidden_node_round_trip() {
        let (_tracker, genome) = sample_genome_with_hidden_node();
        assert_round_trip_json(&genome);
    }

    #[test]
    fn genome_with_mutations_round_trip() {
        let (mut tracker, genome) = sample_tracker_and_genome();
        let first = genome.innovations().next().unwrap();
        let mutated = genome
            .apply_mutations(
                &mut tracker,
                &[
                    Mutation::PerturbWeight {
                        innovation: first,
                        delta: 0.3,
                    },
                    Mutation::AddNode {
                        split_innovation: first,
                    },
                    Mutation::AddConnection {
                        in_node: NodeId(1),
                        out_node: NodeId(3),
                        weight: -0.7,
                    },
                ],
            )
            .unwrap();

        let restored = assert_round_trip_json(&mutated);
        assert_eq!(restored.connection_count(), mutated.connection_count());
        assert_eq!(restored.nodes.len(), mutated.nodes.len());
    }

    // -- InnovationTracker --

    #[test]
    fn innovation_tracker_round_trip() {
        let mut tracker = InnovationTracker::new();
        let _ = Genome::minimal_fully_connected(3, 2, &mut tracker, |_, _| 0.0);
        let _ = tracker.next_hidden_node_id();
        let _ = tracker.next_connection_innovation();
        assert_round_trip_json(&tracker);
    }

    // -- SpeciationConfig --

    // #[test]
    // fn speciation_config_round_trip() {
    //     assert_round_trip_json(&SpeciationConfig::default());
    //     assert_round_trip_json(&SpeciationConfig {
    //         compatibility_threshold: 0.1,
    //         distance_coefficients: DistanceCoefficients {
    //             excess: 2.0,
    //             disjoint: 3.0,
    //             weight: 0.5,
    //             small_genome_threshold: 10,
    //         },
    //         stagnation_limit: 20,
    //     });
    // }

    // -- SpeciesId --

    #[test]
    fn species_id_round_trip() {
        assert_round_trip_json(&SpeciesId(0));
        assert_round_trip_json(&SpeciesId(42));
    }

    // -- Species --

    #[test]
    fn species_round_trip() {
        let (_tracker, genome) = sample_tracker_and_genome();
        let species = Species {
            id: SpeciesId(1),
            representative: genome,
            member_indices: vec![0, 3, 7],
            stagnation_counter: 5,
            best_fitness: 12.3,
        };
        assert_round_trip_json(&species);
    }

    // -- ReproductionConfig --

    #[test]
    fn reproduction_config_round_trip() {
        assert_round_trip_json(&ReproductionConfig::default());
    }

    // -- ActivationConfig --

    #[test]
    fn activation_config_round_trip() {
        assert_round_trip_json(&ActivationConfig::default());
        assert_round_trip_json(&ActivationConfig {
            recurrent_iterations: 50,
            recurrent_epsilon: 1e-6,
            logging_enabled: true,
        });
    }

    // -- EvolutionConfig --

    #[test]
    fn evolution_config_round_trip() {
        assert_round_trip_json(&EvolutionConfig::default());
    }

    // -- MatchConfig --

    #[test]
    fn match_config_round_trip() {
        assert_round_trip_json(&MatchConfig::default());
        assert_round_trip_json(&MatchConfig {
            players_per_match: 4,
            matches_per_organism: 20,
        });
    }

    // -- EvolutionCheckpoint (the full integration test) --

    #[test]
    fn evolution_checkpoint_round_trip() {
        let mut tracker = InnovationTracker::new();
        let mut r = rand::thread_rng();
        let genomes = Genome::random_fully_connected_population(5, 3, 2, &mut tracker, &mut r);

        let species = vec![Species {
            id: SpeciesId(1),
            representative: genomes[0].clone(),
            member_indices: vec![0, 1, 2],
            stagnation_counter: 3,
            best_fitness: 8.5,
        }];

        let checkpoint = EvolutionCheckpoint {
            config: EvolutionConfig::default(),
            match_config: MatchConfig::default(),
            tracker,
            genomes,
            species,
            next_species_id: 2,
            generation: 10,
            rng_seed_state: vec![42u8; 32],
        };

        assert_round_trip_json(&checkpoint);
    }

    // -- Genome connection_to_innovation map specifically --

    #[test]
    fn genome_connection_to_innovation_map_survives_round_trip() {
        let (mut tracker, genome) = sample_tracker_and_genome();
        let split = genome.innovations().next().unwrap();
        let genome = genome.with_added_node(&mut tracker, split).unwrap();
        let genome = genome
            .with_added_connection(&mut tracker, NodeId(1), NodeId(3), 0.5)
            .unwrap();

        let restored = assert_round_trip_json(&genome);

        // Verify every connection key maps to the same innovation
        for (key, innov) in &genome.connection_to_innovation {
            let restored_innov = restored
                .connection_to_innovation
                .get(key)
                .expect("key missing after round trip");
            assert_eq!(innov, restored_innov);
        }
        assert_eq!(
            genome.connection_to_innovation.len(),
            restored.connection_to_innovation.len()
        );
    }
}
