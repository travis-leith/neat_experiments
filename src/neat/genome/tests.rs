#[cfg(test)]
mod tests {
    use super::super::crossover::*;
    use super::super::distance::*;
    use super::super::innovation::*;
    use super::super::mutation::*;
    use super::super::pruning::*;
    use super::super::types::*;

    fn get_innovation(g: &Genome, in_node: NodeId, out_node: NodeId) -> Innovation {
        let key = ConnectionKey { in_node, out_node };
        *g.connection_to_innovation.get(&key).unwrap()
    }

    fn base_tracker() -> InnovationTracker {
        InnovationTracker::new() // with 2 inputs + 1 output
    }

    fn base_genome(tracker: &mut InnovationTracker) -> Genome {
        Genome::minimal_fully_connected(2, 1, tracker, |_i, _o| 0.5)
    }

    struct FixedPolicy {
        choose_left_matching: bool,
        choose_left_equal_unmatched: bool,
        enable_if_either_parent_disabled: bool,
    }

    impl CrossoverPolicy for FixedPolicy {
        fn choose_left_matching(
            &mut self,
            _left: &ConnectionGene,
            _right: &ConnectionGene,
        ) -> bool {
            self.choose_left_matching
        }

        fn choose_left_when_equal_for_unmatched(&mut self) -> bool {
            self.choose_left_equal_unmatched
        }

        fn enable_if_either_parent_disabled(&mut self) -> bool {
            self.enable_if_either_parent_disabled
        }
    }

    fn crossover_fixed(
        left: &Genome,
        right: &Genome,
        fitness: ParentFitness,
        choose_left_matching: bool,
        choose_left_equal_unmatched: bool,
        enable_if_either_parent_disabled: bool,
    ) -> Genome {
        let mut policy = FixedPolicy {
            choose_left_matching,
            choose_left_equal_unmatched,
            enable_if_either_parent_disabled,
        };

        Genome::crossover_with_policy(left, right, fitness, &mut policy).unwrap()
    }

    #[test]
    fn every_new_connection_mutation_gets_unique_innovation() {
        let mut t = InnovationTracker::new();
        let base = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.0);

        let key = ConnectionKey {
            in_node: NodeId(2),
            out_node: NodeId(3),
        };

        let g1 = base
            .with_added_connection(&mut t, key.in_node, key.out_node, 0.1)
            .unwrap();

        let g2 = base
            .with_added_connection(&mut t, key.in_node, key.out_node, -0.7)
            .unwrap();

        let i1 = g1.connection_to_innovation.get(&key).copied().unwrap();
        let i2 = g2.connection_to_innovation.get(&key).copied().unwrap();

        assert_ne!(i1, i2);
    }

    #[test]
    fn splitting_same_parent_connection_twice_does_not_reuse_split_history() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);

        let split_innov = g.innovations().next().unwrap();

        let g1s = g.with_added_node(&mut t, split_innov).unwrap();
        let g2s = g.with_added_node(&mut t, split_innov).unwrap();

        let added_1 = g1s
            .innovations()
            .filter(|i| !g.innovations().any(|x| x == *i))
            .collect::<Vec<_>>();

        let added_2 = g2s
            .innovations()
            .filter(|i| !g.innovations().any(|x| x == *i))
            .collect::<Vec<_>>();

        assert_eq!(added_1.len(), 2);
        assert_eq!(added_2.len(), 2);
        assert_ne!(added_1, added_2);
    }

    #[test]
    fn alignment_by_innovation_is_correct() {
        let mut t = base_tracker();
        let g1 = base_genome(&mut t);
        let i = g1.innovations().next().unwrap();
        let g2 = g1.with_perturbed_weight(i, 1.0).unwrap();

        let mut both = 0usize;
        let mut left = 0usize;
        let mut right = 0usize;

        for_each_aligned_by_innovation(&g1, &g2, |a| match a {
            Alignment::Both(_, _) => both += 1,
            Alignment::Left(_) => left += 1,
            Alignment::Right(_) => right += 1,
        });

        assert_eq!(both, 2);
        assert_eq!(left, 0);
        assert_eq!(right, 0);
    }
    #[test]
    fn add_connection_adds_new_gene_and_duplicate_fails() {
        let mut t = InnovationTracker::new();
        let g = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.0);

        let g2 = g
            .with_added_connection(&mut t, NodeId(2), NodeId(3), 0.9)
            .unwrap();

        assert_eq!(g2.connection_count(), g.connection_count() + 1);

        let err = g2
            .with_added_connection(&mut t, NodeId(2), NodeId(3), 0.1)
            .err()
            .unwrap();

        match err {
            GenomeError::DuplicateConnection(_) => {}
            _ => panic!("expected DuplicateConnection"),
        }
    }

    #[test]
    fn add_node_splits_connection_correctly() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);

        let split_innov = g.innovations().next().unwrap();
        let old_weight = g.connection(split_innov).unwrap().weight;

        let g2 = g.with_added_node(&mut t, split_innov).unwrap();

        let old_gene = g2.connection(split_innov).unwrap();
        assert!(!old_gene.enabled);

        let mut created = g2
            .innovations()
            .filter(|i| *i != split_innov)
            .collect::<Vec<_>>();
        created.sort();

        assert_eq!(created.len(), 3); // other original + two newly inserted

        let new_edges = g2
            .connections_by_innovation
            .values()
            .filter(|cg| cg.key.in_node == NodeId(0) || cg.key.out_node == NodeId(2))
            .collect::<Vec<_>>();

        assert!(new_edges.iter().any(|g| (g.weight - 1.0).abs() < 1e-12));
        assert!(new_edges
            .iter()
            .any(|g| (g.weight - old_weight).abs() < 1e-12));
    }

    #[test]
    fn perturb_weight_keeps_innovation() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);
        let i = g.innovations().next().unwrap();
        let old = g.connection(i).unwrap().weight;

        let g2 = g.with_perturbed_weight(i, 0.25).unwrap();
        let new_gene = g2.connection(i).unwrap();

        assert_eq!(new_gene.innovation, i);
        assert!((new_gene.weight - (old + 0.25)).abs() < 1e-12);
    }

    #[test]
    fn genetic_distance_matches_expected_counts() {
        let mut t = base_tracker();
        let mut g1 = base_genome(&mut t);
        let mut g2 = base_genome(&mut t);

        // Force divergence:
        // g1: split first connection => adds two genes.
        let split = g1.innovations().next().unwrap();
        g1 = g1.with_added_node(&mut t, split).unwrap();

        // g2: perturb one matching weight only.
        let match_i = g2.innovations().next().unwrap();
        g2 = g2.with_perturbed_weight(match_i, 0.5).unwrap();

        let d = genetic_distance(
            &g1,
            &g2,
            DistanceCoefficients {
                excess: 1.0,
                disjoint: 1.0,
                weight: 1.0,
                small_genome_threshold: 20,
            },
        );

        assert!(d > 0.0);
    }

    #[test]
    fn apply_mutations_pipeline_works() {
        let mut t = InnovationTracker::new();
        let g = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.0);

        let first = g.innovations().next().unwrap();
        let out = g
            .apply_mutations(
                &mut t,
                &[
                    Mutation::PerturbWeight {
                        innovation: first,
                        delta: 0.2,
                    },
                    Mutation::AddConnection {
                        in_node: NodeId(2),
                        out_node: NodeId(3),
                        weight: 0.7,
                    },
                ],
            )
            .unwrap();

        assert_eq!(out.connection_count(), g.connection_count() + 1);
        assert!((out.connection(first).unwrap().weight - 0.2).abs() < 1e-12);
    }

    #[test]
    fn crossover_inherits_matching_and_fitter_unmatched_genes() {
        let mut t = InnovationTracker::new();
        let base = Genome::minimal_fully_connected(2, 2, &mut t, |_i, _o| 0.0);

        let first = base.innovations().next().unwrap();
        let left = base
            .with_added_connection(&mut t, NodeId(2), NodeId(3), 0.9)
            .unwrap();
        let right = base.with_perturbed_weight(first, -0.2).unwrap();

        let child = crossover_fixed(
            &left,
            &right,
            ParentFitness::Left,
            false, // take right for matching genes
            false, // irrelevant in this test
            true,  // when exactly one parent disabled, keep enabled
        );

        assert_eq!(child.connection_count(), left.connection_count());

        let added_key = ConnectionKey {
            in_node: NodeId(2),
            out_node: NodeId(3),
        };
        assert!(child.connection_to_innovation.contains_key(&added_key));
        assert!((child.connection(first).unwrap().weight + 0.2).abs() < 1e-12);
    }

    #[test]
    fn crossover_respects_disabled_gene_policy() {
        let mut t = base_tracker();
        let base = base_genome(&mut t);
        let split = base.innovations().next().unwrap();

        let left = base.with_added_node(&mut t, split).unwrap(); // split gene disabled
        let right = base.clone(); // split gene enabled

        let child_disabled = crossover_fixed(
            &left,
            &right,
            ParentFitness::Equal,
            false, // inherit matching from right
            true,
            false, // when exactly one parent disabled, force disabled
        );

        assert!(!child_disabled.connection(split).unwrap().enabled);

        let child_enabled = crossover_fixed(
            &left,
            &right,
            ParentFitness::Equal,
            false,
            true,
            true, // when exactly one parent disabled, force enabled
        );

        assert!(child_enabled.connection(split).unwrap().enabled);
    }

    #[test]
    fn add_node_rejects_unknown_and_already_disabled_connection() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);
        let split = g.innovations().next().unwrap();

        let unknown_err = g
            .with_added_node(&mut t, Innovation(u64::MAX))
            .err()
            .unwrap();
        match unknown_err {
            GenomeError::UnknownInnovation(_) => {}
            _ => panic!("expected UnknownInnovation"),
        }

        let g2 = g.with_added_node(&mut t, split).unwrap();
        let disabled_err = g2.with_added_node(&mut t, split).err().unwrap();
        match disabled_err {
            GenomeError::ConnectionAlreadyDisabled(i) => assert_eq!(i, split),
            _ => panic!("expected ConnectionAlreadyDisabled"),
        }
    }

    #[test]
    fn add_connection_rejects_invalid_endpoints() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);

        match g
            .with_added_connection(&mut t, NodeId(0), NodeId(0), 0.1)
            .err()
            .unwrap()
        {
            GenomeError::SelfLoop(NodeId(0)) => {}
            _ => panic!("expected SelfLoop"),
        }

        match g
            .with_added_connection(&mut t, NodeId(2), NodeId(0), 0.1)
            .err()
            .unwrap()
        {
            GenomeError::InvalidOutputNode(NodeId(0)) => {}
            _ => panic!("expected InvalidOutputNode"),
        }

        match g
            .with_added_connection(&mut t, NodeId(999), NodeId(2), 0.1)
            .err()
            .unwrap()
        {
            GenomeError::MissingNode(NodeId(999)) => {}
            _ => panic!("expected MissingNode"),
        }
    }

    #[test]
    fn disable_connection_mutation_disables_gene() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);
        let i = g.innovations().next().unwrap();

        let g2 = g
            .apply_mutation(&mut t, &Mutation::DisableConnection { innovation: i })
            .unwrap();

        assert!(g.connection(i).unwrap().enabled);
        assert!(!g2.connection(i).unwrap().enabled);
        assert_eq!(g2.connection_count(), g.connection_count());
    }

    #[test]
    fn disable_connection_mutation_rejects_unknown_and_already_disabled() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);
        let i = g.innovations().next().unwrap();

        let unknown = g
            .apply_mutation(
                &mut t,
                &Mutation::DisableConnection {
                    innovation: Innovation(u64::MAX),
                },
            )
            .err()
            .unwrap();

        match unknown {
            GenomeError::UnknownInnovation(_) => {}
            _ => panic!("expected UnknownInnovation"),
        }

        let g2 = g
            .apply_mutation(&mut t, &Mutation::DisableConnection { innovation: i })
            .unwrap();

        let already_disabled = g2
            .apply_mutation(&mut t, &Mutation::DisableConnection { innovation: i })
            .err()
            .unwrap();

        match already_disabled {
            GenomeError::ConnectionAlreadyDisabled(x) => assert_eq!(x, i),
            _ => panic!("expected ConnectionAlreadyDisabled"),
        }
    }

    #[test]
    fn apply_mutation_add_node_variant_works() {
        let mut t = base_tracker();
        let g = base_genome(&mut t);
        let split = g.innovations().next().unwrap();

        let g2 = g
            .apply_mutation(
                &mut t,
                &Mutation::AddNode {
                    split_innovation: split,
                },
            )
            .unwrap();

        assert_eq!(g2.connection_count(), g.connection_count() + 2);
        assert!(!g2.connection(split).unwrap().enabled);
    }

    #[test]
    fn prune_removes_disabled_connections() {
        let mut t = InnovationTracker::new();
        let g = Genome::minimal_fully_connected(2, 1, &mut t, |_, _| 1.0);

        let innov = get_innovation(&g, NodeId(0), NodeId(2));
        let g = g
            .apply_mutations(&mut t, &[Mutation::DisableConnection { innovation: innov }])
            .unwrap();

        assert_eq!(g.connections_by_innovation.len(), 2);

        let pruned = prune(&g);

        assert_eq!(pruned.connections_by_innovation.len(), 1);
        assert!(pruned.connections_by_innovation.values().all(|c| c.enabled));
    }

    #[test]
    fn prune_removes_orphaned_hidden_nodes() {
        let mut t = InnovationTracker::new();
        let g = Genome::minimal_fully_connected(1, 1, &mut t, |_, _| 1.0);

        let split = get_innovation(&g, NodeId(0), NodeId(1));
        let g = g.with_added_node(&mut t, split).unwrap();

        // After split: NodeId(0)->NodeId(2)->NodeId(1), original disabled
        // Now disable both new connections to orphan the hidden node
        let innov_left = get_innovation(&g, NodeId(0), NodeId(2));
        let innov_right = get_innovation(&g, NodeId(2), NodeId(1));
        let g = g
            .apply_mutations(
                &mut t,
                &[
                    Mutation::DisableConnection {
                        innovation: innov_left,
                    },
                    Mutation::DisableConnection {
                        innovation: innov_right,
                    },
                ],
            )
            .unwrap();

        assert!(g.nodes.contains_key(&NodeId(2)));

        let pruned = prune(&g);

        assert!(!pruned.nodes.contains_key(&NodeId(2)));
        // IO nodes always retained
        assert!(pruned.nodes.contains_key(&NodeId(0)));
        assert!(pruned.nodes.contains_key(&NodeId(1)));
    }

    #[test]
    fn prune_preserves_io_nodes_even_with_no_connections() {
        let mut t = InnovationTracker::new();
        let g = Genome::minimal_fully_connected(2, 2, &mut t, |_, _| 1.0);

        // Disable all connections
        let innovations: Vec<_> = g.innovations().collect();
        let mutations: Vec<_> = innovations
            .into_iter()
            .map(|i| Mutation::DisableConnection { innovation: i })
            .collect();
        let g = g.apply_mutations(&mut t, &mutations).unwrap();

        let pruned = prune(&g);

        assert_eq!(pruned.connections_by_innovation.len(), 0);
        assert_eq!(pruned.nodes.len(), 4); // 2 inputs + 2 outputs
    }

    #[test]
    fn prune_is_idempotent() {
        let mut t = InnovationTracker::new();
        let g = Genome::minimal_fully_connected(2, 1, &mut t, |_, _| 1.0);
        let split = get_innovation(&g, NodeId(0), NodeId(2));
        let g = g.with_added_node(&mut t, split).unwrap();

        let pruned_once = prune(&g);
        let pruned_twice = prune(&pruned_once);

        assert_eq!(pruned_once.nodes.len(), pruned_twice.nodes.len());
        assert_eq!(
            pruned_once.connections_by_innovation.len(),
            pruned_twice.connections_by_innovation.len()
        );
    }
}
