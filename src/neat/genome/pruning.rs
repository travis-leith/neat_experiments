use super::types::{Genome, NodeId, NodeKind};
use std::collections::BTreeSet;

fn referenced_nodes(genome: &Genome) -> BTreeSet<NodeId> {
    genome
        .connections_by_innovation
        .values()
        .filter(|c| c.enabled)
        .flat_map(|c| [c.key.in_node, c.key.out_node])
        .collect()
}

fn io_nodes(genome: &Genome) -> BTreeSet<NodeId> {
    genome
        .nodes
        .iter()
        .filter(|(_, kind)| matches!(kind, NodeKind::Sensor | NodeKind::Output))
        .map(|(id, _)| *id)
        .collect()
}

fn retained_nodes(genome: &Genome) -> BTreeSet<NodeId> {
    let referenced = referenced_nodes(genome);
    let io = io_nodes(genome);
    &referenced | &io
}

/// Returns a new genome with all disabled connections removed
/// and all hidden nodes not referenced by any enabled connection removed.
/// Sensor and Output nodes are always retained.
pub fn prune(genome: &Genome) -> Genome {
    let keep_nodes = retained_nodes(genome);

    let nodes = genome
        .nodes
        .iter()
        .filter(|(id, _)| keep_nodes.contains(id))
        .map(|(id, kind)| (*id, *kind))
        .collect();

    let mut pruned = Genome {
        n_inputs: genome.n_inputs,
        n_outputs: genome.n_outputs,
        nodes,
        connections_by_innovation: std::collections::BTreeMap::new(),
        connection_to_innovation: std::collections::HashMap::new(),
    };

    for gene in genome.connections_by_innovation.values() {
        if gene.enabled {
            pruned.insert_connection_gene(gene.clone());
        }
    }

    pruned
}

/// Prune all genomes in a population, returning the pruned population.
pub fn prune_population(genomes: &[Genome]) -> Vec<Genome> {
    genomes.iter().map(prune).collect()
}
