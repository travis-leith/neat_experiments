use super::innovation::InnovationTracker;
use super::types::{ConnectionGene, ConnectionKey, Genome, Innovation, NodeId, NodeKind};
use rand::{Rng, RngExt};
use std::collections::{BTreeMap, HashMap};

fn build_nodes(input_nodes: &[NodeId], output_nodes: &[NodeId]) -> BTreeMap<NodeId, NodeKind> {
    let sensors = input_nodes
        .iter()
        .copied()
        .map(|id| (id, NodeKind::Sensor))
        .collect::<BTreeMap<_, _>>();

    let outputs = output_nodes
        .iter()
        .copied()
        .map(|id| (id, NodeKind::Output))
        .collect::<BTreeMap<_, _>>();

    let mut nodes = sensors;
    nodes.extend(outputs);
    nodes
}

fn fully_connected_topology(
    tracker: &mut InnovationTracker,
    input_nodes: &[NodeId],
    output_nodes: &[NodeId],
) -> Vec<(ConnectionKey, Innovation)> {
    output_nodes
        .iter()
        .copied()
        .flat_map(|out_node| {
            input_nodes
                .iter()
                .copied()
                .map(move |in_node| ConnectionKey { in_node, out_node })
        })
        .map(|key| (key, tracker.next_connection_innovation()))
        .collect::<Vec<_>>()
}

fn build_genome_from_topology<F>(
    n_inputs: usize,
    n_outputs: usize,
    nodes: &BTreeMap<NodeId, NodeKind>,
    topology: &[(ConnectionKey, Innovation)],
    mut weight_of: F,
) -> Genome
where
    F: FnMut(NodeId, NodeId) -> f64,
{
    let mut g = Genome {
        n_inputs,
        n_outputs,
        nodes: nodes.clone(),
        connections_by_innovation: BTreeMap::new(),
        connection_to_innovation: HashMap::new(),
    };

    topology.iter().for_each(|(key, innovation)| {
        let gene = ConnectionGene {
            key: key.clone(),
            innovation: *innovation,
            weight: weight_of(key.in_node, key.out_node),
            enabled: true,
        };
        g.insert_connection_gene(gene);
    });

    g
}

impl Genome {
    pub fn minimal_fully_connected<F>(
        n_inputs: usize,
        n_outputs: usize,
        tracker: &mut InnovationTracker,
        mut initial_weight: F,
    ) -> Self
    where
        F: FnMut(NodeId, NodeId) -> f64,
    {
        let (input_nodes, output_nodes) = tracker.io_nodes(n_inputs, n_outputs);
        let nodes = build_nodes(&input_nodes, &output_nodes);
        let topology = fully_connected_topology(tracker, &input_nodes, &output_nodes);

        build_genome_from_topology(n_inputs, n_outputs, &nodes, &topology, |i, o| {
            initial_weight(i, o)
        })
    }

    pub fn random_fully_connected_population<R: Rng>(
        n: usize,
        n_inputs: usize,
        n_outputs: usize,
        tracker: &mut InnovationTracker,
        rng: &mut R,
    ) -> Vec<Self> {
        let (input_nodes, output_nodes) = tracker.io_nodes(n_inputs, n_outputs);
        let nodes = build_nodes(&input_nodes, &output_nodes);
        let topology = fully_connected_topology(tracker, &input_nodes, &output_nodes);

        (0..n)
            .map(|_| {
                build_genome_from_topology(n_inputs, n_outputs, &nodes, &topology, |_i, _o| {
                    rng.random_range(-1.0..1.0)
                })
            })
            .collect::<Vec<_>>()
    }
}
