use super::innovation::InnovationTracker;
use super::types::{ConnectionGene, ConnectionKey, Genome, NodeId, NodeKind};
use std::collections::{BTreeMap, HashMap};

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
        let sensors = (0..n_inputs)
            .map(|i| (NodeId(i as u32), NodeKind::Sensor))
            .collect::<BTreeMap<_, _>>();

        let outputs = (0..n_outputs)
            .map(|o| (NodeId((n_inputs + o) as u32), NodeKind::Output))
            .collect::<BTreeMap<_, _>>();

        let mut nodes = sensors;
        nodes.extend(outputs);

        let mut g = Self {
            n_inputs,
            n_outputs,
            nodes,
            connections_by_innovation: BTreeMap::new(),
            connection_to_innovation: HashMap::new(),
        };

        for out_idx in 0..n_outputs {
            let out_node = NodeId((n_inputs + out_idx) as u32);
            for in_idx in 0..n_inputs {
                let in_node = NodeId(in_idx as u32);
                let weight = initial_weight(in_node, out_node);
                let key = ConnectionKey { in_node, out_node };
                let innovation = tracker.next_connection_innovation();
                let gene = ConnectionGene {
                    key: key.clone(),
                    innovation,
                    weight,
                    enabled: true,
                };
                g.insert_connection_gene(gene);
            }
        }

        g
    }
}
