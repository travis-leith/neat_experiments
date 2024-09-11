
use rustc_hash::FxHashMap;
use super::genome::{GeneKey, NodeId};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct InnovationNumber(pub usize);

impl InnovationNumber {
    fn inc(mut self) -> InnovationNumber {
        self.0 += 1;
        self
    }
}

pub struct InnovationContext {
    pub next_innovation_number: InnovationNumber,
    pub innovation_map: FxHashMap<GeneKey, InnovationNumber>,
}

impl InnovationContext {
    pub fn get_innovation_number(&mut self, gene_key: GeneKey) -> InnovationNumber {
        match self.innovation_map.try_insert(gene_key, self.next_innovation_number) {
            Ok(i) => {
                self.next_innovation_number = self.next_innovation_number.inc();
                *i
            },
            Err(i) => *i.entry.get()
        }
    }

    pub fn init(n_sensor_nodes: usize, n_output_nodes: usize) -> InnovationContext {
        let mut innovation_map = FxHashMap::default();
        let mut innovation_number = InnovationNumber(0);

        for i in 0..n_sensor_nodes {
            for j in 0..n_output_nodes {
                let in_node_id = NodeId(i);
                let out_node_id = NodeId(j + n_sensor_nodes);
                let gene_key = GeneKey{in_node_id, out_node_id};
                innovation_map.insert(gene_key, innovation_number);
                innovation_number = innovation_number.inc();
            }
        }

        InnovationContext {
            next_innovation_number: innovation_number,
            innovation_map
        }
    }
}