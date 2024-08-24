
use rustc_hash::{FxHashMap};
use super::{genome::GeneKey, phenome::NodeIndex};

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
            Err(i) => i.entry.get().clone()
        }
    }
}