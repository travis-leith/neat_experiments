// use rustc_hash::{FxHashMap};
use super::phenome::NodeIndex;

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
    // pub innovation_map: FxHashMap<(NodeIndex, NodeIndex), InnovationNumber>,
}