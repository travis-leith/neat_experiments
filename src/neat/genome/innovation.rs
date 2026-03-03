use super::types::{Innovation, NodeId};

#[derive(Debug, Clone)]
pub struct InnovationTracker {
    next_innovation: u64,
    next_node_id: u32,
}

impl InnovationTracker {
    pub fn new(start_node_id: NodeId) -> Self {
        Self {
            next_innovation: 1,
            next_node_id: start_node_id.0,
        }
    }

    pub fn next_connection_innovation(&mut self) -> Innovation {
        let id = Innovation(self.next_innovation);
        self.next_innovation += 1;
        id
    }

    pub fn next_hidden_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }
}
