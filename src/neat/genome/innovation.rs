use super::types::{Innovation, NodeId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IoLayout {
    n_inputs: usize,
    n_outputs: usize,
    input_nodes: Vec<NodeId>,
    output_nodes: Vec<NodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovationTracker {
    next_innovation: u64,
    next_node_id: u32,
    io_layout: Option<IoLayout>,
}

impl InnovationTracker {
    pub fn new() -> Self {
        Self {
            next_innovation: 1,
            next_node_id: 0,
            io_layout: None,
        }
    }

    pub fn io_nodes(&mut self, n_inputs: usize, n_outputs: usize) -> (Vec<NodeId>, Vec<NodeId>) {
        if let Some(layout) = &self.io_layout {
            assert!(
                layout.n_inputs == n_inputs && layout.n_outputs == n_outputs,
                "InnovationTracker was already initialized with different IO sizes"
            );
            return (layout.input_nodes.clone(), layout.output_nodes.clone());
        }

        let total_io = (n_inputs + n_outputs) as u32;

        if self.next_node_id < total_io {
            self.next_node_id = total_io;
        }

        let input_nodes = (0..n_inputs).map(|i| NodeId(i as u32)).collect::<Vec<_>>();
        let output_nodes = (0..n_outputs)
            .map(|o| NodeId((n_inputs + o) as u32))
            .collect::<Vec<_>>();

        self.io_layout = Some(IoLayout {
            n_inputs,
            n_outputs,
            input_nodes: input_nodes.clone(),
            output_nodes: output_nodes.clone(),
        });

        (input_nodes, output_nodes)
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

    /// Allocate `n` non-overlapping child trackers for parallel reproduction.
    /// Each child gets a disjoint range of innovation numbers and node IDs.
    /// The budget should be a generous upper bound on how many innovations/nodes
    /// a single partition will need.
    pub fn fork(&mut self, n: usize, innovation_budget: u64, node_budget: u32) -> Vec<Self> {
        let children = (0..n)
            .map(|i| {
                let i = i as u64;
                InnovationTracker {
                    next_innovation: self.next_innovation + i * innovation_budget,
                    next_node_id: self.next_node_id + (i as u32) * node_budget,
                    io_layout: self.io_layout.clone(),
                }
            })
            .collect();

        // Advance the parent past all allocated ranges
        self.next_innovation += (n as u64) * innovation_budget;
        self.next_node_id += (n as u32) * node_budget;

        children
    }

    /// Merge a child tracker back, advancing the parent to at least the child's position.
    pub fn join(&mut self, child: Self) {
        self.next_innovation = self.next_innovation.max(child.next_innovation);
        self.next_node_id = self.next_node_id.max(child.next_node_id);
    }
}
