use std::ops::Index;
use std::ops::IndexMut;
use super::genome::GeneIndex;
#[derive(PartialEq, Default)]
pub enum NodeType{
    Sensor,
    #[default]
    Hidden,
    Output,
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct NodeIndex(pub usize);

#[derive(Default)]
pub struct Node {
    pub value: f64,
    // pub is_active: bool,
    // pub has_active_inputs: bool,
    pub inputs: Vec<GeneIndex>,
    // pub active_sum: f64,
    pub node_type: NodeType,
}

impl Node {
    fn create(node_type: NodeType) -> Node {
        Node{
            value: 0.,
            // is_active: node_type == NodeType::Sensor,
            // has_active_inputs: false,
            inputs: Vec::new(),
            // active_sum: 0.,
            node_type,
        }
    }
}

pub struct Phenome(Vec<Node>);
impl Phenome {
    pub fn create_disconnected(n_sensor_nodes: usize, n_output_nodes: usize, max_node_id: usize) -> Phenome {
        let hidden_start = n_sensor_nodes + n_output_nodes;
        let nodes: Vec<Node> = (0 .. max_node_id + 1).map(|i:usize|{
            if i < n_sensor_nodes {
                Node::create(NodeType::Sensor)
            } else if i < hidden_start {
                Node::create(NodeType::Output)
            } else {
                Node::create(NodeType::Hidden)
            }
        }).collect();
        Phenome(nodes)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<Node> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Node> {
        self.0.iter_mut()
    }
}

impl Index<NodeIndex> for Phenome {
    type Output = Node;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

impl IndexMut<NodeIndex> for Phenome {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}