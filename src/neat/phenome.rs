use std::ops::Index;
use std::ops::IndexMut;
use super::genome::GeneIndex;

#[derive(PartialEq, Default, Clone, Debug)]
pub enum NodeType{
    Sensor,
    #[default]
    Hidden,
    Output,
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Hash, Eq)]
pub struct NodeIndex(pub usize);

impl NodeIndex {
    pub fn inc(self) -> NodeIndex {
        NodeIndex(self.0 + 1)
    }
}

#[derive(Default, Clone)]
pub struct Node {
    pub value: f64,
    pub inputs: Vec<GeneIndex>,
    pub node_type: NodeType,
}

impl Node {
    fn create(node_type: NodeType) -> Node {
        Node{
            value: 0.,
            inputs: Vec::new(),
            node_type,
        }
    }
}

#[derive(Clone)]
pub struct Phenome(Vec<Node>);
impl Phenome {
    pub fn create_disconnected(n_sensor_nodes: usize, n_output_nodes: usize, last_node_id: usize) -> Phenome {
        let hidden_start = n_sensor_nodes + n_output_nodes;
        let nodes: Vec<Node> = (0 .. last_node_id).map(|i:usize|{
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