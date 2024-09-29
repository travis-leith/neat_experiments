use std::collections::VecDeque;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use serde::{Serialize, Deserialize};
use itertools::Itertools;

use crate::neat::graph::tarjan_scc;

use super::genome::Genome;
use super::genome::NodeId;

#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum NodeType{
    Sensor,
    Hidden,
    Output,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Phenome{
    pub activation_order: Vec<(usize, Vec<(usize, f64)>)>,
    pub node_values: Vec<f64>,
    pub node_ids: Vec<NodeId>,
    pub node_indices: FxHashMap<NodeId, usize>,
    pub inputs: Vec<Option<usize>>,
    pub outputs: Vec<Option<usize>>,
    n_sensor_nodes: usize,
    n_output_nodes: usize,
}

impl Phenome {

    pub fn create_from_genome(genome: &Genome) -> Phenome {
        struct MapEntry {
            inputs: Vec<(NodeId, f64)>,
            outputs: Vec<(NodeId, f64)>
        }

        let mut node_map: FxHashMap<NodeId, MapEntry> = FxHashMap::default();

        for (gene_key, gene_val) in genome.iter() {
            let in_node_id = gene_key.in_node_id;
            let out_node_id = gene_key.out_node_id;
            let weight = gene_val.weight;

            let entry = node_map.entry(out_node_id).or_insert(MapEntry{inputs: Vec::new(), outputs: Vec::new()});
            entry.inputs.push((in_node_id, weight));

            let entry = node_map.entry(in_node_id).or_insert(MapEntry{inputs: Vec::new(), outputs: Vec::new()});
            entry.outputs.push((out_node_id, weight));
        }

        // Function to perform BFS and return all reachable nodes from a given set of start nodes
        fn bfs_reachable(nodes: &FxHashMap<NodeId, MapEntry>, initial_nodes: &[NodeId], next_getter: impl Fn(&MapEntry) -> Vec<(NodeId, f64)>) -> FxHashSet<NodeId> {
            let mut visited = FxHashSet::default();
            let mut queue = VecDeque::new();

            for &start in initial_nodes {
                queue.push_back(start);
                visited.insert(start);
            }

            while let Some(node_index) = queue.pop_front() {
                let node_entry = nodes.get(&node_index).unwrap();
                let next_nodes = next_getter(node_entry);
                for (new_index, _) in next_nodes {
                    if visited.insert(new_index) {
                        queue.push_back(new_index);
                    }
                }
            }

            visited
        }

        let sensor_node_ids = (0..genome.n_sensor_nodes).map(NodeId).collect_vec();
        let output_node_ids = (genome.n_sensor_nodes..genome.n_sensor_nodes + genome.n_output_nodes).map(NodeId).collect_vec();

        let get_inputs = |entry: &MapEntry| entry.inputs.clone();
        let get_outputs = |entry: &MapEntry| entry.outputs.clone();

        let reachable_from_inputs = bfs_reachable(&node_map, &output_node_ids, get_inputs);
        let reachable_to_outputs = bfs_reachable(&node_map, &sensor_node_ids, get_outputs);

        let mut node_id_to_index: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut node_ids: Vec<NodeId> = Vec::new();

        for (index, node_id) in reachable_from_inputs.intersection(&reachable_to_outputs).cloned().enumerate() {
            node_id_to_index.insert(node_id, index);
            node_ids.push(node_id);
        }

        let enumerated_graph: Vec<Vec<(usize, f64)>> = 
            node_ids.iter().map(|node_id| {
                let entry = node_map.get(node_id).unwrap();
                let inputs = entry.inputs.iter().filter_map(|(in_node_id, weight)| {
                    match node_id_to_index.get(in_node_id) {
                        Some(&in_index) => Some((in_index, *weight)),
                        None => None
                    }
                }).collect_vec();
                inputs
            }).collect_vec();

        let topo_order = tarjan_scc(&enumerated_graph);
        let activation_order = topo_order.iter().flat_map(|component| {
            component.iter().filter_map(|&node_index| {
                let inputs = &enumerated_graph[node_index];
                if !inputs.is_empty() {
                    Some((node_index, inputs.clone()))
                } else {
                    None
                }
            })
        }).collect_vec();

        let inputs = sensor_node_ids.iter().map(|node_id| node_id_to_index.get(node_id).cloned()).collect_vec();
        let outputs = output_node_ids.iter().map(|node_id| node_id_to_index.get(node_id).cloned()).collect_vec();
        Phenome {
            activation_order,
            node_values: vec![0.; node_ids.len()],
            node_ids,
            node_indices: node_id_to_index,
            inputs,
            outputs,
            n_sensor_nodes: genome.n_sensor_nodes,
            n_output_nodes: genome.n_output_nodes
        }
    }

    pub fn activate(&mut self, sensor_values: &[f64]) -> Vec<f64> {
        fn relu(x: f64) -> f64 {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }

        // fn sigmoid(x: f64) -> f64 {
        //     1.0 / (1.0 + (-4.9 * x).exp())
        // }

        for (i, &input) in sensor_values.iter().enumerate() {
            match self.inputs[i] {
                Some(node_index) => {
                    self.node_values[node_index] = input;
                },
                None => {}
            }
        }

        for &(node_index, ref inputs) in &self.activation_order {
            let active_sum = inputs.iter().fold(0., |acc, &(input_index, w)| {
                acc + w * self.node_values[input_index]
            });
            self.node_values[node_index] = relu(active_sum);
        }

        self.outputs.iter().map(|node_index_opt| {
            match node_index_opt {
                Some(node_index) => self.node_values[*node_index],
                None => 0.
            }
        } ).collect()
    }

    pub fn clear_values(&mut self) {
        for value in self.node_values.iter_mut() {
            *value = 0.;
        }
    }

    fn get_node_type(&self, node_id: NodeId) -> NodeType {
        if node_id.0 < self.n_sensor_nodes {
            NodeType::Sensor
        } else if node_id.0 < self.n_sensor_nodes + self.n_output_nodes {
            NodeType::Output
        } else {
            NodeType::Hidden
        }
    }

    pub fn print_mermaid_graph(&self) {
        println!("graph TD");
        

        for (node_index, node_id) in self.node_ids.iter().enumerate() {
            let node_type = match self.get_node_type(*node_id) {
                NodeType::Sensor => "S",
                NodeType::Hidden => "H",
                NodeType::Output => "O",
            };
            
            println!("{}[{}:{}/{}]", node_index, node_index, node_id.0, node_type);
        }

        for &(node_index, ref inputs) in &self.activation_order {
            for (input_index, weight) in inputs {
                println!("{} -->|{:.4}|{}", input_index, weight, node_index);
            }
        }
    }
}

mod tests {

    #[test]
    fn test_dead_ends() {
        use crate::neat::{genome::{Gene, GeneExt, Genome}, phenome::Phenome};
        let genome = 
            Genome::create(vec![
                Gene::create(0, 4, 0.0),
                Gene::create(4, 2, 0.0),
                Gene::create(7, 6, 0.0),
                Gene::create(4, 6, 0.0),
                Gene::create(4, 8, 0.0),
                Gene::create(6, 5, 0.0),
                Gene::create(5, 4, 0.0),
                Gene::create(1, 5, 0.0),
                Gene::create(5, 3, 0.0),
                Gene::create(9, 7, 0.0),
                
            ], 2, 2);

        let phenome =  Phenome::create_from_genome(&genome);

        phenome.print_mermaid_graph();
    }
    
    
    
}