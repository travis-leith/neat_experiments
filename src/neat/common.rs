pub struct Settings {
    pub excess_coefficient: f64,
    pub disjoint_coefficient: f64,
    pub weight_coefficient: f64,
    pub n_organisms: usize,
    pub n_sensor_nodes: usize,
    pub n_output_nodes: usize,
    pub mutate_add_connection_rate: f64,
    pub mutate_add_node_rate: f64,
    pub mutate_weight_rate: f64,
    pub mutate_weight_scale: f64,
    pub n_species_min: usize,
    pub n_species_max: usize,
    //TODO add more settings
    //inter_species_mating_rate
    //intra_species_mating_rate
    
}

impl Settings {
    pub fn standard() -> Settings {
        Settings {
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_coefficient: 0.4,
            n_organisms: 100,
            n_sensor_nodes: 3,
            n_output_nodes: 1,
            mutate_weight_rate: 0.2,
            mutate_add_connection_rate: 0.2,
            mutate_add_node_rate: 0.2,
            mutate_weight_scale: 0.1,
            n_species_min: 5,
            n_species_max: 10
        }
    }
}