mod tictactoe;
mod neat;
use crate::tictactoe::cli::game_loop;
use crate::tictactoe::cli::get_user_move;
use crate::neat::common::*;
use crate::tictactoe::game::*;


impl Cell {
    fn as_f64(self) -> f64 {
        match self.0 {
            Some(Player::Cross) => -1.,
            Some(Player::Circle) => -1.,
            None => 0.
        }
    }
}

impl GameBoard {
    fn as_sensor_values(&self) -> Vec<f64> {
        self.cells.into_iter().map(|c|c.as_f64()).collect()
    }
}

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

struct InitNetworkAiVsUser{
    network: Network
}
impl Controller for InitNetworkAiVsUser {
    fn retry_allowed(&mut self) -> bool {
        false
    }

    fn circle_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        self.network.activate(gameboard.as_sensor_values());
        let network_output = self.network.get_output();
        let index_of_max = 
            network_output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap_or(0)
                ;

        let res = CellLocation::from_usize(index_of_max).unwrap_or(CellLocation::BotLft);
        println!("AI has selected move: {:?}", res);
        res
    }

    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        get_user_move(gameboard)
    }
}

struct RandomAiVsUser;
impl Controller for RandomAiVsUser {
    fn retry_allowed(&mut self) -> bool {
        false
    }

    fn circle_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        tictactoe::cli::get_random_move(gameboard)
    }

    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation {
        get_user_move(gameboard)
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let simple_ai = Network::init(&mut rng, 9, 9);
    let mut ai_controller = InitNetworkAiVsUser {network:simple_ai};
    // let population_size = 100;

    // let organisms : Vec<Organism> = (0 .. population_size).map(|_|Organism::init(9, 9)).collect();

    // fn get_ai_move(organism: &Organism, gameboard: &GameBoard) -> PlayerMove {
    //     let sensor_values = gameboard.as_sensor_values();
    //     let activated_organism = organism.activate(sensor_values);
        

    // }
    game_loop(&mut ai_controller);
}