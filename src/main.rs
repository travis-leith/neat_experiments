mod tictactoe;
mod neat;
use crate::tictactoe::cli::game_loop;
// use crate::neat::common::*;
// use crate::tictactoe::game::*;

// impl Cell {
//     fn as_f64(self) -> f64 {
//         match self.0 {
//             Some(Player::Cross) => -1.,
//             Some(Player::Circle) => -1.,
//             None => 0.
//         }
//     }
// }

// impl GameBoard {
//     fn as_sensor_values(self) -> Vec<f64> {
//         vec![
//             self.top_lft.as_f64(),
//             self.top_mid.as_f64(),
//             self.top_rgt.as_f64(),
//             self.mid_lft.as_f64(),
//             self.mid_mid.as_f64(),
//             self.mid_rgt.as_f64(),
//             self.bot_lft.as_f64(),
//             self.bot_mid.as_f64(),
//             self.bot_rgt.as_f64()
//         ]
//     }
// }

fn main() {
    // let population_size = 100;

    // let organisms : Vec<Organism> = (0 .. population_size).map(|_|Organism::init(9, 9)).collect();

    // fn get_ai_move(organism: &Organism, gameboard: &GameBoard) -> PlayerMove {
    //     let sensor_values = gameboard.as_sensor_values();
    //     let activated_organism = organism.activate(sensor_values);
        

    // }
    game_loop();
}