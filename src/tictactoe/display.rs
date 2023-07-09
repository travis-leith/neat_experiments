use std::fmt;
use crate::tictactoe::game::*;


impl fmt::Display for Cell {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn cell_to_char(cell: &Cell) -> char {
            match &cell.0 {
                Some(Player::Cross) => 'X',
                Some(Player::Circle) => 'O',
                None => '_',
            }
        }
        write!(f, "{}", cell_to_char(self))
    }
}

pub fn board_to_string(board: &GameBoard) -> String {
    let tl = &board.top_lft;
    let tm = &board.top_mid;
    let tr = &board.top_rgt;
    let ml = &board.mid_lft;
    let mm = &board.mid_mid;
    let mr = &board.mid_rgt;
    let bl = &board.bot_lft;
    let bm = &board.bot_mid;
    let br = &board.bot_rgt;
    format!{"{tl} {tm} {tr}\n{ml} {mm} {mr}\n{bl} {bm} {br}"}
}