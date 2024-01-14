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
    use CellLocation::*;
    let tl = &board.get_cell(TopLft);
    let tm = &board.get_cell(TopMid);
    let tr = &board.get_cell(TopRgt);
    let ml = &board.get_cell(MidLft);
    let mm = &board.get_cell(MidMid);
    let mr = &board.get_cell(MidRgt);
    let bl = &board.get_cell(BotLft);
    let bm = &board.get_cell(BotMid);
    let br = &board.get_cell(BotRgt);
    format!{"{tl} {tm} {tr}\n{ml} {mm} {mr}\n{bl} {bm} {br}"}
}