use std::io::{self, BufRead, stdout};
use rand::seq::SliceRandom;
use crate::tictactoe::game::*;
use crate::tictactoe::display::board_to_string;

pub fn get_random_move(gameboard: &GameBoard) -> CellLocation {
    let make_move = |m:CellLocation| {
        match gameboard.get_cell(m) {
            Cell(None) => Some(m),
            _ => None
        }
    };

    let possible_moves: [Option<CellLocation>; 9] = [
        make_move(CellLocation::TopLft),
        make_move(CellLocation::TopMid),
        make_move(CellLocation::TopRgt),
        make_move(CellLocation::MidLft),
        make_move(CellLocation::MidMid),
        make_move(CellLocation::MidRgt),
        make_move(CellLocation::BotLft),
        make_move(CellLocation::BotMid),
        make_move(CellLocation::BotRgt),
    ];

    let possible_moves: Vec<CellLocation> = possible_moves.iter().flatten().cloned().collect();
    let mut rng = rand::thread_rng();
    possible_moves.choose(&mut rng).cloned().unwrap_or(CellLocation::BotLft)
}

fn get_valid_input_from_user<T>(f:fn() -> Option<T>) -> T {
    loop {
        match f() {
            Some(result) => return result,
            None => println!("You did not enter a valid response. Try again.")
        }
    }
}

fn read_line_from_stdin() -> io::Result<String> {
    let stdin = io::stdin();
    stdin.lock().lines().next().unwrap_or_else(|| Ok(String::new()))
}

fn get_yes_no_from_user(message: &str) -> Option<bool> {
    println!("Yes or No\n{message}");
    let line = read_line_from_stdin().unwrap_or_default();

    match line.to_lowercase().as_str() {
        "yes" | "y"   => Some(true),
        "no" | "n"   => Some(false),
        _ => None
    }
}

fn get_first_player_from_user() -> bool {
    let f = || get_yes_no_from_user("Do you want to play first?");
    get_valid_input_from_user(f)
}

fn start_game() -> PlayingGameState {
    let player_turn = 
        if get_first_player_from_user() {
            Player::Cross
        } else {
            Player::Circle
        };

    // let gameboard = empty_game_board();
    // PlayingGameState { gameboard, player_turn }
    new_game(player_turn)
}

fn string_to_player_move(s: String) -> Option<CellLocation> {
    match s.to_lowercase().as_str() {
        "q" => Some(CellLocation::TopLft),
        "w" => Some(CellLocation::TopMid),
        "e" => Some(CellLocation::TopRgt),
        "a" => Some(CellLocation::MidLft),
        "s" => Some(CellLocation::MidMid),
        "d" => Some(CellLocation::MidRgt),
        "z" => Some(CellLocation::BotLft),
        "x" => Some(CellLocation::BotMid),
        "c" => Some(CellLocation::BotRgt),
        _   => None
    }
}

fn maybe_get_user_move() -> Option<CellLocation> {
    println!("Select move from qweasdzxc");
    let line = read_line_from_stdin().unwrap_or_default();

    string_to_player_move(line)
}
use crossterm::{execute, terminal::{Clear, ClearType}};

fn clear_screen() {
    match execute!(stdout(), Clear(ClearType::Purge)) {
        Ok(_) => (),
        Err(e) => eprintln!("Error clearing screen: {e}")
    }
}

pub fn get_user_move(gameboard: &GameBoard) -> CellLocation {
    
    clear_screen();
    let s = board_to_string(gameboard);
    println!("{s}");
    let f = || maybe_get_user_move();
    get_valid_input_from_user(f)
}

fn ask_user_for_new_game() -> bool {
    let f = || get_yes_no_from_user("Do you want to play again?");
    get_valid_input_from_user(f)
}

fn end_game(gameboard: &GameBoard, game_over_state: &GameOverState) {
    clear_screen();
    let s = board_to_string(gameboard);
    println!("{s}");

    match game_over_state {
        GameOverState::Tied => println!("No winners - game is tied!"),
        GameOverState::Won(Player::Cross) => println!("User (you) have won!"),
        GameOverState::Won(Player::Circle) => println!("AI has won!"),
        GameOverState::Disqualified(Player::Cross, m) => println!("User (you) have tried to play an illegal move ({:?}) and are disqualified!", m),
        GameOverState::Disqualified(Player::Circle, m) => println!("AI tried to play an illegal move ({:?}) and is disqualified!", m)
    }
}

fn single_game_loop(ctrl: &mut impl Controller, mut playing_state: PlayingGameState) -> (GameBoard, GameOverState) {
    loop {
        match play_game( ctrl, playing_state) {
            Ok(game_over_state) => 
                break game_over_state,
            Err(new_playing_state) => {
                println!("you have entered an invalid move. Try again ...");
                playing_state = new_playing_state;
            }
        }
    }
}

pub fn game_loop(ctrl: &mut impl Controller) {
    loop {
        let gamestate = start_game();

        let (gameboard, game_over_state) = single_game_loop(ctrl, gamestate);
        end_game(&gameboard, &game_over_state);

        if !ask_user_for_new_game() {
            break
        } else {
            ctrl.reset();
        }
    }
}
