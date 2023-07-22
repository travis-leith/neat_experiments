use std::io::{self, BufRead};
use rand::seq::SliceRandom;
use crate::tictactoe::game::*;
use crate::tictactoe::display::board_to_string;


fn get_valid_input_from_user<T>(f:fn() -> Option<T>) -> T {
    loop {
        match f() {
            Some(result) => return result,
            None => println!("You did not enter a valid response. Try again.")
        }
    }
}

fn get_yes_no_from_user(message: &str) -> Option<bool> {
    println!("Yes or No\n{message}");
    let stdin = io::stdin();
    let mut iterator = stdin.lock().lines();
    let line1 = iterator.next().unwrap().unwrap();

    match line1.to_lowercase().as_str() {
        "yes" => Some(true),
        "y"   => Some(true),
        "no"  => Some(false),
        "n"   => Some(false),
        _     => None
    }
}

fn get_first_player_from_user() -> bool {
    let f = || get_yes_no_from_user("Do you want to play first?");
    get_valid_input_from_user(f)
}

fn start_game() -> GameState {
    let player_turn = 
        if get_first_player_from_user() {
            Player::Cross
        } else {
            Player::Circle
        };

    let gameboard = empty_game_board();
    GameState::Playing(PlayingGameState { gameboard, player_turn })
}

fn get_random_move(gameboard: &GameBoard) -> Option<PlayerMove> {
    fn make_move(x: &Cell, m:PlayerMove) -> Option<PlayerMove>{
        match x.0 {
            None => Some(m),
            _    => None
        }
    }

    let possible_moves: [Option<PlayerMove>; 9] = [
        make_move(&gameboard.top_lft, PlayerMove::TopLft),
        make_move(&gameboard.top_mid, PlayerMove::TopMid),
        make_move(&gameboard.top_rgt, PlayerMove::TopRgt),
        make_move(&gameboard.mid_lft, PlayerMove::MidLft),
        make_move(&gameboard.mid_mid, PlayerMove::MidMid),
        make_move(&gameboard.mid_rgt, PlayerMove::MidRgt),
        make_move(&gameboard.bot_lft, PlayerMove::BotLft),
        make_move(&gameboard.bot_mid, PlayerMove::BotMid),
        make_move(&gameboard.bot_rgt, PlayerMove::BotRgt),
    ];

    let possible_moves: Vec<PlayerMove> = possible_moves.iter().flatten().cloned().collect();
    let mut rng = rand::thread_rng();
    possible_moves.choose(&mut rng).copied()
}

fn get_ai_move(gameboard: &GameBoard) -> PlayerMove {
    match get_random_move(gameboard) {
        Some(player_move) => player_move,
        None => PlayerMove::TopLft
    }
}

fn string_to_player_move(s: String) -> Option<PlayerMove> {
    match s.to_lowercase().as_str() {
        "q" => Some(PlayerMove::TopLft),
        "w" => Some(PlayerMove::TopMid),
        "e" => Some(PlayerMove::TopRgt),
        "a" => Some(PlayerMove::MidLft),
        "s" => Some(PlayerMove::MidMid),
        "d" => Some(PlayerMove::MidRgt),
        "z" => Some(PlayerMove::BotLft),
        "x" => Some(PlayerMove::BotMid),
        "c" => Some(PlayerMove::BotRgt),
        _   => None
    }
}

fn maybe_get_user_move() -> Option<PlayerMove> {
    println!("Select move from qweasdzxc");
    let stdin = io::stdin();
    let mut iterator = stdin.lock().lines();
    let line1 = iterator.next().unwrap().unwrap();

    string_to_player_move(line1)
}

fn get_user_move(_: &GameBoard) -> PlayerMove {
    let f = || maybe_get_user_move();
    get_valid_input_from_user(f)
}

fn ask_user_for_new_game() -> bool {
    let f = || get_yes_no_from_user("Do you want to play again?");
    get_valid_input_from_user(f)
}

fn end_game(game_over_state: &GameOverState) {
    match game_over_state {
        GameOverState::Tied => println!("No winners - game is tied!"),
        GameOverState::Won(Player::Cross) => println!("User (you) have won!"),
        GameOverState::Won(Player::Circle) => println!("AI has won!")
    }
}

fn clear_screen() {
    print!("{}[2J", 27 as char);
}

fn display_game(gamestate: &GameState) {
    clear_screen();
    match gamestate {
        GameState::Starting => println!("New game is starting"),
        GameState::Playing(playing_state) => {
            let s = board_to_string(&playing_state.gameboard);
            println!("{s}");
        },
        GameState::GameOver(game_over_state) => end_game(&game_over_state)
    }
}

impl GameState {
    fn not_exiting(&self) -> bool {
        match &self {
            GameState::GameOver(_) => false,
            _ => true
        }
    }
}
fn inner_game_loop() {
    let mut gamestate = GameState::Starting;

    display_game(&gamestate);
    while gamestate.not_exiting() {
        transition_gamestate(&mut gamestate, start_game, get_user_move, get_ai_move);
        display_game(&gamestate);
    }
}

pub fn game_loop() {
    let mut play_a_game = true;
    while play_a_game {
        inner_game_loop();
        play_a_game = ask_user_for_new_game()
    }
}