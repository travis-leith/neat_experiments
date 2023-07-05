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
    let user_is_first = get_first_player_from_user();
    let gameboard = empty_game_board();
    GameState::Playing(PlayingGameState { gameboard, user_turn: user_is_first })
}

fn get_random_move(gameboard: &GameBoard) -> Option<UserMove> {
    fn make_move(x: &Cell, m:UserMove) -> Option<UserMove>{
        match x.0 {
            None => Some(m),
            _    => None
        }
    }

    let possible_moves: [Option<UserMove>; 9] = [
        make_move(&gameboard.top_lft, UserMove::TopLft),
        make_move(&gameboard.top_mid, UserMove::TopMid),
        make_move(&gameboard.top_rgt, UserMove::TopRgt),
        make_move(&gameboard.mid_lft, UserMove::MidLft),
        make_move(&gameboard.mid_mid, UserMove::MidMid),
        make_move(&gameboard.mid_rgt, UserMove::MidRgt),
        make_move(&gameboard.bot_lft, UserMove::BotLft),
        make_move(&gameboard.bot_mid, UserMove::BotMid),
        make_move(&gameboard.bot_rgt, UserMove::BotRgt),
    ];

    let possible_moves: Vec<UserMove> = possible_moves.iter().flatten().cloned().collect();
    let mut rng = rand::thread_rng();
    possible_moves.choose(&mut rng).copied()
}

fn do_ai_move(gameboard: &mut GameBoard) -> bool {
    match get_random_move(gameboard) {
       Some(user_move) => {
           apply_move(gameboard, user_move, false)
       },
       None => false
    }
}

fn string_to_user_move(s: String) -> Option<UserMove> {
    match s.to_lowercase().as_str() {
        "q" => Some(UserMove::TopLft),
        "w" => Some(UserMove::TopMid),
        "e" => Some(UserMove::TopRgt),
        "a" => Some(UserMove::MidLft),
        "s" => Some(UserMove::MidMid),
        "d" => Some(UserMove::MidRgt),
        "z" => Some(UserMove::BotLft),
        "x" => Some(UserMove::BotMid),
        "c" => Some(UserMove::BotRgt),
        _   => None
    }
}

fn maybe_get_user_move() -> Option<UserMove> {
    println!("Select move from qweasdzxc");
    let stdin = io::stdin();
    let mut iterator = stdin.lock().lines();
    let line1 = iterator.next().unwrap().unwrap();

    string_to_user_move(line1)
}

fn get_user_move() -> UserMove {
    let f = || maybe_get_user_move();
    get_valid_input_from_user(f)
}

fn do_user_move(gameboard: &mut GameBoard) -> bool {
    let user_move = get_user_move();
    apply_move(gameboard, user_move, true)
}

fn ask_user_for_new_game() -> bool {
    let f = || get_yes_no_from_user("Do you want to play again?");
    get_valid_input_from_user(f)
}

fn end_game(game_over_state: &GameOverState) {
    match game_over_state {
        GameOverState::Tied => println!("No winners - game is tied!"),
        GameOverState::Won(true) => println!("User (you) have won!"),
        GameOverState::Won(false) => println!("AI has won!")
    }
}

fn transition_gamestate(gamestate: &mut GameState) {
    match gamestate {
        GameState::Starting => {
                *gamestate = start_game()
        },
        GameState::Playing(playing_state) => {
            if playing_state.user_turn {
                if do_user_move(&mut playing_state.gameboard) {
                    playing_state.user_turn = false;
                    
                }
                true
            } else {
                playing_state.user_turn = true;
                do_ai_move(&mut playing_state.gameboard)
            };
            if let Some(game_over_state) = game_is_over(&playing_state.gameboard) {
                *gamestate = GameState::GameOver(game_over_state);
            }
        },
        GameState::GameOver(_) => {
            if ask_user_for_new_game() {
                *gamestate = GameState::Starting
            } else {
                *gamestate = GameState::Exiting
            }
        },
        GameState::Exiting => ()
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
        GameState::GameOver(game_over_state) => end_game(&game_over_state),
        GameState::Exiting => ()
    }
}

impl GameState {
    fn not_exiting(&self) -> bool {
        match &self {
            GameState::Exiting => false,
            _ => true
        }
    }
}
pub fn game_loop() {
    let mut gamestate = GameState::Starting;

    display_game(&gamestate);
    while gamestate.not_exiting() {
        transition_gamestate(&mut gamestate);
        display_game(&gamestate);
    }
}