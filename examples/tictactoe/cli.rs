use crate::tictactoe::display::board_to_string;
use crate::tictactoe::game::*;
use crossterm::{
    execute,
    terminal::{Clear, ClearType},
};
use rand::seq::SliceRandom;
use std::io::{self, stdout, BufRead};

pub struct RandomAgent;

impl Agent for RandomAgent {
    fn select_move(&mut self, state: &PlayingGameState) -> CellLocation {
        let moves: Vec<_> = state.gameboard.available_moves().collect();
        let mut rng = rand::thread_rng();
        moves
            .choose(&mut rng)
            .copied()
            .unwrap_or(CellLocation::MidMid)
    }
}

pub struct CliAgent;

impl Agent for CliAgent {
    fn select_move(&mut self, state: &PlayingGameState) -> CellLocation {
        get_user_move(&state.gameboard)
    }
}

fn get_valid_input_from_user<T>(f: impl Fn() -> Option<T>) -> T {
    loop {
        match f() {
            Some(result) => return result,
            None => println!("You did not enter a valid response. Try again."),
        }
    }
}

fn read_line_from_stdin() -> io::Result<String> {
    let stdin = io::stdin();
    stdin
        .lock()
        .lines()
        .next()
        .unwrap_or_else(|| Ok(String::new()))
}

fn get_yes_no_from_user(message: &str) -> Option<bool> {
    println!("Yes or No\n{message}");
    let line = read_line_from_stdin().unwrap_or_default();

    match line.to_lowercase().as_str() {
        "yes" | "y" => Some(true),
        "no" | "n" => Some(false),
        _ => None,
    }
}

fn get_first_player_from_user() -> bool {
    get_valid_input_from_user(|| get_yes_no_from_user("Do you want to play first?"))
}

fn start_game() -> (PlayingGameState, Player) {
    // In this game, Cross always starts.
    // The user choice determines marker ownership, not turn order directly.
    let user_player = if get_first_player_from_user() {
        Player::Cross
    } else {
        Player::Circle
    };

    (new_game(Player::Cross), user_player)
}

fn string_to_player_move(s: &str) -> Option<CellLocation> {
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
        _ => None,
    }
}

fn maybe_get_user_move() -> Option<CellLocation> {
    println!("Select move from qweasdzxc");
    let line = read_line_from_stdin().unwrap_or_default();
    string_to_player_move(&line)
}

fn clear_screen() {
    let _ = execute!(stdout(), Clear(ClearType::Purge));
}

pub fn get_user_move(gameboard: &GameBoard) -> CellLocation {
    clear_screen();
    println!("{}", board_to_string(gameboard));
    get_valid_input_from_user(maybe_get_user_move)
}

fn ask_user_for_new_game() -> bool {
    get_valid_input_from_user(|| get_yes_no_from_user("Do you want to play again?"))
}

fn end_game(gameboard: &GameBoard, game_over_state: &GameOverState, user_player: Player) {
    clear_screen();
    println!("{}", board_to_string(gameboard));

    match game_over_state {
        GameOverState::Tied => println!("No winners - game is tied!"),
        GameOverState::Won(winner) => {
            if *winner == user_player {
                println!("User (you) have won!");
            } else {
                println!("AI has won!");
            }
        }
        GameOverState::Disqualified(player, m) => {
            if *player == user_player {
                println!(
                    "User (you) have tried to play an illegal move ({:?}) and are disqualified!",
                    m
                );
            } else {
                println!(
                    "AI tried to play an illegal move ({:?}) and is disqualified!",
                    m
                );
            }
        }
    }
}

pub fn game_loop() {
    let mut cli_agent = CliAgent;
    let mut ai_agent = RandomAgent;

    loop {
        let (initial_state, user_player) = start_game();

        let (gameboard, game_over_state) = match user_player {
            Player::Cross => play_game(&mut cli_agent, &mut ai_agent, initial_state),
            Player::Circle => play_game(&mut ai_agent, &mut cli_agent, initial_state),
        };

        end_game(&gameboard, &game_over_state, user_player);

        if !ask_user_for_new_game() {
            break;
        }
    }
}
