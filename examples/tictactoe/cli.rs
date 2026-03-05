use crate::tictactoe::display::board_to_string;
use crate::tictactoe::game::*;
use crate::tictactoe::neat_agent::NeatAgent;
use crossterm::{
    execute,
    terminal::{Clear, ClearType},
};
use neat_experiments::neat::phenome::Phenome;
use rand::seq::SliceRandom;
use std::io::{self, stdout, BufRead};

struct RandomAgent;

impl Agent for RandomAgent {
    fn select_move(&mut self, state: &PlayingGameState) -> CellLocation {
        let moves: Vec<_> = state.gameboard.available_moves().collect();
        let mut rng = rand::thread_rng();
        *moves
            .choose(&mut rng)
            .expect("select_move called with no available moves")
    }
}

struct CliAgent;

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

fn read_line_from_stdin() -> String {
    let stdin = io::stdin();
    stdin
        .lock()
        .lines()
        .next()
        .expect("stdin closed unexpectedly")
        .expect("failed to read line from stdin")
}

fn get_yes_no_from_user(message: &str) -> Option<bool> {
    println!("Yes or No\n{message}");
    let line = read_line_from_stdin();

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
    let line = read_line_from_stdin();
    string_to_player_move(&line)
}

fn clear_screen() {
    execute!(stdout(), Clear(ClearType::Purge)).expect("failed to clear screen");
}

fn get_user_move(gameboard: &GameBoard) -> CellLocation {
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

fn play_loop(mut opponent: impl Agent) {
    let mut cli_agent = CliAgent;

    loop {
        let (initial_state, user_player) = start_game();

        let (gameboard, game_over_state) = match user_player {
            Player::Cross => play_game(&mut cli_agent, &mut opponent, initial_state),
            Player::Circle => play_game(&mut opponent, &mut cli_agent, initial_state),
        };

        end_game(&gameboard, &game_over_state, user_player);

        if !ask_user_for_new_game() {
            break;
        }
    }
}

pub fn game_loop() {
    play_loop(RandomAgent);
}

pub fn play_against_neat(phenome: Phenome) {
    play_loop(DynamicNeatAgent { phenome });
}

/// An agent wrapper that sets the NeatAgent's perspective based on the current game state.
struct DynamicNeatAgent {
    phenome: Phenome,
}

impl Agent for DynamicNeatAgent {
    fn select_move(&mut self, state: &PlayingGameState) -> CellLocation {
        let mut agent = NeatAgent::new(self.phenome.clone(), state.player_turn);
        agent.select_move(state)
    }
}
