#[derive(PartialEq, Copy, Clone)]
pub enum Player {
    Cross,
    Circle
}

pub struct Cell(pub Option<Player>);

pub struct GameBoard {
    pub top_lft: Cell,
    pub top_mid: Cell,
    pub top_rgt: Cell,
    pub mid_lft: Cell,
    pub mid_mid: Cell,
    pub mid_rgt: Cell,
    pub bot_lft: Cell,
    pub bot_mid: Cell,
    pub bot_rgt: Cell
}

#[derive(Copy, Clone)]
pub enum PlayerMove {
    TopLft,
    TopMid,
    TopRgt,
    MidLft,
    MidMid,
    MidRgt,
    BotLft,
    BotMid,
    BotRgt
}

pub struct PlayingGameState {
    pub gameboard: GameBoard,
    pub player_turn: Player
}

pub enum GameOverState {
    Tied,
    Won(Player)
}

pub enum GameState {
    Starting,
    Playing(PlayingGameState),
    GameOver(GameOverState)
}

pub fn empty_game_board() -> GameBoard {
    GameBoard {
        top_lft: Cell(None),
        top_mid: Cell(None),
        top_rgt: Cell(None),
        mid_lft: Cell(None),
        mid_mid: Cell(None),
        mid_rgt: Cell(None),
        bot_lft: Cell(None),
        bot_mid: Cell(None),
        bot_rgt: Cell(None)
    }
}

pub fn apply_move(gameboard: &mut GameBoard, player_move: PlayerMove, player: Player) -> bool {
    let check_assign = |cell: &mut Cell| {
        if cell.0 == None {
            *cell = Cell(Some(player));
            true
        } else {
            false
        }
    };
    match player_move {
        PlayerMove::TopLft => check_assign(&mut gameboard.top_lft),
        PlayerMove::TopMid => check_assign(&mut gameboard.top_mid),
        PlayerMove::TopRgt => check_assign(&mut gameboard.top_rgt),
        PlayerMove::MidLft => check_assign(&mut gameboard.mid_lft),
        PlayerMove::MidMid => check_assign(&mut gameboard.mid_mid),
        PlayerMove::MidRgt => check_assign(&mut gameboard.mid_rgt),
        PlayerMove::BotLft => check_assign(&mut gameboard.bot_lft),
        PlayerMove::BotMid => check_assign(&mut gameboard.bot_mid),
        PlayerMove::BotRgt => check_assign(&mut gameboard.bot_rgt)
    }
}

pub fn transition_gamestate(
    gamestate: &mut GameState, 
    start_game: fn() -> GameState, 
    cross_move: fn(&GameBoard) -> PlayerMove,
    circle_move: fn(&GameBoard) -> PlayerMove) {
    match gamestate {
        GameState::Starting => {
                *gamestate = start_game()
        },
        GameState::Playing(playing_state) => {
            let (player_move, this_player, next_player) =
                match playing_state.player_turn {
                    Player::Cross => (cross_move(&playing_state.gameboard), Player::Cross, Player::Circle),
                    Player::Circle => (circle_move(&playing_state.gameboard), Player::Circle, Player::Cross)
                };

            if apply_move(&mut playing_state.gameboard, player_move, this_player) {
                playing_state.player_turn = next_player
            }

            if let Some(game_over_state) = game_is_over(&playing_state.gameboard) {
                *gamestate = GameState::GameOver(game_over_state);
            }
        },
        GameState::GameOver(_) => ()
    }
}

fn game_is_tied(gameboard: &GameBoard) -> bool {
    gameboard.top_lft.0.is_some() &&
    gameboard.top_mid.0.is_some() &&
    gameboard.top_rgt.0.is_some() &&
    gameboard.mid_lft.0.is_some() &&
    gameboard.mid_mid.0.is_some() &&
    gameboard.mid_rgt.0.is_some() &&
    gameboard.bot_lft.0.is_some() &&
    gameboard.bot_mid.0.is_some() &&
    gameboard.bot_rgt.0.is_some()
}

fn game_is_won(gameboard: &GameBoard) -> Option<Player> {
    fn win_line(c1: &Cell, c2: &Cell, c3: &Cell) -> Option<Player> {
        let b = 
            c1.0.is_some() &&
            c1.0 == c2.0 &&
            c2.0 == c3.0;

        if b {
            c1.0
        } else {
            None
        }

    }

    //horizontal win lines
    win_line(&gameboard.top_lft, &gameboard.top_mid, &gameboard.top_rgt)
    .or_else(|| win_line(&gameboard.mid_lft, &gameboard.mid_mid, &gameboard.mid_rgt))
    .or_else(|| win_line(&gameboard.bot_lft, &gameboard.bot_mid, &gameboard.bot_rgt))
    //vertical win lines
    .or_else(|| win_line(&gameboard.top_lft, &gameboard.mid_lft, &gameboard.bot_lft))
    .or_else(|| win_line(&gameboard.top_mid, &gameboard.mid_mid, &gameboard.bot_mid))
    .or_else(|| win_line(&gameboard.top_rgt, &gameboard.mid_rgt, &gameboard.bot_rgt))
    //diagonal win lines
    .or_else(|| win_line(&gameboard.top_lft, &gameboard.mid_mid, &gameboard.bot_rgt))
    .or_else(|| win_line(&gameboard.bot_lft, &gameboard.mid_mid, &gameboard.top_rgt))
}

pub fn game_is_over(gameboard: &GameBoard) -> Option<GameOverState> {
    match game_is_won(gameboard){
        Some(player) => Some(GameOverState::Won(player)),
        None => {
            if game_is_tied(gameboard) {
                Some(GameOverState::Tied)
            } else {
                None
            }
        }
    }
}

