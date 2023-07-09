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
    GameOver(GameOverState),
    Exiting
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
    fn check_assign(cell: &mut Cell, val: Player) -> bool {
        if cell.0 == None {
            *cell = Cell(Some(val)); //TODO: can a closure be used to eliminate the val param?
            true
        } else {
            false
        }
    } 
    match player_move {
        PlayerMove::TopLft => check_assign(&mut gameboard.top_lft, player),
        PlayerMove::TopMid => check_assign(&mut gameboard.top_mid, player),
        PlayerMove::TopRgt => check_assign(&mut gameboard.top_rgt, player),
        PlayerMove::MidLft => check_assign(&mut gameboard.mid_lft, player),
        PlayerMove::MidMid => check_assign(&mut gameboard.mid_mid, player),
        PlayerMove::MidRgt => check_assign(&mut gameboard.mid_rgt, player),
        PlayerMove::BotLft => check_assign(&mut gameboard.bot_lft, player),
        PlayerMove::BotMid => check_assign(&mut gameboard.bot_mid, player),
        PlayerMove::BotRgt => check_assign(&mut gameboard.bot_rgt, player)
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

