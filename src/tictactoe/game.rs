pub struct Cell(pub Option<bool>);

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
pub enum UserMove {
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
    pub user_turn: bool
}

pub enum GameOverState {
    Tied,
    Won(bool)
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

pub fn apply_move(gameboard: &mut GameBoard, user_move: UserMove, is_user: bool) -> bool {
    fn check_assign(cell: &mut Cell, val: bool) -> bool {
        if cell.0 == None {
            *cell = Cell(Some(val)); //TODO: can a closure be used to eliminate the val param?
            true
        } else {
            false
        }
    } 
    match user_move {
        UserMove::TopLft => check_assign(&mut gameboard.top_lft, is_user),
        UserMove::TopMid => check_assign(&mut gameboard.top_mid, is_user),
        UserMove::TopRgt => check_assign(&mut gameboard.top_rgt, is_user),
        UserMove::MidLft => check_assign(&mut gameboard.mid_lft, is_user),
        UserMove::MidMid => check_assign(&mut gameboard.mid_mid, is_user),
        UserMove::MidRgt => check_assign(&mut gameboard.mid_rgt, is_user),
        UserMove::BotLft => check_assign(&mut gameboard.bot_lft, is_user),
        UserMove::BotMid => check_assign(&mut gameboard.bot_mid, is_user),
        UserMove::BotRgt => check_assign(&mut gameboard.bot_rgt, is_user)
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

fn game_is_won(gameboard: &GameBoard) -> Option<bool> {
    fn win_line(c1: &Cell, c2: &Cell, c3: &Cell) -> Option<bool> {
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
        Some(user_won) => Some(GameOverState::Won(user_won)),
        None => {
            if game_is_tied(gameboard) {
                Some(GameOverState::Tied)
            } else {
                None
            }
        }
    }
}

