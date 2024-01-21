use tailcall::tailcall;

#[derive(PartialEq, Copy, Clone)]
pub enum Player {
    Cross,
    Circle
}

#[derive(Copy, Clone)]
pub struct Cell(pub Option<Player>);

pub struct GameBoard {
    pub cells: [Cell; 9]
}

#[derive(Copy, Clone, Debug)]
pub enum CellLocation {
    TopLft = 0,
    TopMid = 1,
    TopRgt = 2,
    MidLft = 3,
    MidMid = 4,
    MidRgt = 5,
    BotLft = 6,
    BotMid = 7,
    BotRgt = 8
}

impl CellLocation {
    pub fn from_usize(i:usize) -> Option<CellLocation> {
        match i {
            0 => Some(Self::TopLft),
            1 => Some(Self::TopMid),
            2 => Some(Self::TopRgt),
            3 => Some(Self::MidLft),
            4 => Some(Self::MidMid),
            5 => Some(Self::MidRgt),
            6 => Some(Self::BotLft),
            7 => Some(Self::BotMid),
            8 => Some(Self::BotRgt),
            _ => None
        }
    }
}

pub struct PlayingGameState {
    pub gameboard: GameBoard,
    pub player_turn: Player
}

pub enum GameOverState {
    Tied,
    Won(Player),
    Disqualified(Player, CellLocation)
}

pub enum GameState {
    Playing(PlayingGameState),
    // BadMove(PlayingGameState),
    GameOver(GameBoard, GameOverState)
}

fn empty_game_board() -> GameBoard {
    GameBoard {
        cells: [Cell(None); 9]
    }
}

impl GameBoard {
    pub fn get_cell(&self, cell_loc: CellLocation) -> Cell {
        let i = cell_loc as usize;
        self.cells[i]
    }

    fn try_set_cell(&mut self, cell_loc: CellLocation, player: Player) -> bool {
        let i = cell_loc as usize;
        match self.cells[i] {
            Cell(None) => {
                self.cells[i] = Cell(Some(player));
                true
            },
            _ => false
        }
    }
}
pub fn new_game(first_player: Player) -> PlayingGameState{
    PlayingGameState {gameboard: empty_game_board(), player_turn: first_player}
}

fn gameboard_is_full (gameboard: &GameBoard) -> bool {
    gameboard.cells.iter().all(|cell| cell.0.is_some())
}

fn game_winner(gameboard: &GameBoard) -> Option<Player>{
    let win_line = |m1: CellLocation, m2: CellLocation, m3: CellLocation| {
        match (gameboard.get_cell(m1), gameboard.get_cell(m2), gameboard.get_cell(m3)) {
            (Cell(Some(p1)), Cell(Some(p2)), Cell(Some(p3))) if p1 == p2 && p1 == p3 => Some(p1),
            _ => None
        }
    };
    
    //horizontal win lines
    win_line(CellLocation::TopLft, CellLocation::TopMid, CellLocation::TopRgt)
    .or_else(|| win_line(CellLocation::MidLft, CellLocation::MidMid, CellLocation::MidRgt))
    .or_else(|| win_line(CellLocation::BotLft, CellLocation::BotMid, CellLocation::BotRgt))
    //vertical win lines
    .or_else(|| win_line(CellLocation::TopLft, CellLocation::MidLft, CellLocation::BotLft))
    .or_else(|| win_line(CellLocation::TopMid, CellLocation::MidMid, CellLocation::BotMid))
    .or_else(|| win_line(CellLocation::TopRgt, CellLocation::MidRgt, CellLocation::BotRgt))
    //diagonal win lines
    .or_else(|| win_line(CellLocation::TopLft, CellLocation::MidMid, CellLocation::BotRgt))
    .or_else(|| win_line(CellLocation::BotLft, CellLocation::MidMid, CellLocation::TopRgt))

}

fn check_game_over (gameboard: &GameBoard) -> Option<GameOverState> {
    match game_winner(gameboard) {
        Some(player) => Some(GameOverState::Won(player)),
        None if gameboard_is_full(gameboard) => Some(GameOverState::Tied),
        None => None
    }
}

pub trait Controller {
    fn circle_mover(&mut self, gameboard: &GameBoard) -> CellLocation;
    fn cross_mover(&mut self, gameboard: &GameBoard) -> CellLocation;
    fn retry_allowed(&mut self) -> bool;
}

fn play_one_move(ctrl: &mut impl Controller, mut playing_state: PlayingGameState) -> Result<GameState, PlayingGameState> {
    let (player_move, next_player) = match playing_state.player_turn {
        Player::Cross => (ctrl.cross_mover(&playing_state.gameboard), Player::Circle),
        Player::Circle => (ctrl.circle_mover(&playing_state.gameboard), Player::Cross)
    };

    if playing_state.gameboard.try_set_cell(player_move, playing_state.player_turn) {
        match check_game_over(&playing_state.gameboard) {
            Some(game_over_state) => Ok(GameState::GameOver(playing_state.gameboard, game_over_state)),
            None => {
                playing_state.player_turn = next_player;
                Ok(GameState::Playing(playing_state))
            }
        }
    } else if ctrl.retry_allowed() {
        Err(playing_state)
    } else {
        Ok(GameState::GameOver(playing_state.gameboard, GameOverState::Disqualified(playing_state.player_turn, player_move)))
    }
}

// #[tailcall]
// pub fn play_game(ctrl: &mut impl Controller, playing_state: PlayingGameState) -> Result<(GameBoard, GameOverState), PlayingGameState> {
//     match play_one_move(ctrl, playing_state) {
//         Ok(GameState::Playing(new_playing_state)) => play_game(ctrl, new_playing_state),
//         Ok(GameState::GameOver(gameboard, game_over_state)) => Ok((gameboard, game_over_state)),
//         Err(new_playing_state) => Err(new_playing_state)
//     }
// }

//play_game takes a playing_state so that games can be resumable mid play
pub fn play_game(ctrl: &mut impl Controller, mut playing_state: PlayingGameState) -> Result<(GameBoard, GameOverState), PlayingGameState> {
    loop {
        match play_one_move(ctrl, playing_state) {
            Ok(GameState::Playing(new_playing_state)) => {
                playing_state = new_playing_state;
            },
            Ok(GameState::GameOver(gameboard, game_over_state)) => return Ok((gameboard, game_over_state)),
            Err(new_playing_state) => return Err(new_playing_state)
        }
    }
}