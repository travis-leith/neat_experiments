#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Player {
    Cross,
    Circle,
}

impl Player {
    pub fn opponent(self) -> Self {
        match self {
            Player::Cross => Player::Circle,
            Player::Circle => Player::Cross,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Cell(pub Option<Player>);

#[derive(Copy, Clone, Debug)]
pub struct GameBoard {
    pub cells: [Cell; 9],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CellLocation {
    TopLft = 0,
    TopMid = 1,
    TopRgt = 2,
    MidLft = 3,
    MidMid = 4,
    MidRgt = 5,
    BotLft = 6,
    BotMid = 7,
    BotRgt = 8,
}

impl CellLocation {
    pub fn from_usize(i: usize) -> Option<Self> {
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
            _ => None,
        }
    }

    pub fn all() -> [Self; 9] {
        [
            Self::TopLft,
            Self::TopMid,
            Self::TopRgt,
            Self::MidLft,
            Self::MidMid,
            Self::MidRgt,
            Self::BotLft,
            Self::BotMid,
            Self::BotRgt,
        ]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PlayingGameState {
    pub gameboard: GameBoard,
    pub player_turn: Player,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GameOverState {
    Tied,
    Won(Player),
    Disqualified(Player, CellLocation),
}

#[derive(Copy, Clone, Debug)]
pub enum GameState {
    Playing(PlayingGameState),
    GameOver(GameBoard, GameOverState),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MoveError {
    CellOccupied,
}

fn empty_game_board() -> GameBoard {
    GameBoard {
        cells: [Cell(None); 9],
    }
}

impl GameBoard {
    pub fn get_cell(&self, cell_loc: CellLocation) -> Cell {
        self.cells[cell_loc as usize]
    }

    pub fn is_cell_empty(&self, cell_loc: CellLocation) -> bool {
        self.cells[cell_loc as usize].0.is_none()
    }

    pub fn available_moves(&self) -> impl Iterator<Item = CellLocation> + '_ {
        CellLocation::all()
            .into_iter()
            .filter(|&loc| self.is_cell_empty(loc))
    }

    fn with_cell(mut self, cell_loc: CellLocation, player: Player) -> Self {
        self.cells[cell_loc as usize] = Cell(Some(player));
        self
    }

    fn is_full(&self) -> bool {
        let mut i = 0;
        while i < 9 {
            if self.cells[i].0.is_none() {
                return false;
            }
            i += 1;
        }
        true
    }

    fn check_line(&self, m1: CellLocation, m2: CellLocation, m3: CellLocation) -> Option<Player> {
        let c1 = self.cells[m1 as usize];
        let c2 = self.cells[m2 as usize];
        let c3 = self.cells[m3 as usize];

        match (c1.0, c2.0, c3.0) {
            (Some(p1), Some(p2), Some(p3)) if p1 == p2 && p1 == p3 => Some(p1),
            _ => None,
        }
    }

    pub fn winner(&self) -> Option<Player> {
        use CellLocation::*;

        // Horizontal
        if let Some(p) = self.check_line(TopLft, TopMid, TopRgt) {
            return Some(p);
        }
        if let Some(p) = self.check_line(MidLft, MidMid, MidRgt) {
            return Some(p);
        }
        if let Some(p) = self.check_line(BotLft, BotMid, BotRgt) {
            return Some(p);
        }

        // Vertical
        if let Some(p) = self.check_line(TopLft, MidLft, BotLft) {
            return Some(p);
        }
        if let Some(p) = self.check_line(TopMid, MidMid, BotMid) {
            return Some(p);
        }
        if let Some(p) = self.check_line(TopRgt, MidRgt, BotRgt) {
            return Some(p);
        }

        // Diagonal
        if let Some(p) = self.check_line(TopLft, MidMid, BotRgt) {
            return Some(p);
        }
        if let Some(p) = self.check_line(BotLft, MidMid, TopRgt) {
            return Some(p);
        }

        None
    }

    fn game_over_state(&self) -> Option<GameOverState> {
        match self.winner() {
            Some(player) => Some(GameOverState::Won(player)),
            None if self.is_full() => Some(GameOverState::Tied),
            None => None,
        }
    }
}

pub fn new_game(first_player: Player) -> PlayingGameState {
    PlayingGameState {
        gameboard: empty_game_board(),
        player_turn: first_player,
    }
}

impl PlayingGameState {
    pub fn apply_move(self, cell_loc: CellLocation) -> Result<GameState, MoveError> {
        if !self.gameboard.is_cell_empty(cell_loc) {
            return Err(MoveError::CellOccupied);
        }

        let new_board = self.gameboard.with_cell(cell_loc, self.player_turn);

        match new_board.game_over_state() {
            Some(game_over) => Ok(GameState::GameOver(new_board, game_over)),
            None => Ok(GameState::Playing(PlayingGameState {
                gameboard: new_board,
                player_turn: self.player_turn.opponent(),
            })),
        }
    }

    pub fn apply_move_or_disqualify(self, cell_loc: CellLocation) -> GameState {
        match self.apply_move(cell_loc) {
            Ok(state) => state,
            Err(MoveError::CellOccupied) => GameState::GameOver(
                self.gameboard,
                GameOverState::Disqualified(self.player_turn, cell_loc),
            ),
        }
    }
}

pub trait Agent {
    fn select_move(&mut self, state: &PlayingGameState) -> CellLocation;
}

pub fn play_game<A1: Agent, A2: Agent>(
    cross_agent: &mut A1,
    circle_agent: &mut A2,
    state: PlayingGameState,
) -> (GameBoard, GameOverState) {
    fn play_recursive<A1: Agent, A2: Agent>(
        cross_agent: &mut A1,
        circle_agent: &mut A2,
        state: PlayingGameState,
    ) -> (GameBoard, GameOverState) {
        let chosen_move = match state.player_turn {
            Player::Cross => cross_agent.select_move(&state),
            Player::Circle => circle_agent.select_move(&state),
        };

        match state.apply_move_or_disqualify(chosen_move) {
            GameState::Playing(new_state) => {
                become play_recursive(cross_agent, circle_agent, new_state)
            }
            GameState::GameOver(board, game_over) => (board, game_over),
        }
    }

    play_recursive(cross_agent, circle_agent, state)
}
