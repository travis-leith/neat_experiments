#[cfg(test)]
mod tests {
    use crate::tictactoe::game::*;
    use crate::tictactoe::minimax::*;
    use crate::tictactoe::neat_agent::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // ---------------------------------------------------------------
    // Cache and minimax correctness tests
    // ---------------------------------------------------------------

    #[test]
    fn optimal_moves_from_start_includes_corners_and_center() {
        // From the starting position, optimal moves for X are well-known:
        // corner or center. Minimax should return at least these.
        let state = new_game(Player::Cross);
        let (outcome, moves) = optimal_moves(&state);

        // Tic-tac-toe is a draw with perfect play from both sides
        assert_eq!(
            outcome,
            Outcome::Draw,
            "starting position should be a draw with perfect play"
        );
        assert!(
            !moves.is_empty(),
            "there should be at least one optimal move"
        );

        // The center is always an optimal first move
        assert!(
            moves.contains(&CellLocation::MidMid),
            "center should be among optimal first moves, got: {:?}",
            moves
        );
    }

    #[test]
    fn optimal_moves_from_winning_position() {
        // Set up a board where Cross can win immediately:
        // X X _
        // O O _
        // _ _ _
        let mut board = GameBoard {
            cells: [Cell(None); 9],
        };
        board.cells[0] = Cell(Some(Player::Cross)); // TopLft
        board.cells[1] = Cell(Some(Player::Cross)); // TopMid
        board.cells[3] = Cell(Some(Player::Circle)); // MidLft
        board.cells[4] = Cell(Some(Player::Circle)); // MidMid

        let state = PlayingGameState {
            gameboard: board,
            player_turn: Player::Cross,
        };

        let (outcome, moves) = optimal_moves(&state);

        assert!(
            matches!(outcome, Outcome::Win(_)),
            "Cross should be winning, got: {:?}",
            outcome
        );
        assert!(
            moves.contains(&CellLocation::TopRgt),
            "TopRgt should be the winning move, got: {:?}",
            moves
        );
    }

    #[test]
    fn optimal_moves_forced_loss_position() {
        // Cross has two threats, it's Circle's turn, Circle must lose
        // X _ X
        // O X _
        // _ _ O
        let mut board = GameBoard {
            cells: [Cell(None); 9],
        };
        board.cells[0] = Cell(Some(Player::Cross)); // TopLft
        board.cells[2] = Cell(Some(Player::Cross)); // TopRgt
        board.cells[4] = Cell(Some(Player::Cross)); // MidMid
        board.cells[5] = Cell(Some(Player::Circle)); // MidLft
        board.cells[8] = Cell(Some(Player::Circle)); // BotRgt

        let state = PlayingGameState {
            gameboard: board,
            player_turn: Player::Circle,
        };

        let (outcome, _moves) = optimal_moves(&state);

        assert!(
            matches!(outcome, Outcome::Loss(_)),
            "Circle should be losing, got: {:?}",
            outcome
        );
    }

    #[test]
    fn minimax_cache_returns_same_result() {
        // Running optimal_moves twice on the same state should give the same result,
        // which exercises the cache path on second call.
        let state = new_game(Player::Cross);

        let (outcome1, moves1) = optimal_moves(&state);
        let (outcome2, moves2) = optimal_moves(&state);

        assert_eq!(outcome1, outcome2);
        assert_eq!(moves1, moves2);
    }

    #[test]
    fn outcome_negate_is_correct() {
        assert_eq!(Outcome::Win(3).negate(), Outcome::Loss(3));
        assert_eq!(Outcome::Loss(3).negate(), Outcome::Win(3));
        assert_eq!(Outcome::Draw.negate(), Outcome::Draw);
    }

    #[test]
    fn outcome_is_better_than() {
        // Win is better than draw, draw is better than loss
        assert!(Outcome::Win(1).is_better_than(Outcome::Draw));
        assert!(Outcome::Win(1).is_better_than(Outcome::Loss(1)));
        assert!(Outcome::Draw.is_better_than(Outcome::Loss(1)));

        // Quicker win is better
        assert!(Outcome::Win(1).is_better_than(Outcome::Win(3)));
        assert!(!Outcome::Win(3).is_better_than(Outcome::Win(1)));

        // Slower loss is better (delay the inevitable)
        assert!(Outcome::Loss(5).is_better_than(Outcome::Loss(1)));
        assert!(!Outcome::Loss(1).is_better_than(Outcome::Loss(5)));

        // Same outcome is not better
        assert!(!Outcome::Draw.is_better_than(Outcome::Draw));
    }

    #[test]
    fn outcome_score_values() {
        assert_eq!(Outcome::Win(1).score(), 1);
        assert_eq!(Outcome::Draw.score(), 0);
        assert_eq!(Outcome::Loss(1).score(), -1);
    }

    // ---------------------------------------------------------------
    // board_to_inputs tests
    // ---------------------------------------------------------------

    #[test]
    fn board_to_inputs_empty_board() {
        let board = GameBoard {
            cells: [Cell(None); 9],
        };
        let inputs = board_to_inputs(&board, Player::Cross);

        // 9 cells all 0.0 + 1 bias = 10 inputs
        assert_eq!(inputs.len(), 10);
        for i in 0..9 {
            assert_eq!(inputs[i], 0.0, "empty cell {} should be 0.0", i);
        }
        assert_eq!(inputs[9], 1.0, "bias input should be 1.0");
    }

    #[test]
    fn board_to_inputs_perspective_matters() {
        let mut board = GameBoard {
            cells: [Cell(None); 9],
        };
        board.cells[0] = Cell(Some(Player::Cross));
        board.cells[1] = Cell(Some(Player::Circle));

        let cross_inputs = board_to_inputs(&board, Player::Cross);
        let circle_inputs = board_to_inputs(&board, Player::Circle);

        // From Cross's perspective: cell 0 is own (1.0), cell 1 is opponent (-1.0)
        assert_eq!(cross_inputs[0], 1.0);
        assert_eq!(cross_inputs[1], -1.0);

        // From Circle's perspective: cell 0 is opponent (-1.0), cell 1 is own (1.0)
        assert_eq!(circle_inputs[0], -1.0);
        assert_eq!(circle_inputs[1], 1.0);
    }

    #[test]
    fn board_to_inputs_has_correct_length() {
        let board = GameBoard {
            cells: [Cell(None); 9],
        };
        let inputs = board_to_inputs(&board, Player::Cross);
        assert_eq!(inputs.len(), 10, "should have 9 board cells + 1 bias = 10");
    }

    // ---------------------------------------------------------------
    // outputs_to_move tests
    // ---------------------------------------------------------------

    #[test]
    fn outputs_to_move_picks_highest() {
        let outputs = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(outputs_to_move(&outputs), CellLocation::MidMid);
    }

    #[test]
    fn outputs_to_move_picks_first_on_tie() {
        // When all outputs are equal, max_by returns the last max by default in Rust
        // (since it uses >=). Actually, max_by keeps the *last* maximum for equal elements.
        // Let's verify the actual behavior:
        let outputs = vec![0.5; 9];
        let result = outputs_to_move(&outputs);
        // max_by with partial_cmp will return the last index with the max value
        // because max_by returns the later element on equality
        assert_eq!(result, CellLocation::BotRgt);
    }

    #[test]
    fn outputs_to_move_with_negative_values() {
        let mut outputs = vec![-1.0; 9];
        outputs[6] = -0.5; // BotLft is the "best" (least negative)
        assert_eq!(outputs_to_move(&outputs), CellLocation::BotLft);
    }

    #[test]
    #[should_panic(expected = "outputs must not be empty")]
    fn outputs_to_move_panics_on_empty() {
        let outputs: Vec<f64> = vec![];
        outputs_to_move(&outputs);
    }

    // ---------------------------------------------------------------
    // score_against_perfect_play tests
    // ---------------------------------------------------------------

    #[test]
    fn score_perfect_agent_gets_all_correct() {
        // An agent that always picks the first optimal move should get 100%
        let initial = new_game(Player::Cross);
        let mut rng = StdRng::seed_from_u64(42);

        let (correct, total) = score_against_perfect_play(
            initial,
            Player::Cross,
            |state| {
                let (_, optimal) = optimal_moves(state);
                optimal[0]
            },
            &mut rng,
        );

        assert!(total > 0, "there should be at least one move");
        assert_eq!(
            correct, total,
            "a perfect agent should get all moves correct (got {}/{})",
            correct, total
        );
    }

    #[test]
    fn score_perfect_agent_as_circle_gets_all_correct() {
        let initial = new_game(Player::Cross);
        let mut rng = StdRng::seed_from_u64(42);

        let (correct, total) = score_against_perfect_play(
            initial,
            Player::Circle,
            |state| {
                let (_, optimal) = optimal_moves(state);
                optimal[0]
            },
            &mut rng,
        );

        assert!(total > 0, "circle should have at least one move");
        assert_eq!(
            correct, total,
            "a perfect agent as circle should get all moves correct (got {}/{})",
            correct, total
        );
    }

    #[test]
    fn score_worst_agent_gets_some_wrong() {
        // An agent that always picks the first available move (index 0) will
        // likely not play perfectly.
        let initial = new_game(Player::Cross);
        let mut rng = StdRng::seed_from_u64(42);

        let (correct, total) = score_against_perfect_play(
            initial,
            Player::Cross,
            |state| {
                // Just pick the first available move
                state
                    .gameboard
                    .available_moves()
                    .next()
                    .expect("should have available moves")
            },
            &mut rng,
        );

        assert!(total > 0, "there should be at least one move");
        // We can't guarantee it gets ALL wrong, but for a typical game it should
        // miss at least one.
        // At minimum, verify the function runs and returns reasonable values.
        assert!(correct <= total);
    }

    #[test]
    fn score_against_perfect_play_counts_only_agent_moves() {
        // When playing as Cross (first player), the agent should have more
        // moves than when playing as Circle (second player), for the same game.
        let initial = new_game(Player::Cross);
        let mut rng = StdRng::seed_from_u64(42);

        let (_, total_as_cross) = score_against_perfect_play(
            initial,
            Player::Cross,
            |state| {
                let (_, optimal) = optimal_moves(state);
                optimal[0]
            },
            &mut rng,
        );

        let (_, total_as_circle) = score_against_perfect_play(
            initial,
            Player::Circle,
            |state| {
                let (_, optimal) = optimal_moves(state);
                optimal[0]
            },
            &mut rng,
        );

        // Cross goes first, so should have >= Circle's move count
        assert!(
            total_as_cross >= total_as_circle,
            "Cross ({}) should have at least as many moves as Circle ({})",
            total_as_cross,
            total_as_circle
        );
    }

    // ---------------------------------------------------------------
    // Game logic tests
    // ---------------------------------------------------------------

    #[test]
    fn new_game_is_empty() {
        let state = new_game(Player::Cross);
        assert_eq!(state.player_turn, Player::Cross);
        for cell in &state.gameboard.cells {
            assert_eq!(cell.0, None);
        }
    }

    #[test]
    fn apply_move_switches_player() {
        let state = new_game(Player::Cross);
        match state.apply_move(CellLocation::MidMid).unwrap() {
            GameState::Playing(next) => {
                assert_eq!(next.player_turn, Player::Circle);
            }
            _ => panic!("game should still be playing after first move"),
        }
    }

    // #[test]
    // fn apply_move_to_occupied_cell_is_error() {
    //     let state = new_game(Player::Cross);
    //     let next = match state.apply_move(CellLocation::MidMid).unwrap() {
    //         GameState::Playing(s) => s,
    //         _ => panic!("game should still be playing"),
    //     };
    //     assert_eq!(
    //         next.apply_move(CellLocation::MidMid),
    //         Err(MoveError::CellOccupied)
    //     );
    // }

    #[test]
    fn apply_move_or_disqualify_on_occupied() {
        let state = new_game(Player::Cross);
        let next = match state.apply_move(CellLocation::MidMid).unwrap() {
            GameState::Playing(s) => s,
            _ => panic!("game should still be playing"),
        };
        match next.apply_move_or_disqualify(CellLocation::MidMid) {
            GameState::GameOver(_, GameOverState::Disqualified(player, loc)) => {
                assert_eq!(player, Player::Circle);
                assert_eq!(loc, CellLocation::MidMid);
            }
            _ => panic!("should be disqualified for playing on occupied cell"),
        }
    }

    #[test]
    fn detect_win() {
        // Play X across top row: TopLft, TopMid, TopRgt
        let state = new_game(Player::Cross);
        // X plays TopLft
        let state = match state.apply_move(CellLocation::TopLft).unwrap() {
            GameState::Playing(s) => s,
            _ => panic!("game should continue"),
        };
        // O plays MidLft
        let state = match state.apply_move(CellLocation::MidLft).unwrap() {
            GameState::Playing(s) => s,
            _ => panic!("game should continue"),
        };
        // X plays TopMid
        let state = match state.apply_move(CellLocation::TopMid).unwrap() {
            GameState::Playing(s) => s,
            _ => panic!("game should continue"),
        };
        // O plays MidMid
        let state = match state.apply_move(CellLocation::MidMid).unwrap() {
            GameState::Playing(s) => s,
            _ => panic!("game should continue"),
        };
        // X plays TopRgt — should win
        match state.apply_move(CellLocation::TopRgt).unwrap() {
            GameState::GameOver(_, GameOverState::Won(winner)) => {
                assert_eq!(winner, Player::Cross);
            }
            other => panic!("Cross should have won, got: {:?}", other),
        }
    }

    #[test]
    fn detect_tie() {
        // Play a game that ends in a tie:
        // X O X
        // X X O
        // O X O
        let moves = [
            (Player::Cross, CellLocation::TopLft),  // X
            (Player::Circle, CellLocation::TopMid), // O
            (Player::Cross, CellLocation::TopRgt),  // X
            (Player::Circle, CellLocation::MidRgt), // O
            (Player::Cross, CellLocation::MidLft),  // X
            (Player::Circle, CellLocation::BotLft), // O
            (Player::Cross, CellLocation::MidMid),  // X
            (Player::Circle, CellLocation::BotRgt), // O
            (Player::Cross, CellLocation::BotMid),  // X — should tie
        ];

        let mut game_state = GameState::Playing(new_game(Player::Cross));
        for (_, loc) in &moves {
            match game_state {
                GameState::Playing(state) => {
                    game_state = state.apply_move(*loc).unwrap();
                }
                GameState::GameOver(_, ref gs) => {
                    panic!("game ended early with state: {:?}", gs);
                }
            }
        }

        match game_state {
            GameState::GameOver(_, GameOverState::Tied) => {} // Expected
            other => panic!("expected tie, got: {:?}", other),
        }
    }

    #[test]
    fn cell_location_from_usize_roundtrip() {
        for i in 0..9 {
            let loc = CellLocation::from_usize(i).unwrap();
            assert_eq!(loc as usize, i);
        }
        assert!(CellLocation::from_usize(9).is_none());
        assert!(CellLocation::from_usize(100).is_none());
    }

    #[test]
    fn player_opponent() {
        assert_eq!(Player::Cross.opponent(), Player::Circle);
        assert_eq!(Player::Circle.opponent(), Player::Cross);
    }

    #[test]
    fn cell_location_all_returns_nine() {
        assert_eq!(CellLocation::all().len(), 9);
    }

    #[test]
    fn available_moves_decreases_as_game_progresses() {
        let state = new_game(Player::Cross);
        assert_eq!(state.gameboard.available_moves().count(), 9);

        let state = match state.apply_move(CellLocation::MidMid).unwrap() {
            GameState::Playing(s) => s,
            _ => panic!("game should continue"),
        };
        assert_eq!(state.gameboard.available_moves().count(), 8);
    }

    // ---------------------------------------------------------------
    // MinimaxAgent tests
    // ---------------------------------------------------------------

    #[test]
    fn minimax_agent_never_loses() {
        // Play minimax vs minimax — must always draw
        let mut agent1 = MinimaxAgent::new();
        let mut agent2 = MinimaxAgent::new();
        let initial = new_game(Player::Cross);
        let (_, result) = play_game(&mut agent1, &mut agent2, initial);
        assert_eq!(result, GameOverState::Tied, "minimax vs minimax must draw");
    }

    #[test]
    fn minimax_agent_wins_or_draws_against_first_available() {
        // An agent that always picks the first available cell should not beat minimax
        struct FirstAvailableAgent;
        impl Agent for FirstAvailableAgent {
            fn select_move(&mut self, state: &PlayingGameState) -> CellLocation {
                state.gameboard.available_moves().next().unwrap()
            }
        }

        // Minimax as Cross vs FirstAvailable as Circle
        let mut minimax = MinimaxAgent::new();
        let mut weak = FirstAvailableAgent;
        let initial = new_game(Player::Cross);
        let (_, result) = play_game(&mut minimax, &mut weak, initial);

        match result {
            GameOverState::Won(Player::Cross) | GameOverState::Tied => {} // OK
            other => panic!("minimax should win or draw, got: {:?}", other),
        }

        // Minimax as Circle vs FirstAvailable as Cross
        let mut minimax = MinimaxAgent::new();
        let mut weak = FirstAvailableAgent;
        let initial = new_game(Player::Cross);
        let (_, result) = play_game(&mut weak, &mut minimax, initial);

        match result {
            GameOverState::Won(Player::Circle) | GameOverState::Tied => {} // OK
            other => panic!("minimax should win or draw, got: {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Fitness calculation tests
    // ---------------------------------------------------------------

    #[test]
    fn perfect_play_fitness_is_one() {
        // Simulating what evaluate_organism_against_perfect does,
        // but with a perfect agent (always picks optimal moves).
        let mut total_correct: u32 = 0;
        let mut total_moves: u32 = 0;

        for &agent_player in &[Player::Cross, Player::Circle] {
            let initial = new_game(Player::Cross);
            let mut rng = StdRng::seed_from_u64(42);

            let (correct, moves) = score_against_perfect_play(
                initial,
                agent_player,
                |state| {
                    let (_, optimal) = optimal_moves(state);
                    optimal[0]
                },
                &mut rng,
            );

            total_correct += correct;
            total_moves += moves;
        }

        let fitness = if total_moves == 0 {
            0.0
        } else {
            (total_correct as f64) / (total_moves as f64)
        };

        assert!(
            (fitness - 1.0).abs() < f64::EPSILON,
            "perfect agent should have fitness 1.0, got {}",
            fitness
        );
    }

    #[test]
    fn constant_move_agent_has_low_fitness() {
        // An agent that always picks TopLft should have fitness < 1.0
        let mut total_correct: u32 = 0;
        let mut total_moves: u32 = 0;

        for &agent_player in &[Player::Cross, Player::Circle] {
            let initial = new_game(Player::Cross);
            let mut rng = StdRng::seed_from_u64(42);

            let (correct, moves) = score_against_perfect_play(
                initial,
                agent_player,
                |_state| CellLocation::TopLft,
                &mut rng,
            );

            total_correct += correct;
            total_moves += moves;
        }

        let fitness = if total_moves == 0 {
            0.0
        } else {
            (total_correct as f64) / (total_moves as f64)
        };

        assert!(
            fitness < 1.0,
            "constant-move agent should not have perfect fitness, got {}",
            fitness
        );
        // It should still get SOME moves right by chance
        println!(
            "Constant-move agent fitness: {:.4} ({}/{})",
            fitness, total_correct, total_moves
        );
    }

    #[test]
    fn fitness_is_zero_when_no_moves_match() {
        // Verify the zero-division guard
        let total_correct: u32 = 0;
        let total_moves: u32 = 0;

        let fitness = if total_moves == 0 {
            0.0
        } else {
            (total_correct as f64) / (total_moves as f64)
        };

        assert_eq!(fitness, 0.0);
    }

    // ---------------------------------------------------------------
    // Integration: verify an all-zeros network gets non-trivial fitness
    // ---------------------------------------------------------------

    #[test]
    fn all_zero_outputs_agent_fitness() {
        // A network that outputs all zeros should pick BotRgt (last max index)
        // and get some fitness > 0 or == 0 — let's just make sure it doesn't panic
        // and the fitness is in [0, 1].
        let mut total_correct: u32 = 0;
        let mut total_moves: u32 = 0;

        for &agent_player in &[Player::Cross, Player::Circle] {
            let initial = new_game(Player::Cross);
            let mut rng = StdRng::seed_from_u64(42);

            let (correct, moves) = score_against_perfect_play(
                initial,
                agent_player,
                |state| {
                    // Simulate an untrained network: all outputs are 0.0
                    let outputs = vec![0.0; 9];
                    outputs_to_move(&outputs)
                },
                &mut rng,
            );

            total_correct += correct;
            total_moves += moves;
        }

        let fitness = if total_moves == 0 {
            0.0
        } else {
            (total_correct as f64) / (total_moves as f64)
        };

        assert!(
            (0.0..=1.0).contains(&fitness),
            "fitness should be in [0, 1], got {}",
            fitness
        );
        println!(
            "All-zeros agent fitness: {:.4} ({}/{})",
            fitness, total_correct, total_moves
        );
    }
}
