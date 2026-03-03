#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

mod tictactoe;

fn main() {
    tictactoe::cli::game_loop();
}

// run with `cargo +nightly run --example play_tictactoe`