extern crate rand;

use nannou::prelude::*;
use rand::Rng;

const TILES_WIDTH: i16 = 32;
const TILES_HEIGHT: i16 = 32;
const TILE_SIZE: i16 = 32;
const MUTATIONS_PER_LOOP: i16 = 1;
fn main() {
    // We'll build the window in a moment...
    nannou::app(model).view(view).update(update).run();
}

enum Tile {
    UpLeft,
    UpRight,
}

impl Tile {
    fn from_u32(val: u32) -> Tile {
        match val {
            0 => Tile::UpLeft,
            1 => Tile::UpRight,
            _ => panic!("Invalid value for a tile (0-1) : {}", val),
        }
    }
}

struct Model {
    width: i16,
    height: i16,
    cells: Vec<Tile>,
}

fn cells_interlocked() -> Vec<Tile> {
    let mut rng = rand::thread_rng();
    let mut cells: Vec<Tile> = Vec::new();
    for i in 0..(TILES_WIDTH * TILES_HEIGHT) {
        cells.push(Tile::from_u32(rng.gen_range(0, 2)));
    }
    cells
}

fn model(_app: &App) -> Model {
    let out = Model {
        width: TILES_WIDTH,
        height: TILES_HEIGHT,
        cells: cells_interlocked(),
    };
    _app.new_window()
        .event(event)
        .with_dimensions(
            (TILE_SIZE * TILES_WIDTH) as u32,
            (TILE_SIZE * TILES_HEIGHT) as u32,
        )
        .build()
        .unwrap();
    out
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    let mut rng = rand::thread_rng();
    for i in 0..MUTATIONS_PER_LOOP {
        _model.cells[rng.gen_range(0, TILES_WIDTH * TILES_HEIGHT) as usize] =
            Tile::from_u32(rng.gen_range(0, 2));
    }
}

fn event(_app: &App, _model: &mut Model, _event: WindowEvent) {
    match _event {
        MouseReleased(_button) => {
            _model.cells = cells_interlocked();
        }
        _ => (),
    }
}

fn view(_app: &App, _model: &Model, frame: &Frame) {
    frame.clear(PURPLE);
    let draw = _app.draw();
    const CENTER: i16 = TILE_SIZE * (TILES_WIDTH / 2);
    const MIDDLE: i16 = TILE_SIZE * (TILES_HEIGHT / 2);
    for x in 0..TILES_HEIGHT {
        for y in 0..TILES_WIDTH {
            match _model.cells[(x + (y * TILES_WIDTH)) as usize] {
                Tile::UpRight => {
                    draw.line()
                        .start(pt2(
                            (x * TILE_SIZE - CENTER - 1).into(),
                            ((y + 1) * TILE_SIZE - MIDDLE + 1).into(),
                        ))
                        .end(pt2(
                            ((x + 1) * TILE_SIZE - CENTER + 1).into(),
                            (y * TILE_SIZE - MIDDLE - 1).into(),
                        ))
                        .weight(4.0)
                        .color(STEELBLUE);
                }
                Tile::UpLeft => {
                    draw.line()
                        .start(pt2(
                            (x * TILE_SIZE - CENTER - 1).into(),
                            (y * TILE_SIZE - MIDDLE - 1).into(),
                        ))
                        .end(pt2(
                            ((x + 1) * TILE_SIZE - CENTER + 1).into(),
                            ((y + 1) * TILE_SIZE - MIDDLE + 1).into(),
                        ))
                        .weight(4.0)
                        .color(STEELBLUE);
                }
            };
        } //width
    } //height
    draw.to_frame(_app, &frame).unwrap();
}
