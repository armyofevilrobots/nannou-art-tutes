extern crate rand;

use audrey;
use nannou::prelude::*;
use nannou_audio as audio;
use nannou_audio::Buffer;
use rand::Rng;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
//use rustfft::{FFTplanner, FFT};
use rustfft::algorithm::Radix4;
use rustfft::FFT;
use std::sync::mpsc::{channel, Receiver, Sender};

fn main() {
    // We'll build the window in a moment...
    nannou::app(model).view(view).update(update).run();
}

const BITRATE: u32 = 44100;
const SPEC_BANDS: u32 = 25;
const BANDS: u16 = 4096;
const SAMPLES: usize = 4;
const FFT_OUT_SCALE: f32 = 20.0;

struct Model {
    bands: Vec<f32>,
    music: audio::Stream<Audio>,
    viz_samples: Vec<f32>,
    viz_recv: Receiver<Vec<f32>>,
    third_octaves: Vec<(f32, f32, f32)>,
    third_octave_lookups: Vec<(f32, f32)>,
}

struct Audio {
    song: audrey::read::BufFileReader,
    viz_send: Sender<Vec<f32>>,
}

/* Returns a table of low and high frequencies for
   third octave boundaries for the visualization
   code
*/
fn _calc_third_octaves() -> Vec<(f32, f32, f32)> {
    let mut table = Vec::new();
    let mut f_center: f32 = 55.0; // 4 octaves below middle C
    let linear_spacing = ((BITRATE / 3) as f32 / SPEC_BANDS as f32);
    let f_begin = (BITRATE / 6) as f32;
    println!("F_BEGIN is {}", f_begin);
    for _i in 0..SPEC_BANDS {
        // let f_low = f_center / ((2.0).powf(1.0 / 6.0));
        // let f_high = f_center * (2.0).powf(1.0 / 6.0);
        /// Experiment; let's see if we can extend the range and look better
        // let f_low = f_center / ((2.0).powf(1.0 / 2.0));
        // let f_high = f_center * (2.0).powf(1.0 / 2.0);
        //table.push((f_low, f_center, f_high));
        //f_center = 2.0.powf(1.0 / 3.0) * f_center;
        /// OK, That also looks like crap. Linear grouping?
        let start = f_begin + _i as f32 * linear_spacing;
        let end = f_begin + (_i + 1) as f32 * linear_spacing;
        table.push((start, (start + end) / 2.0, end));
    }
    println!("Table: {:?}", &table);
    table
}
fn _convert_freq_to_table(_octaves: &Vec<(f32, f32, f32)>) -> Vec<(f32, f32)> {
    let mut table = Vec::new();
    for band in 0..SPEC_BANDS {
        table.push((
            _octaves[band as usize].0 * BANDS as f32 / BITRATE as f32,
            _octaves[band as usize].2 * BANDS as f32 / BITRATE as f32,
        ))
    }
    println!("Octave table offsets: {:?}", &table);
    table
}

fn model(_app: &App) -> Model {
    // Build window with specific geom.
    _app.new_window()
        .event(event)
        .with_dimensions(1024, 1024)
        .build()
        .unwrap();
    // Set up the audio
    // Initialise the audio API so we can spawn an audio stream.
    let audio_host = audio::Host::new();

    // Initialise the state that we want to live on the audio thread.
    let assets = _app.assets_path().expect("could not find assets directory");
    // let path = assets.join("joydivision").join("sweep.wav");
    let path = assets
        .join("joydivision")
        .join("14-LoveWillTearUsApart.wav");
    let song = audrey::open(path).expect("failed to load song");
    //println!("Format deets: {:?}", song.description());

    let (viz_send, viz_recv) = channel::<Vec<f32>>();
    let model = Audio { song, viz_send };
    let stream = audio_host
        .new_output_stream(model)
        .render(audio)
        .build()
        .unwrap();

    // Fill those bands
    let mut bands: Vec<f32> = vec![0.0; BANDS as usize];
    let third_octaves = _calc_third_octaves();
    let octave_table = _convert_freq_to_table(&third_octaves);
    Model {
        bands: bands,
        music: stream,
        viz_samples: Vec::new(),
        viz_recv: viz_recv,
        third_octaves: third_octaves,
        third_octave_lookups: octave_table,
    }
}

fn _calc_viz(_buffer: &Vec<f32>) -> Vec<f32> {
    //  println!("Bug is {} samples", _buffer.len());
    let mut input: Vec<Complex<f32>> = (0.._buffer.len())
        .map(|val| Complex::new(_buffer[val] as f32, 0.0))
        .collect();
    // let mut output: Vec<Complex<f32>> = vec![Complex::zero(); _buffer.len()];
    let mut output = input.clone();
    // let mut planner = FFTplanner::new(false);
    let mut _bands: Vec<f32> = vec![0.0; BANDS as usize];
    let buf_size = input.len();
    // let fft = planner.plan_fft(_buffer.len());
    //let mut fft = FFT::new(buf_size, false);
    let mut fft = Radix4::new(BANDS as usize, false);
    fft.process(&mut input, &mut output);
    // for sample in 0..buf_size as usize {
    //println!("OUT: {:?}", output);
    for sample in 0..(BANDS as usize) {
        let x = ((sample as f64 / (BANDS as usize * 2) as f64).powi(2) / (1.0 / 2.0).powi(2)
            * (BANDS as usize * 2) as f64) as usize
            / 2;
        _bands[sample] = (output[x].to_polar().0).sqrt(); //.log(10.0).powi(4);
                                                          //let y = output[x].norm().powi(2);
    }
    _bands
}

fn audio(_audio: &mut Audio, buffer: &mut Buffer) {
    // println!("Buffer deets: {:?}", buffer.sample_rate());
    let mut file_frames = _audio.song.frames::<[f32; 2]>().filter_map(Result::ok);
    let mut viz_vec: Vec<f32> = Vec::new();
    for frame in buffer.frames_mut() {
        let mut tmp_viz: f32 = 0.0;
        let file_frame = &file_frames.next().expect("dead frame");
        for (sample, file_sample) in frame.iter_mut().zip(file_frame) {
            // println!("SAMPLE: {}", file_sample);
            *sample += *file_sample;
            tmp_viz += *file_sample / 2.0;
        }
        viz_vec.push(tmp_viz);
    }
    // println!("VIZ VEC: {:?}", viz_vec);
    //  println!("Send {:?} samples", viz_vec.len());
    _audio.viz_send.send(viz_vec);
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    //  println!("UPDATE!");
    let mut new_samples = Vec::<f32>::new();
    loop {
        match _model.viz_recv.try_recv() {
            Ok(samples_tmp) => new_samples.extend(&samples_tmp),
            _ => break,
        }
    }
    //  println!("Received {} samples", new_samples.len());
    _model.viz_samples.extend(&new_samples);
    // println!(
    //     "We got {} samples for a total of {}",
    //     &new_samples.len(),
    //     &_model.viz_samples.len()
    // );
    //TODO: break into the buckets and then have persistence
    //      via accumulator so it's less jumpy.
    if _model.viz_samples.len() >= (BANDS as usize) {
        let _viz_smp_tmp = _model.viz_samples.split_off(BANDS as usize);
        let results = _calc_viz(&_model.viz_samples);
        _model.viz_samples = Vec::<f32>::new();
        for i in 0..results.len() {
            // _model.bands[i].push(results[i]);
            // if _model.bands[i].len() > SAMPLES {
            //     _model.bands[i].remove(0);
            // }
            _model.bands[i] = _model.bands[i] * 0.75 + results[i] * 0.25;
        }
    }
}

fn event(_app: &App, _model: &mut Model, _event: WindowEvent) {
    match _event {
        MouseReleased(_button) => {}
        _ => (),
    }
}

fn view(_app: &App, _model: &Model, frame: &Frame) {
    let draw = _app.draw();
    frame.clear(BLACK);
    /*
    /// This little section does a real simple spectrum analyzer.
    let mut points = Vec::new(); // where we put the points.
    let x_scale = (1024.0 / BANDS as f32) / (BANDS as f32 / 2.0); // scale so it fills screen, matches log2 entries
    for band in 1..(BANDS / 2) {
        points.push((
            pt2(
                -512.0 + (2.0 * (band as f32).powi(2)) * x_scale,
                -512.0 + 100.0 * _model.bands[band as usize],
            ),
            WHITE,
        ));
    }
    draw.polyline()
        .weight(4.0)
        .join_round()
        .colored_points(points);
    */
    let y_spacing = 1024.0 / (SPEC_BANDS as f32 + 3.0);
    let mut y_ofs = 512.0 - (y_spacing * 2.0); // Some headroom at the top...
    let x_start = -512.0 * 0.75;
    let x_end = -x_start;
    for _band in (1..SPEC_BANDS as usize).rev() {
        y_ofs -= y_spacing;
        let mut points = Vec::new(); // where we put the points.
        let start = _model.third_octave_lookups[_band].0.trunc() as i32;
        let end = _model.third_octave_lookups[_band].1.trunc() as i32;
        let num_samples = end - start;
        let num_points = num_samples + 2;
        let x_spacing = (x_end - x_start) / num_points as f32;
        points.push((pt2(x_start + 0.0, y_ofs + 0.0), WHITE));
        let mut x_ofs = x_start;
        //println!("Pulling band {} from {}->{}", &_band, &start, &end);
        for i in start..end {
            x_ofs += x_spacing;
            let y_scale = (_band as f32).sqrt() * ((3.14 / 2.0) * x_ofs / x_end).cos().powi(4);
            points.push((
                pt2(
                    x_ofs,
                    y_ofs + (y_scale * _model.bands[i as usize]) * FFT_OUT_SCALE,
                ),
                WHITE,
            ));
        }
        points.push((pt2(x_ofs + x_spacing, y_ofs + 0.0), WHITE));

        let polypoints = points.iter().map(|point| point.0);

        //println!("Points: {:?}", points);
        draw.polygon().color(BLACK).points(polypoints);
        draw.polyline()
            .weight(4.0)
            .join_round()
            .colored_points(points);
    }
    draw.to_frame(_app, &frame).unwrap();
}
