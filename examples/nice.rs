use std::{error::Error,
          f64::consts::PI,
          fs::File,
          io::BufWriter};
use curve_sampling::Sampling;

fn main() -> Result<(), Box<dyn Error>> {
    let f = |t: f64| [t.cos(), (2. * t).sin()];
    let s = Sampling::param(f, 0., 2. * PI).build();
    s.write(&mut BufWriter::new(File::create("/tmp/nice1.dat")?))?;

    let f = |x: f64| (- x.powi(2)).exp();
    let s = Sampling::fun(f, -2.5, 2.5).n(53).build();
    s.write(&mut BufWriter::new(File::create("/tmp/nice2.dat")?))?;
    Ok(())
}
