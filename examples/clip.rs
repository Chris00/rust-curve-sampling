use std::{error::Error,
          fs::File,
          f64::NAN};
use curve_sampling as cs;

fn main() -> Result<(), Box<dyn Error>> {
    let s = cs::Sampling::from_iter(
        [[0., -0.5], [1.5, 1.], [0.2, 0.5], [0.3, 1.5], [1., 0.6],
         [NAN, NAN], [-0.5, 0.5], [-1., 0.], [0.5, 0.5]]);
    s.write(&mut File::create("/tmp/clip0.dat")?)?;
    s.latex().write(&mut File::create("/tmp/clip0.tex")?)?;
    let s1 = s.clip(cs::BoundingBox { xmin: 0., xmax: 1.,
                                      ymin: 0., ymax: 1. });
    s1.write(&mut File::create("/tmp/clip1.dat")?)?;
    s1.latex().write(&mut File::create("/tmp/clip1.tex")?)?;

    Ok(())
}
