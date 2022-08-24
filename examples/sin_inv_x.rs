#![feature(closure_lifetime_binder)]

use std::{error::Error,
          fs::File,
          io::Write};
use curve_sampling::Sampling;


type R = Result<(), Box<dyn Error>>;

fn main() -> R {
    let mut fh = File::create("/tmp/graphs.gp")?;
    write!(fh, "set terminal pngcairo\n\
                set grid\n")?;
    let mut d = 0;
    let mut save = |s: &Sampling, n, title| -> R {
        d += 1;
        let fname = format!("/tmp/graph{}.dat", d);
        s.write(&mut File::create(&fname)?)?;
        write!(fh, "set output \"graph{}.png\"\n\
                    plot '{}' with l lt 1 lw 2 title \"{} ({} pts)\"\n",
               d, &fname, title, n)?;
        write!(fh, "set output \"graph{}_p.png\"\n\
                    plot '{}' with l lt 5 lw 2 title \"{}\", \
                    '{}' with p lt 1 pt 5 ps 0.2 title \"points ({})\"\n",
               d, &fname, title, &fname, n)?;
        Ok(())
    };

    let f = |x: f64| x * (1. / x).sin();
    let s = Sampling::fun(f, -0.4, 0.4).n(227).build();
    save(&s, 227, "x sin(1/x)")?;
    let s = Sampling::fun(f, -0.4, 0.4).n(389).build();
    save(&s, 389, "x sin(1/x)")?;

    let s = Sampling::fun(|x: f64| (1. / x).sin(), -0.4, 0.4).n(391).build();
    save(&s, 391, "sin(1/x)")?;

    Ok(())
}
