use std::{error::Error,
          fs::File,
          io::Write};
use curve_sampling as cs;

fn main() -> Result<(), Box<dyn Error>> {
    let path = "/tmp/latex_speed.tex";
    let mut fh = File::create(path)?;
    write!(fh, "\\documentclass[12pt,a4paper]{{article}}\n\
                \\usepackage{{tikz}}\n\
                \\begin{{document}}\n\
                \\begin{{tikzpicture}}\n")?;
    let n = 40_000;
    println!("Run \"latex {}\" measure LaTeX speed with {} points.\n",
             path, n);
    let s = cs::uniform(f64::sin, -6., 6.).n(n).build();
    s.latex().write(&mut fh)?;
    write!(fh, "\\end{{tikzpicture}}\n\
                \\end{{document}}")?;
    Ok(())
}
