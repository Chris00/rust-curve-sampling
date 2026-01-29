//! Fast approximate math functions.
// See https://gist.github.com/aliakseis/3f92a73da2ea6c85683b58e9dced604c

use std::f64::consts::{SQRT_2, FRAC_1_SQRT_2, PI};

pub fn hypot(mut x: f64, mut y: f64) -> f64 {
    x = x.abs();
    y = y.abs();
    let mut t = x.min(y);
    x = x.max(y);
    y = t;
    t = t / x;

    const THRESHOLD: f64 = SQRT_2 - 1.;
    if t > THRESHOLD {
        let x_ = x + y;
        let y_ = x - y;

        t = y_ / x_;
        return x_ * (1. + t * t / 2.) * FRAC_1_SQRT_2;
    }
    x * (1. + t * t / 2.)
}

fn atan(x: f64) -> f64 {
	return x / (1. + 0.3211 * x * x);
}

fn do_atan2(y: f64, x: f64,  abs_x_lt_abs_y: bool) -> f64 {
	if abs_x_lt_abs_y {
		let z = x / y;
		let atan_z = atan(z);

		if y > 0. {
			return -atan_z + PI / 2.;
		}
		else {
			return -atan_z - PI / 2.;
		}
	}

	let z = y / x;
	let atan_z = atan(z);

	if x > 0. {
		return atan_z;
	}
	else if y >= 0. {
		return atan_z + PI;
	}
	atan_z - PI
}

fn do_atan2_rotated(y: f64, x: f64) -> f64 {
	if x.abs() < y.abs() {
		let z = x / y;
		let  atan_z = atan(z);

		if y > 0. {
			return -atan_z + PI / 2. - PI / 4.;
		}
		else {
			return -atan_z - PI / 2. - PI / 4. ;
		}
	}

	let z = y / x;
	let atan_z = atan(z);

	if x > 0. {
		return atan_z - PI / 4.;
	}
	atan_z + PI - PI / 4.
}

// The maximum absolute error across all intervals is less than 0.00012.
pub fn atan2(y: f64, x: f64) -> f64 {
	const TAN_PI_OVER_8: f64 = 0.41421356237;

	if x == 0. {
		if y > 0. {
            return PI / 2.
        } else if y < 0. {
            return -PI / 2.
        } else {
            return 0.
        }
	}

	let abs_x = x.abs();
	let abs_y = y.abs();

	let abs_x_lt_abs_y = abs_x < abs_y;

	if if abs_x_lt_abs_y { abs_x > TAN_PI_OVER_8 * abs_y } else { abs_y > TAN_PI_OVER_8 * abs_x } {
		let x_ = x - y;
		let y_ = x + y;
		return do_atan2_rotated(y_, x_);
	}

	return do_atan2(y, x, abs_x_lt_abs_y);
}
