//!

use std::{fmt::{self, Display, Formatter},
          io::{self, Write},
          iter::Iterator};
use rand::prelude::*;
use rgb::*;

/// All fields MUST be finite (in samplings).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    t: f64, // "time" parameter.
    x: f64,
    y: f64,
    cost: f64, // Cache the cost
}

/// A 2D point with coordinates (`x`, `y`) supposed to be given by a
/// function evaluated at "time" `t`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CutOr<T> {
    Pt(T), // Decouple the point implementation.
    Undef { t: f64 }, // Function not defined at this `t`.  Visually
    // this is a cut in the path but with extra information.
    Cut, // Cut in the path, for example when clipping
}

/// A box \[`xmin`, `xmax`\] × \[`ymin`, `ymax`\].
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

impl BoundingBox {
    /// Return `true` if the bounding box has a non-empty interior.
    #[inline]
    pub fn is_empty(&self) -> bool {
        !(self.xmin < self.xmax && self.ymin < self.ymax) // NAN ⟹ empty
    }

    /// Return `true` if the point `p` belongs to `bb` (possibly on
    /// the boundary).
    #[inline]
    fn contains(&self, p: Point) -> bool {
        self.xmin <= p.x && p.x <= self.xmax
            && self.ymin <= p.y && p.y <= self.ymax
    }

    /// Return the smaller bounding-box containing both `self` and
    /// `other`.
    #[inline]
    pub fn hull(&self, other: &Self) -> Self {
        BoundingBox { xmin: self.xmin.min(other.xmin),
                      xmax: self.xmax.max(other.xmax),
                      ymin: self.ymin.min(other.ymin),
                      ymax: self.ymax.max(other.ymax) }
    }
}


////////////////////////////////////////////////////////////////////////
//
// Sampling datastructure

/// Safe interface to the sampling datastructure
mod sampling {
    use std::{collections::BinaryHeap,
              cmp::Ordering,
              marker::PhantomData,
              ptr};
    use super::{Point, CutOr, BoundingBox};

    /// `f64` numbers at the exclusion of NaN.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct NotNAN(f64);

    impl Eq for NotNAN {}

    impl PartialOrd for NotNAN {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl Ord for NotNAN {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct Segment {
        p0: *const Point,
        p1: *const Point,
        cost: NotNAN,
    }

    // Only compare the costs (equality has to be compatible with ordering).
    impl PartialEq for Segment {
        fn eq(&self, other: &Self) -> bool {
            self.cost == other.cost
        }
    }

    impl Eq for Segment {}

    impl PartialOrd for Segment {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(Segment::cmp(self, other))
        }
    }

    impl Ord for Segment {
        fn cmp(&self, other: &Self) -> Ordering {
            self.cost.cmp(&other.cost)
        }
    }

    /// Double linked list structure holding points of cuts.
    struct Node {
        p: CutOr<Point>,
        prev: *mut Node,
        next: *mut Node,
    }

    /// A 2D sampling.  This can be thought as a path, with possible
    /// "cuts" because of discontinuities or leaving the domain of the
    /// (parametric) function describing the path.
    pub struct Sampling {
        // If `pq` is empty but not `path`, it means that the costs
        // need to to be updated.
        pq: BinaryHeap<Segment>, // Priority queue of segments.
        path_begin: *mut Node,
        path_end: *mut Node,
        vp: Option<BoundingBox>, // viewport (zone of interest)
    }

    impl Drop for Sampling {
        fn drop(&mut self) {
            let mut node_ptr = self.path_begin;
            while !node_ptr.is_null() {
                let node = unsafe { Box::from_raw(node_ptr) };
                node_ptr = node.next;
            }
        }
    }

    impl Sampling {
        /// Return `true` if the sampling is empty.
        pub fn is_empty(&self) -> bool {
            self.path_begin.is_null()
        }

        /// Create an empty sampling.
        #[inline]
        pub(crate) fn empty() -> Self {
            Self { pq: BinaryHeap::new(),
                   path_begin: ptr::null_mut(),
                   path_end: ptr::null_mut(),
                   vp: None }
        }

        #[inline]
        pub(crate) fn singleton(t: f64, x: f64, y: f64) -> Self {
            let p = Point { t, x, y, cost: 0. };
            let node = Box::into_raw(Box::new(
                Node { p: CutOr::Pt(p),
                       prev: ptr::null_mut(),
                       next: ptr::null_mut() }));
            Self { pq: BinaryHeap::new(),
                   path_begin: node,
                   path_end: node,
                   vp: None }
        }

        /// Push `p` at the end of the path, not checking for the
        /// validity of `p` and not updating the priority queue.  This
        /// is an easy way to build a path incrementally.
        #[inline]
        pub(crate) fn push_unchecked(&mut self, p: CutOr<Point>) {
            let node = Box::into_raw(Box::new(
                Node { p,
                       prev: self.path_end,
                       next: ptr::null_mut() }));
            if !self.path_end.is_null() {
                unsafe { (*self.path_end).next = node; }
            } else {
                self.path_begin = node;
            }
            self.path_end = node;
        }

        /// Assumes the path is NOT empty.
        #[inline]
        pub(crate) fn pop_unchecked(&mut self) -> CutOr<Point> {
            let node = unsafe { Box::from_raw(self.path_end) };
            self.path_end = node.prev;
            unsafe { (*node.prev).next = ptr::null_mut() };
            node.p
        }

        #[inline]
        pub(crate) fn remove_trailing_cuts(&mut self) {
            use CutOr::*;
            while !self.path_end.is_null()
                && matches!(unsafe { &*self.path_end }.p,
                           Undef { t: _ } | Cut) {
                    self.pop_unchecked();
            }
        }


        #[inline]
        pub(crate) fn set_vp(&mut self, bb: BoundingBox) {
            self.vp = Some(bb);
        }

        #[inline]
        pub(crate) fn vp(&self) -> Option<BoundingBox> {
            self.vp
        }

        pub(crate) fn iter_points(&self) -> IterPoints {
            IterPoints { next: self.path_begin }
        }

        /// Iterate on the points (and cuts) of the path.  More
        /// precisely, a path is made of continuous segments whose
        /// points are given by contiguous values `Some(p)`
        /// interspaced by `None`.  Two `None` never follow each
        /// other.  Isolated points `p` are given by ... `None`,
        /// `Some(p)`, `None`,...
        pub fn iter(&self) -> Iter<'_> {
            Iter { next: self.path_begin,
                   prev_is_cut: true,
                   marker: PhantomData }
        }

    }

    /// "Raw" iterator on the list of points composing the path.
    pub(crate) struct IterPoints {
        next: *const Node
    }

    impl Iterator for IterPoints {
        type Item = CutOr<Point>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.next.is_null() { return None }
            let node = unsafe {&*self.next as &Node };
            self.next = node.next;
            Some(node.p)
        }
    }


    /// Iterator on the points of the [`Sampling`].
    /// See [`Sampling::iter`] for more information.
    pub struct Iter<'a> {
        next: *const Node,
        prev_is_cut: bool,
        marker: PhantomData<&'a Node>,
    }

    impl<'a> Iterator for Iter<'a> {
        type Item = Option<[f64; 2]>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.next.is_null() { return None }
            let node = unsafe { &*self.next as &Node };
            self.next = node.next;
            use CutOr::*;
            match node.p {
                Pt(p) => {
                    self.prev_is_cut = false;
                    Some(Some([p.x, p.y]))
                }
                Undef{ t: _} | Cut => {
                    if self.prev_is_cut { None }
                    else { self.prev_is_cut = true; Some(None) }
                }
            }
        }
    }
}

pub use sampling::{Sampling, Iter};


/// Intersection of a segment with the bounding box.
#[derive(Debug)]
enum Intersection {
    Empty,
    Pt(Point),
    Seg(Point, Point),
}

impl Sampling {
    /// Return the smallest rectangle enclosing all the points of the
    /// sampling `self`.  If the path is empty, the "min" fields of
    /// the bounding box are set to +∞ and "max" fields to -∞.
    pub fn bounding_box(&self) -> BoundingBox {
        let mut bb = BoundingBox {
            xmin: f64::INFINITY,  xmax: f64::NEG_INFINITY,
            ymin: f64::INFINITY,  ymax: f64::NEG_INFINITY };
        for p_opt in self.iter() {
            match p_opt {
                Some([x, y]) => {
                    if x < bb.xmin { bb.xmin = x }
                    else if bb.xmax < x { bb.xmax = x };
                    if y < bb.ymin { bb.ymin = y }
                    else if bb.ymax < y { bb.ymax = y };
                }
                None => ()
            }
        }
        bb
    }

    /// Assume `p0` ∈ `bb` and `p1` ∉ `bb`.  Return the point
    /// intersecting the boundary of `bb`.  If the intersection point
    /// is the same as `p0`, return `None`
    #[inline]
    fn intersect(p0: Point, p1: Point, bb: BoundingBox) -> Option<Point> {
        let mut t = 1.; // t ∈ [0, 1]
        let dx = p1.x - p0.x; // May be 0.
        let r = (if dx >= 0. {bb.xmax} else {bb.xmin} - p0.x) / dx;
        if r < t { t = r }
        let dy = p1.y - p0.y; // May be 0.
        let r = (if dy >= 0. {bb.ymax} else {bb.ymin} - p0.y) / dy;
        if r < t { t = r };
        if t <= 1e-14 {
            None
        } else {
            Some(Point { t: p0.t + t * (p1.t - p0.t),
                         x: p0.x + t * dx,
                         y: p0.y + t * dy,
                         cost: 0. })
        }
    }

    /// Assume `p0` ∉ `bb` and `p1` ∉ `bb` (thus, not even on the
    /// boundary of `bb`) and `p0` ≠ `p1`.  Return the endpoints of
    /// the segment intersecting the boundary of `bb` if any.  The
    /// "parameter direction" of `p0`, `p1` is preserved.
    #[inline]
    fn intersect_seg(p0: Point, p1: Point, bb: BoundingBox) -> Intersection {
        let mut t0 = 0.; // t0 ∈ [0, 1]
        let mut t1 = 1.; // t1 ∈ [0, 1]
        let dx = p1.x - p0.x; // may be 0.
        let r0;
        let r1; // r0 ≤ r1 or NAN if on x-boundary lines.
        if dx >= 0. {
            r0 = (bb.xmin - p0.x) / dx;
            r1 = (bb.xmax - p0.x) / dx;
        } else {
            r0 = (bb.xmax - p0.x) / dx;
            r1 = (bb.xmin - p0.x) / dx;
        }
        if r0 > 1. || r1 < 0. { return Intersection::Empty }
        if r0 > 0. { t0 = r0 } // if r0 is NAN, keep the whole segment
        if r1 < 1. { t1 = r1 }
        let dy = p1.y - p0.y; // may be 0.
        let r0;
        let r1;
        if dy >= 0. {
            r0 = (bb.ymin - p0.y) / dy;
            r1 = (bb.ymax - p0.y) / dy;
        } else {
            r0 = (bb.ymax - p0.y) / dy;
            r1 = (bb.ymin - p0.y) / dy;
        }
        if r0 > t1 || r1 < t0 { return Intersection::Empty }
        if r0 > t0 { t0 = r0 }
        if r1 < t1 { t1 = r1 }
        if t0 < t1 { // segment not reduced to a point
            let dt = p1.t - p0.t;
            let q0 = Point { t: p0.t + t0 * dt,
                             x: p0.x + t0 * dx,
                             y: p0.y + t0 * dy,
                             cost: 0. };
            let q1 = Point { t: p0.t + t1 * dt,
                             x: p0.x + t1 * dx,
                             y: p0.y + t1 * dy,
                             cost: 0. };
            Intersection::Seg(q0, q1)
        } else if t0 == t1 {
            let q0 = Point { t: p0.t + t0 * (p1.t - p0.t),
                             x: p0.x + t0 * dx,
                             y: p0.y + t0 * dy,
                             cost: 0. };
            Intersection::Pt(q0)
        } else {
            Intersection::Empty
        }
    }

    /// Returns the sampling `self` but clipped to the 2D box `bb`.  A
    /// path that crosses the boundary will get additional nodes at
    /// the points of crossing and the part outside the bounding box
    /// will be dropped.  (Thus a path entirely out of the bounding
    /// box will be removed.)
    #[must_use]
    pub fn clip(&self, bb: BoundingBox) -> Self {
        use CutOr::*;
        if bb.is_empty() {
            panic!("curve_sampling::clip: box with empty interior {:?}", bb)
        }
        if self.is_empty() { return Sampling::empty() };
        let mut s = Sampling::empty();
        let mut p0_opt = Cut;
        let mut p0_inside = false;
        let mut prev_cut = true; // New path ("virtual" cut)
        for p1_opt in self.iter_points() {
            if prev_cut {
                // A cut was pushed at the previous step.  This may be
                // because the original path was just cut (`p0_opt =
                // None`) or because the previous segment was cut
                // (`p0_opt = Some(p0)` with `p0` ∉ `bb`) or because
                // there was a cut a `p0` is the possible next start.
                match (p0_opt, p1_opt) {
                    (Pt(p0), Pt(p1)) => {
                        let p1_inside = bb.contains(p1);
                        if p0_inside { // p0 ∈ bb with cut before
                            s.push_unchecked(p0_opt);
                            if p1_inside {
                                s.push_unchecked(p1_opt);
                                prev_cut = false;
                            } else {
                                if let Some(p) = Self::intersect(p0, p1, bb) {
                                    s.push_unchecked(Pt(p))
                                };
                                s.push_unchecked(Cut);
                            }
                        } else if p1_inside { // p0 ∉ bb, p1 ∈ bb
                            if let Some(p) = Self::intersect(p1, p0, bb) {
                                s.push_unchecked(Pt(p)); // p ≠ p1
                            }
                            s.push_unchecked(p1_opt);
                            prev_cut = false;
                        } else { // p0, p1 ∉ bb but maybe intersection
                            match Self::intersect_seg(p0, p1, bb) {
                                Intersection::Seg(q0, q1) => {
                                    s.push_unchecked(Pt(q0));
                                    s.push_unchecked(Pt(q1));
                                    s.push_unchecked(Cut);
                                }
                                Intersection::Pt(p) => {
                                    s.push_unchecked(Pt(p));
                                    s.push_unchecked(Cut);
                                }
                                Intersection::Empty => (),
                            }
                        }
                        p0_opt = p1_opt;
                        p0_inside = p1_inside;
                    }
                    (Undef { t: _} | Cut, Pt(p1)) => {
                        p0_opt = p1_opt;
                        p0_inside = bb.contains(p1);
                    }
                    (_, Undef { t: _} | Cut) => p0_opt = p1_opt,
                }
            } else {
                // Previous step was not a cut which also implies that
                // `p0_opt = Pt(p0)` with `p0` ∈ `bb`, and `p0` is
                // already present in the final `path`.
                let p0 = match p0_opt { Pt(p) => p,
                                        _ => unreachable!() };
                p0_opt = p1_opt;
                match p1_opt {
                    Undef { t: _ } | Cut => {
                        s.push_unchecked(Cut);
                        prev_cut = true
                    }
                    Pt(p1) => {
                        p0_inside = bb.contains(p1); // update for next step
                        if p0_inside { // p0, p1 ∈ bb
                            s.push_unchecked(p1_opt);
                        } else { // p0 ∈ bb, p1 ∉ bb
                            if let Some(p) = Self::intersect(p0, p1, bb) {
                                s.push_unchecked(Pt(p));
                            }
                            s.push_unchecked(Cut);
                            prev_cut = true;
                        }
                    }
                }
            }
        }
        s.remove_trailing_cuts();
        s.set_vp(bb);
        s
    }
}


impl<T> From<T> for Sampling
where T: IntoIterator<Item = [f64;2]> {
    /// Return an iterator from the points.  Points with non-finite
    /// coordinates are interpreted as cuts.
    fn from(points: T) -> Self {
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        let mut s = Sampling::empty();
        for (i, [x, y]) in points.into_iter().enumerate() {
            use CutOr::*;
            let t = i as f64;
            if x.is_finite() && y.is_finite() {
                if x < xmin { xmin = x }
                if x > xmax { xmax = x }
                if y < ymin { ymin = y }
                if y > ymax { ymax = y }
                s.push_unchecked(Pt(Point { t, x, y, cost: 0. }));
            } else {
                s.push_unchecked(Undef { t });
            }
        }
        s.set_vp(BoundingBox { xmin, xmax, ymin, ymax });
        s
    }
}

////////////////////////////////////////////////////////////////////////
//
// Defining a sampling with standard options & checks

/// Define a structure with standard fields, standard options, and a
/// function to generate it.
macro_rules! new_sampling_fn {
    // Function to init the struct.
    ($(#[$docfn: meta])*, $(#[$docfn_extra: meta])* $fun: ident -> $ft: ty,
     // The structure to hold the options (and other fields).
     $(#[$doc: meta])* $struct: ident
    ) => {
        impl Sampling {
            $(#[$docfn])*
            ///
            /// Panics if `a` or `b` is not finite.
            ///
            $(#[$docfn_extra])*
            #[must_use]
            pub fn $fun<F>(f: F, a: f64, b: f64) -> $struct<F>
            where F: FnMut(f64) -> $ft {
                if !a.is_finite() {
                    panic!("curve_sampling::{}: a = {} must be finite",
                           stringify!($fun), a);
                }
                if !b.is_finite() {
                    panic!("curve_sampling::{}: b = {} must be finite",
                           stringify!($fun), b);
                }
                $struct { f, a, b,  // Order of `a`, `b` reflect orientation
                          n: 100,
                          viewport: None,
                          init: vec![],
                          init_pt: vec![],
                }
            }
        }

        $(#[$doc])*
        pub struct $struct<F> {
            f: F,  a: f64,  b: f64,
            n: usize,
            viewport: Option<BoundingBox>,
            init: Vec<f64>,
            init_pt: Vec<(f64, $ft)>
        }

        impl<F> $struct<F> {
            /// Set the maximum number of evaluations of the function
            /// to build the sampling.  Panic if `n < 2`.
            pub fn n(mut self, n: usize) -> Self {
                if n < 2 {
                    panic!("curve_sampling: n = {} must at least be 2", n)
                }
                self.n = n;
                self
            }

            /// Set the zone of interest for the sampling.  Segments
            /// that end up outside this box will not be refined.
            pub fn viewport(mut self, vp: BoundingBox) -> Self {
                self.viewport = Some(vp);
                self
            }

            /// Add initial values of `t` such that `f(t)` (see [`
            #[doc = stringify!(Sampling::$fun)]
            /// `]) must be included into the sampling in addition to
            /// the `n` evaluations.  Only the values between `a` and
            /// `b` are taken into account (other values are ignored).
            pub fn init<'a, I>(mut self, ts: I) -> Self
            where I: IntoIterator<Item = &'a f64> {
                for &t in ts {
                    if self.a <= t && t <= self.b { // ⟹ t is finite
                        self.init.push(t);
                    }
                }
                self
            }

            /// Add initial points `(t, f(t))` to include into the
            /// sampling in addition to the `n` evaluations.  This
            /// allows you to use previous evaluations of `f`.  Only
            /// the couples with first coordinate `t` between `a` and
            /// `b` (see [`
            #[doc = stringify!(Sampling::$fun)]
            /// `]) are considered (other values are ignored).
            pub fn init_pt<'a, I>(mut self, pts: I) -> Self
            where I: IntoIterator<Item = &'a (f64, $ft)> {
                for &p in pts {
                    if self.a <= p.0 && p.0 <= self.b { // ⟹ p.0 is finite
                        self.init_pt.push(p);
                    }
                }
                self
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Uniform sampling

new_sampling_fn!(
    /// Create a sampling for the graph of `f` on the interval
    /// \[`a`, `b`\] with evenly spaced values of the argument.
    ,
    /// # Example
    ///
    /// ```
    /// use std::fs::File;
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::uniform(|x| x.sin(), 0., 4.).build();
    /// s.write(&mut File::create("target/uniform.dat")?)?;
    /// # Ok(()) }
    /// ```
    uniform -> f64,
    /// Uniform sampling options.  See [`Sampling::uniform`].
    Uniform);

macro_rules! uniform_sampling {
    // `point_of_f` and `point_of_pt` must also validate the outputs
    // (so they are finite).
    ($self: ident, $point_of_f: ident, $point_of_pt: ident,
     $push_eval: ident, $pt_is_valid: ident) => {
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        let mut path = Vec::with_capacity(
            $self.init.len() + $self.init_pt.len() + $self.n);
        // `t` ∈ \[`a`, `b`\] already checked by [`init`] and [`init_pt`].
        for &t in &$self.init { path.push($point_of_f!(t)); }
        for &p in &$self.init_pt { path.push($point_of_pt!(p)); }
        $push_eval!(path);
        if $self.a < $self.b {
            path.sort_unstable_by(|p1, p2| {
                // We know that `t1` and `t2` are finite.
                p1.t.partial_cmp(&p2.t).unwrap() });
        } else {
            path.sort_unstable_by(|p1, p2| {
                p2.t.partial_cmp(&p1.t).unwrap() });
        }
        let mut s = Sampling::empty();
        for &p in path.iter() {
            use CutOr::*;
            if $pt_is_valid!(p) {
                if p.x < xmin { xmin = p.x }
                if p.x > xmax { xmax = p.x }
                if p.y < ymin { ymin = p.y }
                if p.y > ymax { ymax = p.y }
                s.push_unchecked(Pt(p))
            } else {
                s.push_unchecked(Undef { t: p.t })
            }
        }
        //remove_trailing_cuts(&mut path);
        s.set_vp(BoundingBox { xmin, xmax, ymin, ymax });
        s
    }
}

macro_rules! pt_of_couple {
    ($p: expr) => {{
        let (x, y) = $p;
        Point { t: x, x, y, cost: 0. }
    }}}

macro_rules! y_is_valid { ($p: expr) => { $p.y.is_finite() } }

impl<F> Uniform<F>
where F: FnMut(f64) -> f64 {
    pub fn build(&mut self) -> Sampling {
        if self.a == self.b {
            let y = (self.f)(self.a); // `a` is finite by previous tests
            if y.is_finite() {
                return Sampling::singleton(self.a, self.a, y);
            } else {
                return Sampling::empty()
            }
        }
        macro_rules! f { ($t: ident) => {
            Point { t: $t, x: $t, y: (self.f)($t), cost: 0. }
        } }
        let dt = (self.b - self.a) / (self.n - 1) as f64;
        macro_rules! push_eval { ($path: ident) => {
            for i in 0 .. self.n {
                let t = self.a + i as f64 * dt;
                $path.push(f!(t)); }
        } }
        uniform_sampling!{self, f, pt_of_couple, push_eval, y_is_valid}
    }
}

////////////////////////////////////////////////////////////////////////
//
// Cost

////////////////////////////////////////////////////////////////////////
//
// Function sampling

new_sampling_fn!(
    /// Create a sampling of the *graph* of `f` on the interval
    /// \[`a`, `b`\] by evaluating `f` at `n` points.
    ,
    /// # Example
    ///
    /// ```
    /// use std::fs::File;
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::fun(|x| x.sin(), 0., 4.).build();
    /// s.write(&mut File::create("target/fun.dat")?)?;
    /// # Ok(()) }
    /// ```
    fun -> f64,
    /// Options for sampling a function ℝ → ℝ.  See [`Sampling::fun`].
    Fun);

impl<F> Fun<F>
where F: FnMut(f64) -> f64 {
    fn almost_uniform(&mut self, n: usize) -> Sampling {
        macro_rules! f { ($t: expr) => {{
            let t = $t;
            Point { t, x: t, y: (self.f)(t), cost: 0. }
        }} }
        macro_rules! pt { ($p: expr) => {{
            let (x, y) = $p;
            Point { t: x, x, y, cost: 0. }
        }} }
        let dt = (self.b - self.a) / (n - 1) as f64;
        let mut rng = rand::thread_rng();
        macro_rules! push_eval_random { ($path: ident) => {
            $path.push(f!(self.a));
            $path.push(f!(self.a + 0.0625 * dt));
            for i in 0 .. n - 4 {
                let j = i as f64 + rng.gen::<f64>() * 0.125 - 0.0625;
                $path.push(f!(self.a + j * dt));
            }
            $path.push(f!(self.b - 0.0625 * dt));
            $path.push(f!(self.b));
        } }
        uniform_sampling!{self, f, pt, push_eval_random, y_is_valid}
    }

    /// Return the sampling.
    pub fn build(&mut self) -> Sampling {
        if self.a == self.b {
            let y = (self.f)(self.a);
            if y.is_finite() {
                return Sampling::singleton(self.a, self.a, y);
            } else {
                return Sampling::empty()
            }
        }
        let n0 = (self.n / 10).min(10);
        let s = self.almost_uniform(n0);

        s
    }
}


new_sampling_fn!(
    /// Create a sampling of the *image* of `f` on the interval
    /// \[`a`, `b`\] by evaluating `f` at `n` points.
    ,
    /// # Example
    ///
    /// ```
    /// use std::fs::File;
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::param(|t| [t.sin(), t.cos()], 0., 4.).build();
    /// s.write(&mut File::create("target/param.dat")?)?;
    /// # Ok(()) }
    /// ```
    param -> [f64; 2],
    /// Options for sampling a function ℝ → ℝ.  See [`Sampling::param`].
    Param);

impl<F> Param<F>
where F: FnMut(f64) -> [f64; 2] {
    fn almost_uniform(&mut self, n: usize) -> Sampling {
        macro_rules! f { ($t: expr) => {{
            let t = $t;
            let [x, y] = (self.f)(t);
            Point { t, x, y, cost: 0. }
        }}}
        macro_rules! pt { ($p: expr) => {{
            let (t, [x,y]) = $p;
            Point { t, x, y, cost: 0. }
        }}}
        let dt = (self.b - self.a) / (n - 1) as f64;
        let mut rng = rand::thread_rng();
        macro_rules! push_eval_random { ($path: ident) => {
            $path.push(f!(self.a));
            $path.push(f!(self.a + 0.0625 * dt));
            for i in 0 .. n - 4 {
                let j = i as f64 + rng.gen::<f64>() * 0.125 - 0.0625;
                $path.push(f!(self.a + j * dt));
            }
            $path.push(f!(self.b - 0.0625 * dt));
            $path.push(f!(self.b));
        } }
        macro_rules! xy_is_valid { ($p: expr) => {
            $p.x.is_finite() && $p.y.is_finite() }}
        uniform_sampling!{self, f, pt, push_eval_random, xy_is_valid}
    }

    /// Return the sampling.
    pub fn build(&mut self) -> Sampling {
        if self.a == self.b {
            let [x, y] = (self.f)(self.a);
            if x.is_finite() && y.is_finite() {
                return Sampling::singleton(self.a, x , y);
            } else {
                return Sampling::empty()
            }
        }
        let n0 = (self.n / 10).min(10);
        let s = self.almost_uniform(n0);

        s
    }
}


////////////////////////////////////////////////////////////////////////
//
// Output

/// LaTeX output.
///
/// # Example
///
/// ```
/// use std::fs::File;
/// use curve_sampling::Sampling;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let s = Sampling::from([[0., 0.], [1., 1.]]);
/// s.latex().write(&mut File::create("target/sampling.tex")?)?;
/// # Ok(()) }
/// ```
pub struct LaTeX<'a> {
    sampling: &'a Sampling,
    n: usize,
    color: Option<RGB8>,
    arrow: Option<&'a str>,
    arrow_pos: Option<f64>,
}

impl<'a> LaTeX<'a> {
    #[inline]
    fn new(s: &'a Sampling) -> Self {
        Self { sampling: s,  n: 20_000,  color: None,
               arrow: None,  arrow_pos: None }
    }

    /// Set the maximum number of points of a PGF path to `n`.  If it
    /// contains more than `n` points, the sampling curve is drawn as
    /// several PGF paths.  Default: 20_000.
    pub fn n(&mut self, n: usize) -> &mut Self {
        self.n = n;
        self
    }

    /// Set the color of the curve to `color`.  If not specified the
    /// active LaTeX color will be used.
    pub fn color(&mut self, color: RGB8) -> &mut Self {
        self.color = Some(color);
        self
    }

    /// Set the type of arrow to draw to `arrow`.
    /// See the documentation of `\pgfsetarrowsend` in the
    /// [TikZ manual](https://tikz.dev/base-actions#sec-103.3).
    pub fn arrow(&mut self, arrow: &'a str) -> &mut Self {
        self.arrow = Some(arrow);
        self
    }

    /// The position of the arrow as a percent of the curve length (in
    /// the interval \[0.,1.\]).  If [`LaTeX::arrow`] is specified but
    /// not this, it defaults to `0.5`.
    pub fn arrow_pos(&mut self, arrow_pos: f64) -> &mut Self {
        if ! arrow_pos.is_finite() {
            panic!("curve_sampling::LaTeX::arrow_pos: \
                    position must be finite");
        }
        self.arrow_pos = Some(arrow_pos.clamp(0., 1.));
        self
    }

    /// Write the sampling with lines segments.
    fn write_with_lines(&self, f: &mut impl Write) -> Result<(), io::Error> {
        let mut n = 0;
        let mut new_path = true;
        for p in self.sampling.iter() { match p {
            Some([x, y]) => {
                n += 1;
                if new_path {
                    write!(f, "\\pgfpathmoveto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y)?
                } else if n >= self.n {
                    write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n\
                               \\pgfusepath{{stroke}}\n\
                               \\pgfpathmoveto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y, x, y)?;
                    n = 0;
                } else {
                    write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y)?
                }
                new_path = false;
            }
            None =>  {
                write!(f, "\\pgfusepath{{stroke}}\n")?;
                n = 0;
                new_path = true;
            }
        }}
        Ok(())
    }

    /// Write the path, each continuous sub-path with an arrow.
    fn write_with_arrows(&self, f: &mut impl Write, arrow: &str,
                         arrow_pos: f64) -> Result<(), io::Error> {
        // Compute the length of all sub-paths.
        let mut lens = vec![];
        let mut prev_pt: Option<[f64; 2]> = None;
        let mut cur_len = 0.;
        for p in self.sampling.iter() { match p {
            Some([x, y]) => {
                if let Some([x0, y0]) = prev_pt {
                    cur_len += (x - x0).hypot(y - y0);
                }
                prev_pt = p;
            }
            None => {
                lens.push(arrow_pos * cur_len);
                prev_pt = None;
                cur_len = 0.;
            }
        }}
        lens.push(arrow_pos * cur_len);
        if lens.is_empty() { return Ok(()) }
        let mut lens = lens.iter();
        let mut rem_len = *lens.next().unwrap(); // remaining before arrow
        prev_pt = None;
        let mut n = 0;
        for p in self.sampling.iter() { match p {
            Some([x, y]) => {
                n += 1;
                if let Some([x0, y0]) = prev_pt {
                    let dx = x - x0;
                    let dy = y - y0;
                    let l = dx.hypot(dy);
                    if rem_len <= l {
                        write!(f, "\\pgfusepath{{stroke}}\n")?;
                        // Drawing a long path with an arrow specified is
                        // extremely expensive.  Just draw the current segment.
                        let pct = rem_len / l;
                        if pct <= 1e-14 {
                            write!(f, "\\pgfsetarrowsstart{{{}}}\n\
                                       \\pgfpathmoveto{{\\pgfpointxy
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfpathlineto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfusepath{{stroke}}\n\
                                       \\pgfsetarrowsend{{}}\n\
                                       \\pgfpathmoveto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n",
                                   arrow, x0, y0, x, y, x, y)?;
                        } else {
                            let xm = x0 + pct * dx;
                            let ym = y0 + pct * dy;
                            write!(f, "\\pgfsetarrowsend{{{}}}\n\
                                       \\pgfpathmoveto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfpathlineto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfusepath{{stroke}}\n\
                                       \\pgfsetarrowsend{{}}\n\
                                       \\pgfpathmoveto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfpathlineto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n",
                                   arrow, x0, y0, xm, ym, xm, ym, x, y)?;
                            n = 2;
                        }
                        rem_len = f64::INFINITY; // No more arrow for this sub-path
                    } else if n >= self.n {
                        write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                                   {{{:.16}}}{{{:.16}}}}}\n\
                                   \\pgfusepath{{stroke}}\n\
                                   \\pgfpathmoveto{{\\pgfpointxy\
                                   {{{:.16}}}{{{:.16}}}}}\n", x, y, x, y)?;
                        n = 0;
                    } else {
                        write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                                   {{{:.16}}}{{{:.16}}}}}\n", x, y)?
                    }
                    rem_len -= l;
                } else {
                    // No previous point.  New sub-path.
                    write!(f, "\\pgfpathmoveto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y)?
                }
                prev_pt = p;
            }
            None => {
                write!(f, "\\pgfusepath{{stroke}}\n")?;
                rem_len = *lens.next().unwrap();
                prev_pt = None;
            }
        }}
        Ok(())
    }

    /// Write the sampling to the formatter as PGF/TikZ commands.
    pub fn write(&self, f: &mut impl Write) -> Result<(), io::Error> {
        write!(f, "% Written by the Rust curve_sampling crate.\n")?;
        write!(f, "\\begin{{pgfscope}}\n")?;
        match self.color {
            Some(RGB8 {r, g, b}) =>
                write!(f, "\\definecolor{{RustCurveSamplingColor}}{{RGB}}\
                           {{{},{},{}}}\n\
                           \\pgfsetstrokecolor{{RustCurveSamplingColor}}\n",
                       r, g, b)?,
            None => (),
        }
        match (self.arrow, self.arrow_pos) {
            (None, None) => self.write_with_lines(f)?,
            (Some(arrow), None) =>
                self.write_with_arrows(f, arrow, 0.5)?,
            (None, Some(arrow_pos)) =>
                self.write_with_arrows(f, ">",arrow_pos)?,
            (Some(arrow), Some(arrow_pos)) =>
                self.write_with_arrows(f, arrow, arrow_pos)?,
        }
        write!(f, "\\pgfusepath{{stroke}}\n\\end{{pgfscope}}\n")
    }
}

/// # Output
impl Sampling {
    /// Write the sampling `self` using PGF/TikZ commands.
    pub fn latex(&self) -> LaTeX<'_> { LaTeX::new(self) }

    /// Write the sampling to `f` in a tabular form: each point is
    /// written as "x y" on a single line (in scientific notation).
    /// If the path is interrupted, a blank line is printed.  This
    /// format is compatible with Gnuplot.
    pub fn write(&self, f: &mut impl Write) -> Result<(), io::Error> {
        for p in self.iter() {
            match p {
                Some([x, y]) => write!(f, "{:e} {:e}\n", x, y)?,
                None => write!(f, "\n")?,
            }
        }
        Ok(())
    }
}

impl Display for Sampling {
    /// Display the sampling in a tabular form: each point is written
    /// as "x y" on a single line (in scientific notation).  If the
    /// path is interrupted, a blank line is printed.  This format is
    /// compatible with Gnuplot.
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        for p in self.iter() {
            match p {
                Some([x, y]) => write!(f, "{:e} {:e}\n", x, y)?,
                None => write!(f, "\n")?,
            }
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////
//
// Tests

#[cfg(test)]
mod tests {
    use crate::{Sampling, BoundingBox};

    fn xy_of_sampling(s: &Sampling) -> Vec<Option<(f64, f64)>> {
        s.iter().map(|p| p.map(|p| (p[0], p[1]))).collect()
    }

    fn test_clip(bb: BoundingBox,
                 path: Vec<[f64;2]>,
                 expected: Vec<Option<(f64,f64)>>) {
        let s = Sampling::from(path).clip(bb);
        assert_eq!(xy_of_sampling(&s), expected);
    }

    #[test]
    fn clip_base () {
        let bb = BoundingBox { xmin: 0.,  xmax: 3., ymin: 0.,  ymax: 2.};
        test_clip(bb, vec![[0.,1.], [2.,3.]],
                  vec![Some((0.,1.)), Some((1.,2.))]);
        test_clip(bb,
                  vec![[-1.,0.], [2.,3.], [4.,1.]],
                  vec![Some((0.,1.)), Some((1.,2.)), None, Some((3., 2.))]);
        test_clip(bb,
                  vec![[0.,1.], [2.,3.], [4.,1.], [2.,1.], [2., -1.],
                       [0., -1.], [0., 1.] ],
                  vec![Some((0.,1.)), Some((1.,2.)), None,
                       Some((3., 2.)), None,
                       Some((3., 1.)), Some((2., 1.)), Some((2., 0.)), None,
                       Some((0., 0.)), Some((0., 1.))]);
    }

    #[test]
    fn clip_double_cut() {
        let bb = BoundingBox { xmin: 0.,  xmax: 4., ymin: 0.,  ymax: 2.};
        test_clip(bb,
                  vec![[1., 2.], [2.,3.], [5.,0.], [-1., 0.] ],
                  vec![Some((1., 2.)), None,
                       Some((3., 2.)), Some((4., 1.)), None,
                       Some((4., 0.)), Some((0., 0.)) ]);
    }

    #[test]
    fn clip_almost_horiz() {
        let bb = BoundingBox { xmin: 0.,  xmax: 2., ymin: -1.,  ymax: 1.};
        test_clip(bb,
                  vec![[1., 1e-100], [3., -1e-100] ],
                  vec![Some((1., 1e-100)), Some((2., 0.))]);
    }

    #[test]
    fn uniform1() {
        let s = Sampling::uniform(|x| x, 0., 4.).n(3)
            .init(&[1.]).init_pt(&[(3., 0.)]).build();
        assert_eq!(xy_of_sampling(&s),
                   vec![Some((0.,0.)), Some((1.,1.)), Some((2.,2.)),
                        Some((3., 0.)), Some((4.,4.))]);
    }

    #[test]
    fn uniform2() {
        let s = Sampling::uniform(|x| (4. - x).sqrt(), 0., 6.).n(4).build();
        assert_eq!(xy_of_sampling(&s),
                   vec![Some((0.,2.)), Some((2., 2f64.sqrt())),
                        Some((4., 0.)), None]);
    }
}
