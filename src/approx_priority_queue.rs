//! A fast max (approximative) priority queue.
//!

use std::{
    cell::Cell,
    marker::PhantomData,
    rc::Rc,
};

#[derive(Copy, Clone, Debug)]
struct NodeLoc {
    // Index in the array `PQ.p`
    slot: usize,
    // Index in the vector at that `slot`.
    i: usize,
}

#[derive(Debug)]
pub struct Witness<T> {
    // Pointer to the location information
    loc: Rc<Cell<NodeLoc>>,
    marker: PhantomData<T>,
}

struct Node<T> {
    item: T,
    loc: Rc<Cell<NodeLoc>>,
}

/// Priority queue holding elements of type T.
pub struct PQ<T> {
    // Step size.
    dp: f64,
    p: Vec<Vec<Node<T>>>,
    // The first nonempty slot (or 0 if the queue is empty).
    max_slot: usize,
}

// For these constants, see [`lib::cost::segment`].
const PMAX: f64 = 1.;
const N: usize = 256;

impl<T> PQ<T> {
    /// Return an empty priority queue geared to handle `n` levels of
    /// priorities in the range `priority_min .. priority_max`.
    #[inline]
    pub fn new() -> Self {
        let dp = PMAX / N as f64;
        let mut p = Vec::with_capacity(N + 2);
        for _ in 0 .. N + 2 { p.push(vec![]) }
        PQ { dp, p, max_slot: 0 }
    }

    #[cfg(test)]
    pub fn max_priority(&self) -> f64 {
        if self.p[self.max_slot].is_empty() { f64::NAN }
        else { self.max_slot as f64 }
    }

    fn slot_of_priority(&self, priority: f64) -> usize {
        if priority < 0. { 0 }
        else if priority >= PMAX { N + 1 }
        else {
            // `sqrt` to have a higher resolution for small priorities
            (priority.sqrt() / self.dp).ceil() as usize
        }
    }

    /// Pushes `item` with `priority` to the queue and set a reference
    /// to the internal of the PQ in order to be able to increase the
    /// priority.
    ///
    /// # Safety
    /// The [`Witness`] is valid as long as the item stays in PQ.
    /// This is not enforced by the type system.
    pub fn push(&mut self, priority: f64, item: T) -> Witness<T> {
        debug_assert!(!priority.is_nan());
        let slot= self.slot_of_priority(priority);
        let p_slot = &mut self.p[slot];
        let i = p_slot.len();
        // Add a location (at a stable memory location).
        let loc = Rc::new(Cell::new(NodeLoc { slot, i }));
        p_slot.push(Node { item, loc: loc.clone() });
        self.max_slot = self.max_slot.max(slot);
        Witness { loc, marker: PhantomData }
    }

    pub fn pop(&mut self) -> Option<T> {
        let item = self.p[self.max_slot].pop();
        while self.max_slot > 0 && self.p[self.max_slot].is_empty() {
            self.max_slot -= 1;
        }
        match item { None => None,
                     Some(n) =>  Some(n.item) }
        // item
    }

    pub unsafe fn increase_priority(&mut self, w: &Witness<T>, priority: f64) {
        let w_loc = w.loc.get();
        let slot = w_loc.slot;
        let i = w_loc.i;
        let new_slot = self.slot_of_priority(priority);
        // FIXME: Here we could also easily lower the priority.  Do we
        // want to do that?
        if new_slot <= slot { return }
        let p_slot = &mut self.p[slot];
        let n;
        if i + 1 == p_slot.len() {
            n = p_slot.pop().unwrap();
        } else {
            // Exchange with the last one and update the latter location.
            n = p_slot.swap_remove(i);
            let loc = p_slot[i].loc.get();
            p_slot[i].loc.set(NodeLoc { i, .. loc });
        }
        let new_i = self.p[new_slot].len();
        n.loc.set(NodeLoc { slot: new_slot, i: new_i });
        self.p[new_slot].push(n);
        self.max_slot = self.max_slot.max(new_slot);
    }

}


#[cfg(test)]
mod test {
    use super::{PQ, PMAX};

    #[test]
    fn pair() {
        let mut q = PQ::new();
        q.push(0.1 * PMAX, "a");
        q.push(0.2 * PMAX, "b");
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn basic() {
        let mut q = PQ::new();
        q.push(0.2 * PMAX, "a");
        q.push(0.3 * PMAX, "b");
        q.push(0.1 * PMAX, "c");
        q.push(0.0 * PMAX, "d");
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("d"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn increase_priority_simple() {
        let mut q = PQ::new();
        q.push(0.1 * PMAX, "a");
        q.push(0.2 * PMAX, "b");
        let w = q.push(0., "c");
        unsafe { q.increase_priority(&w, 0.3 * PMAX); }
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn increase_priority() {
        let mut q = PQ::new();
        q.push(0.4 * PMAX, "d");
        q.push(0.1 * PMAX, "a");
        q.push(0.3 * PMAX, "c");
        q.push(0.2 * PMAX, "b");
        let w = q.push(0., "e");
        assert_eq!(q.pop(), Some("d"));
        unsafe { q.increase_priority(&w, 0.25 * PMAX); }
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("e"));
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn spike_priority() {
        let n0 = 8;
        let mut q = PQ::new();
        let n1 = n0 / 2;
        for i in 0 .. n1 {
            q.push(i as f64 / n1 as f64 * PMAX, "a");
        }
        for i in n1 .. n0 {
            q.push((n0 - i) as f64 / n1 as f64 * PMAX,"b");
        }
        let mut n = 0;
        while q.pop().is_some() { n += 1 }
        assert_eq!(n, n0)
    }

    #[test]
    fn random_priority() {
        let n0 = 100;
        let mut q = PQ::new();
        // Pseudo-random priorities
        let mut random = 1000;
        const NORMALIZE_01: f64 = 1. / u32::MAX as f64;
        let mut rand = move || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random as f64 * NORMALIZE_01 * PMAX
        };
        for _ in 0 .. n0 {
            q.push(rand(), "a");
        }
        let mut n = 0;
        while q.pop().is_some() { n += 1 }
        assert_eq!(n, n0)
    }
}
