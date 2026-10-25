//! A fast max (approximative) priority queue.

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

/// A witness of a queue entry to be able to update its priority.
#[derive(Debug)]
// This value cannot be cloned so only one instance may be created (no
// need to update several witnesses when the priority is changed).
pub struct Witness<T> {
    loc: Rc<Cell<NodeLoc>>,
    marker: PhantomData<T>,
}

struct Node<T> {
    item: T,
    loc: Rc<Cell<NodeLoc>>,
}

/// Priority queue holding elements of type T and priorities in
/// `0. .. 1.`.  A nonlinear deformation of the priorities is
/// performed in order to have a higher scale of small priorities.
pub struct PQ<T> {
    p: Vec<Vec<Node<T>>>,
    // The first nonempty slot (or 0 if the queue is empty).
    max_slot: usize,
}

// Priorities are clamped to `0. ..= PMAX`.  A segment whose cost has
// been refined enough to be tiny must still be ordered correctly with
// respect to its neighbors (costs decrease geometrically when a
// segment is split), so priorities are mapped to slots on a
// *logarithmic* scale: `SLOTS_PER_OCTAVE` consecutive slots for each
// factor 2 of the priority.  A linear scale (even piecewise) puts all
// small priorities in the same slot, in which case the queue
// degenerates to a LIFO and the sampling stops being adaptive when
// many evaluations are requested.  See [`lib::cost::segment`] and the
// `horror` test.
const PMAX: f64 = 1.;
const SLOTS_PER_OCTAVE: f64 = 8.;
/// Number of slots.  Slot 0 is reserved for priorities ≤ 0.  The
/// slots 1 ..= N-1 cover priorities from about `PMAX · 2^-(N-2)/8`
/// (≈ 3.5e-15, smaller ones are put in slot 1) up to `PMAX`.
const N: usize = 384;

impl<T> PQ<T> {
    /// Return an empty priority queue geared to handle `N` levels of
    /// priorities in the range `0. .. 1.`.
    #[inline]
    pub fn new() -> Self {
        let mut p = Vec::with_capacity(N);
        for _ in 0 .. N { p.push(vec![]) }
        PQ { p, max_slot: 0 }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.p.iter().map(|pi| pi.len()).sum()
    }

    #[cfg(test)]
    pub fn max_priority(&self) -> f64 {
        if self.p[self.max_slot].is_empty() { f64::NAN }
        else { self.max_slot as f64 }
    }

    #[cfg(test)]
    pub fn print_stats(&self) {
        for (i, pi) in self.p.iter().enumerate() {
            if !pi.is_empty() {
                println!("{i:3} {} (capacity: {})", pi.len(), pi.capacity());
            }
        }
        println!("Number of elements: {}", self.len());
    }

    fn slot_of_priority(priority: f64) -> usize {
        let priority = priority.clamp(0., PMAX);
        if priority <= 0. { return 0 }
        // priority ∈ ]0, PMAX] ⟹ log2(priority / PMAX) ≤ 0.
        let s = (N - 1) as f64
            + ((priority / PMAX).log2() * SLOTS_PER_OCTAVE).floor();
        // `as usize` saturates (to 0 for negative values).
        (s as usize).clamp(1, N - 1)
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
        let slot= Self::slot_of_priority(priority);
        let p_slot = &mut self.p[slot];
        let i = p_slot.len();
        // Add a location (at a stable memory location).
        let loc = Rc::new(Cell::new(NodeLoc { slot, i }));
        p_slot.push(Node { item, loc: loc.clone() });
        self.max_slot = self.max_slot.max(slot);
        Witness { loc, marker: PhantomData }
    }

    /// Restore the invariant that `self.p[self.max_slot]` is nonempty
    /// (or `max_slot == 0`) after a slot may have been emptied.
    fn shrink_max_slot(&mut self) {
        while self.max_slot > 0 && self.p[self.max_slot].is_empty() {
            self.max_slot -= 1;
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let item = self.p[self.max_slot].pop();
        self.shrink_max_slot();
        item.map(|n| n.item)
    }

    pub fn update_priority(&mut self, w: &Witness<T>, priority: f64) {
        let NodeLoc { slot, i } = w.loc.get();
        let new_slot = Self::slot_of_priority(priority);
        if new_slot == slot { return }
        let p_slot = &mut self.p[slot];
        let n;
        if i + 1 == p_slot.len() {
            n = p_slot.pop().unwrap();
            self.shrink_max_slot();
        } else {
            // Exchange with the last one and update the latter location.
            n = p_slot.swap_remove(i);
            let loc = p_slot[i].loc.get();
            p_slot[i].loc.set(NodeLoc { i, .. loc });
        };
        let new_i = self.p[new_slot].len();
        n.loc.set(NodeLoc { slot: new_slot, i: new_i });
        self.p[new_slot].push(n);
        self.max_slot = self.max_slot.max(new_slot);
    }

}


#[cfg(test)]
mod test {
    use super::{PQ, PMAX, Witness};

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
    fn update_priority_simple() {
        let mut q = PQ::new();
        q.push(0.1 * PMAX, "a");
        q.push(0.2 * PMAX, "b");
        let w = q.push(0., "c");
        q.update_priority(&w, 0.3 * PMAX);
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn update_priority() {
        let mut q = PQ::new();
        q.push(0.4 * PMAX, "d");
        q.push(0.1 * PMAX, "a");
        let c1 = q.push(0.3 * PMAX, "c1");
        q.push(0.3 * PMAX, "c2");
        q.push(0.2 * PMAX, "b");
        let e = q.push(0., "e");
        assert_eq!(q.len(), 6);
        q.update_priority(&e, 0.25 * PMAX);
        assert_eq!(q.pop(), Some("d"));
        q.update_priority(&c1, 0.2 * PMAX);
        assert_eq!(q.len(), 5);
        assert_eq!(q.pop(), Some("c2"));
        assert_eq!(q.pop(), Some("e"));
        assert_eq!(q.pop(), Some("c1"));
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn lower_priority_from_lone_top_slot() {
        // Regression test: `top` is the *only* item in the current
        // max slot.  Lowering its priority must not leave `max_slot`
        // stuck pointing at the now-empty slot, or `pop` will
        // wrongly report the queue as exhausted while `a` (and
        // `top` itself, once relocated) are still in it.
        let mut q = PQ::new();
        q.push(0.2 * PMAX, "a");
        let top = q.push(0.4 * PMAX, "top");
        q.update_priority(&top, 0.1 * PMAX);
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), Some("top"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn small_priorities_are_ordered() {
        // Priorities spanning many orders of magnitude below 1 must be
        // distinguished (they used to share a single slot, and thus
        // came out in LIFO order).
        let mut q = PQ::new();
        q.push(1e-7 * PMAX, "high");
        q.push(1e-9 * PMAX, "low");
        q.push(1e-12 * PMAX, "lower");
        assert_eq!(q.pop(), Some("high"));
        assert_eq!(q.pop(), Some("low"));
        assert_eq!(q.pop(), Some("lower"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn slot_of_priority_monotone() {
        let mut prev = PQ::<()>::slot_of_priority(0.);
        assert_eq!(prev, 0);
        let mut p = 1e-300;
        while p < PMAX {
            let s = PQ::<()>::slot_of_priority(p);
            assert!(s >= prev && s >= 1 && s < super::N, "p = {p:e}");
            prev = s;
            p *= 1.01;
        }
        assert_eq!(PQ::<()>::slot_of_priority(PMAX), super::N - 1);
        assert_eq!(PQ::<()>::slot_of_priority(5e5), super::N - 1);
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

    /// Random sequences of `push`, `update_priority` (raising or
    /// lowering) and `pop`, checked against a trivial model: every
    /// `pop` must return an item of the highest occupied slot, `None`
    /// only when the queue is really empty, and no item may be lost.
    #[test]
    fn random_ops_match_model() {
        // xorshift64 pseudo-random generator (fixed seed).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Mix of exact bounds, tiny and ordinary priorities so that
        // both the "small" and "large" slot ranges are exercised.
        let prio = |r: u64| -> f64 {
            let u = (r >> 8) as f64 / u64::MAX as f64;
            match r % 5 { 0 => 0., 1 => PMAX, 2 => u * 1e-3, _ => u * PMAX }
        };
        for round in 0 .. 300 {
            let mut q: PQ<usize> = PQ::new();
            // Indexed by item id.  `None` once the item has been popped.
            let mut wit: Vec<Option<Witness<usize>>> = vec![];
            let mut slot: Vec<Option<usize>> = vec![]; // model
            let mut live = 0;
            for _ in 0 .. 400 {
                match rnd() % 4 {
                    0 | 1 => {
                        let id = wit.len();
                        let p = prio(rnd());
                        wit.push(Some(q.push(p, id)));
                        slot.push(Some(PQ::<usize>::slot_of_priority(p)));
                        live += 1;
                    }
                    2 if live > 0 => {
                        let ids: Vec<_> = (0 .. slot.len())
                            .filter(|&i| slot[i].is_some()).collect();
                        let id = ids[rnd() as usize % ids.len()];
                        let p = prio(rnd());
                        q.update_priority(wit[id].as_ref().unwrap(), p);
                        slot[id] = Some(PQ::<usize>::slot_of_priority(p));
                    }
                    _ => {
                        let best = slot.iter().flatten().max().copied();
                        match (q.pop(), best) {
                            (None, None) => (),
                            (Some(id), Some(b)) => {
                                assert_eq!(slot[id], Some(b),
                                    "round {round}: popped a non-maximal item");
                                slot[id] = None;
                                wit[id] = None;
                                live -= 1;
                            }
                            (got, want) => panic!(
                                "round {round}: pop() = {got:?} but the \
                                 highest occupied slot is {want:?} \
                                 ({live} items in the queue)"),
                        }
                    }
                }
                assert_eq!(q.len(), live, "round {round}: item count");
            }
            // Draining must return every remaining item.
            while let Some(id) = q.pop() { slot[id] = None; live -= 1 }
            assert_eq!(live, 0, "round {round}: items lost");
        }
    }
}
