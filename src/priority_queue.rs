//! Priority Queue adapted to our case.  Wee need it
//! - to use non-NaN `f64` values as priorities;
//! - to store pointers (it is made generic);
//! - to have "witnesses" that can invalidate these pointers in order
//!   to change the priority of already stored elements (without the
//!   cost of removing the old ones).

//  Maximum priority queue.  Implemented as a Pairing heap
// (http://en.wikipedia.org/wiki/Pairing_heap) following the paper:
//
// Fredman, Michael L.; Sedgewick, Robert; Sleator, Daniel D.; Tarjan,
// Robert E. (1986). "The pairing heap: a new form of self-adjusting
// heap" (PDF). Algorithmica. 1 (1): 111–129. doi:10.1007/BF01840439.

use std::{cmp::Ordering,
          ptr::NonNull};

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

struct Node<T> {
    priority: NotNAN,
    item: T,
    child: NonNull<Node<T>>, // points to the node itself if no child.
    sibling: NonNull<Node<T>>, // next older sibling, parent if last,
                               // itself if root.
    parent: Option<NonNull<Node<T>>>, // None if root.
}


impl<T> Node<T> {
    /// Return a pointer to a newly created node (i.e., a priority
    /// queue with a single node).  The user is responsible for
    /// freeing the returned node.
    fn new(priority: f64, item: T) -> NonNull<Self> {
        assert!(!priority.is_nan());
        let n = Node { priority: NotNAN(priority), item,
                       child: NonNull::dangling(),
                       sibling: NonNull::dangling(),
                       parent: None };
        unsafe {
            let mut n = NonNull::new_unchecked(
                Box::into_raw(Box::new(n)));
            n.as_mut().child = n;
            n.as_mut().sibling = n;
            n }
    }

    #[inline]
    fn has_children(node: NonNull<Self>) -> bool {
        unsafe { node.as_ref().child != node }
    }

    /// Recursively drop all *non-root* nodes from a priority queue.
    /// When passing the pointer to the root as `node`, use null for
    /// `last_sibling`.
    fn drop(mut node: NonNull<Node<T>>, last_sibling: NonNull<Node<T>>) {
        while node != last_sibling {
            if Self::has_children(node) {
                Self::drop(unsafe { node.as_ref().child }, node);
            }
            let n = unsafe { Box::from_raw(node.as_ptr()) };
            node = n.sibling;
        }
    }

    /// Link two priority queues.
    ///
    /// # Invariant
    /// If `n1` or `n2` is not configured as the root of a PQ, the
    /// resulting node may neither configured as the root of a PQ
    /// (i.e., `sibling` and `parent` may not respect the
    /// specification).
    #[inline]
    fn link(mut n1: NonNull<Self>, mut n2: NonNull<Self>) -> NonNull<Self> {
        unsafe {
            if n1.as_ref().priority >= n2.as_ref().priority {
                // Because of the convention that the sibling = parent
                // if last, we do not have to make a special case if
                // n2 is the first child (in which case n1.child == n1).
                n2.as_mut().sibling = n1.as_ref().child;
                n2.as_mut().parent = Some(n1);
                n1.as_mut().child = n2;
                n1
            } else {
                n1.as_mut().sibling = n2.as_ref().child;
                n1.as_mut().parent = Some(n2);
                n2.as_mut().child = n1;
                n2
            }
        }
    }
}

/// Priority queue holding elements of type T.
pub struct PQ<T> {
    root: Option<NonNull<Node<T>>>
}

impl<T> Drop for PQ<T> {
    fn drop(&mut self) {
        if let Some(root) = self.root {
            if Node::has_children(root) {
                Node::drop(unsafe { root.as_ref().child }, root)
            }
            unsafe { Box::from_raw(root.as_ptr()) }; // drop root
        }
    }
}

/// Witness for a node.  It enables increasing the node priority.
#[derive(Debug)]
pub struct Witness<T> {
    node: NonNull<Node<T>>
}

impl<T> PQ<T> {
    /// Create an empty priority queue.
    #[inline]
    pub fn new() -> Self { PQ { root: None } }

    // Return `true` if the priority queue is empty.
    // #[inline]
    // pub fn is_empty(&self) -> bool { self.root.is_none() }

    /// Push a node to a tree.
    ///
    /// # Invariant
    /// If `node` is not configured as the root of a PQ, `self` may
    /// neither configured as the root of a PQ (i.e., `sibling` and
    /// `parent` may not respect the specification).
    #[inline]
    fn push_node(&mut self, node: NonNull<Node<T>>) {
        match self.root {
            None => self.root = Some(node),
            Some(root) =>
                self.root = Some(Node::link(root, node))
        }
    }

    /// Pushes `item` with `priority` to the queue and set a reference
    /// to the internal of the PQ in order to be able to increase the
    /// priority.
    ///
    /// # Safety
    /// That reference only lives as long as the PQ does.  This is not
    /// enforced by the type system.  (The reference is updated when
    /// the node is popped out of the PQ so cannot be misused).
    pub fn push(&mut self, priority: f64, item: T) -> Witness<T> {
        // `Node::new` checks for NaN and returns a node configured as
        // the root of a PQ.
        let node = Node::new(priority, item);
        self.push_node(node); // The node is owned by the priority queue.
        Witness { node }
    }

    /// Link all the children of a deleted node, the first child being
    /// pointed to by `n0` (which thus cannot be the root).
    /// Return a *root* node to the linked tree.
    fn link_pairs(n0: NonNull<Node<T>>, last_sibling: NonNull<Node<T>>)
                  -> NonNull<Node<T>> {
        // Instead of using recursion, we accumulate the pairs in a vector.
        let mut pair = Vec::new();
        let mut n1 = n0;
        while n1 != last_sibling {
            let n2 = unsafe { n1.as_ref().sibling };
            if n2 != last_sibling {
                // Save the initial sibling because the node pointed
                // to may be changed.
                let sibling = unsafe { n2.as_ref().sibling };
                pair.push(Node::link(n1, n2));
                n1 = sibling;
            } else { // n1_ptr is the last sibling
                pair.push(n1);
                break
            }
        }
        // Combine the (non-empty vector of) pairs right-to-left.
        let mut pair = pair.iter().rev();
        let mut combined = *pair.next().unwrap();
        for &n in pair {
            combined = Node::link(n, combined)
        }
        unsafe { // Configure as root node.
            combined.as_mut().parent = None;
            combined.as_mut().sibling = combined;
        }
        combined
    }

    /// Removes the greatest element from the priority queue and
    /// returns it, or `None` if the queue is empty.
    ///
    /// # Safety
    /// If there were witnesses to the popped out node, they can no
    /// longer be used.
    pub fn pop(&mut self) -> Option<T> {
        match self.root {
            None => None,
            Some(root) => {
                if Node::has_children(root) {
                    self.root = Some(Self::link_pairs(
                        unsafe { root.as_ref().child }, root))
                } else {
                    self.root = None
                }
                // Finally, pop up (drop) the root.
                let node = unsafe { Box::from_raw(root.as_ptr()) };
                Some(node.item)
            }
        }
    }

    /// Set the priority of node pointed by `w` to `priority` if it is
    /// higher than the existing priority (otherwise, do nothing).
    ///
    /// # Safety
    /// BEWARE that the node pointed by `w` must be in the queue
    /// `self` otherwise it is undefined behavior.
    //
    // REMARK: There is a risk that the witness is not for the
    // appropriate queue.  Since this is an internal API and there
    // will be a single PQ per sampling, the risk is minimal and we
    // did not add a pointer to the queue in the witness.
    pub unsafe fn increase_priority(&mut self, w: &Witness<T>, priority: f64) {
        let mut node = w.node;
        assert!(!priority.is_nan());
        if priority <= node.as_ref().priority.0 {
            return
        }
        let priority = NotNAN(priority);
        match node.as_ref().parent {
            None => node.as_mut().priority = priority, // root node
            Some(mut parent) => {
                // Cut `node` (and its children) from the tree and
                // re-insert it with the new `priority`.
                if parent.as_ref().child == node {
                    // First (possibly only) child
                    parent.as_mut().child = node.as_ref().sibling;
                } else {
                    let mut n_prev = parent.as_ref().child;
                    while n_prev.as_ref().sibling != node {
                        n_prev = n_prev.as_ref().sibling;
                    }
                    n_prev.as_mut().sibling = node.as_ref().sibling
                }
                // `node` is detached from the PQ, configure as root
                node.as_mut().priority = priority;
                node.as_mut().sibling = node;
                node.as_mut().parent = None;
                self.push_node(node)
            }
        }
    }
}


#[cfg(test)]
mod test {
    use super::PQ;
    use rand::prelude::*;

    #[test]
    fn basic() {
        let mut q = PQ::new();
        q.push(1., "a");
        q.push(2., "b");
        q.push(0., "c");
        q.push(-1., "d");
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("d"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn increase_priority() {
        let mut q = PQ::new();
        q.push(1., "a");
        q.push(2., "b");
        let w = q.push(0., "c");
        unsafe { q.increase_priority(&w, 3.); }
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn spike_priority() {
        let n0 = 100;
        let mut q = PQ::new();
        let n1 = n0 / 2;
        for i in 0 .. n1 { q.push(i as f64, "a"); }
        for i in n1 .. n0 { q.push((n0 - i) as f64, "a"); }
        let mut n = 0;
        while q.pop().is_some() { n += 1 }
        assert_eq!(n, n0)
    }

    #[test]
    fn random_priority() {
        let n0 = 100;
        let mut q = PQ::new();
        let mut rng = rand::thread_rng();
        for _ in 0 .. n0 {
            q.push(rng.gen::<f64>(), "a");
        }
        let mut n = 0;
        while q.pop().is_some() { n += 1 }
        assert_eq!(n, n0)
    }

}
