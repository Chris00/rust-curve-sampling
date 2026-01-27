//! Doubly linked list (to store paths).  (Unsafe) Witnesses to nodes
//! of the list to be able to perform insertion are offered.  This is
//! necessary in order to evolve the path according to the priority
//! queue.

use std::{marker::PhantomData,
          ptr::NonNull};

pub struct List<T> {
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>, // None ⇔ head = None
    marker: PhantomData<Box<Node<T>>>,
}

struct Node<T> {
    item: T,
    prev: Option<NonNull<Node<T>>>,
    next: Option<NonNull<Node<T>>>,
}

impl<T> Drop for List<T> {
    fn drop(&mut self) {
        let mut node = self.head;
        while let Some(n) = node {
            let n = unsafe { Box::from_raw(n.as_ptr()) };
            node = n.next;
        }
    }
}

impl<T: Clone> Clone for List<T> {
    // Now, `clip` may eat the original sampling ??
    fn clone(&self) -> Self {
        // Duplicate all nodes (including their data) of the list.
        let mut l = Self::new();
        for x in self.iter() {
            l.push_back(x.clone());
        }
        l
    }
}

/// Pointer to a node.
#[derive(Debug)]
pub struct Witness<T> {
    node: NonNull<Node<T>>
}

impl<T> List<T> {
    /// Return a new empty list.
    pub fn new() -> List<T> {
        List { head: None,  tail: None,  marker: PhantomData }
    }

    /// Return `true` iff the list is empty.
    pub fn is_empty(&self) -> bool { self.head.is_none() }

    /// Return the first item in the list, if not empty, otherwise
    /// return `None`.
    pub fn first(&self) -> Option<&T> {
        self.head.map(|node| unsafe { &node.as_ref().item })
    }

    /// Return the first item in the list, if not empty, otherwise
    /// return `None`.
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.head.map(|mut node| unsafe { &mut node.as_mut().item })
    }

    /// Return the last item in the list, if not empty, otherwise
    /// return `None`.
    pub fn last(&self) -> Option<&T> {
        self.tail.map(|node| unsafe { &node.as_ref().item })
    }

    /// Return the last item in the list, if not empty, otherwise
    /// return `None`.
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.tail.map(|mut node| unsafe { &mut node.as_mut().item })
    }

    /// Push a new item to the back of the list.  Return a witness to
    /// the list item that allows to modify it.
    ///
    /// # Safety
    /// The witness is valid only while the list exists.  This is not
    /// enforced by the type system.
    pub fn push_back(&mut self, item: T) -> Witness<T> {
        let node = Node { item, prev: self.tail, next: None };
        let node = unsafe { NonNull::new_unchecked(
            Box::into_raw(Box::new(node))) };
        match self.tail {
            None => self.head = Some(node), // Was an empty list
            Some(mut node0) => unsafe {
                node0.as_mut().next = Some(node) },
        }
        self.tail = Some(node);
        Witness { node }
    }

    /// Replace the item in the list pointed to by `w`.
    ///
    /// # Safety
    /// The node pointed by `w` must be in the list `self`.
    pub unsafe fn replace(&mut self, w: &mut Witness<T>, item: T) {
        w.node.as_mut().item = item
    }

    /// Return a reference to the value pointed by `w`.
    ///
    /// # Safety
    /// The witness must point to an element still in the list.  No
    /// mutable references (created through other witnesses) to the
    /// same item can exist.
    pub unsafe fn get(&self, w: &Witness<T>) -> &T {
        // Because of the way witnesses are implemented, `self` is
        // technically not required here.  The idea is that witnesses
        // act like indices to the list, so alone they do not have the
        // power to mutate it.
        &w.node.as_ref().item
    }

    /// Return a mutable reference to the value.
    ///
    /// # Safety
    /// The witness must point to an element still in the list.  No
    /// mutable references (created through other witnesses) to the
    /// same item can exist.
    pub unsafe fn get_mut(&mut self, w: &Witness<T>) -> &mut T {
        // The witness is simply seen as an index/pointer into the
        // list.  It is not `w` but the list `self` that is being
        // mutated.
        let node = w.node.as_ptr();
        &mut (*node).item
    }

    /// Return a witness to the item right after `self`, if any.
    ///
    /// # Safety
    /// The witness must point to an element still in the list.
    pub unsafe fn next(&self, w: &Witness<T>) -> Option<Witness<T>> {
        w.node.as_ref().next.map(|node| Witness { node })
    }

    /// Return a witness to the item right before `self`, if any.
    ///
    /// # Safety
    /// The witness must point to an element still in the list.
    pub unsafe fn prev(&self, w: &Witness<T>) -> Option<Witness<T>> {
        w.node.as_ref().prev.map(|node| Witness { node })
    }

    /// Insert `item` after the position in `self` pointed to by `w`.
    /// Return a witness to the new list entry.
    ///
    /// # Safety
    /// The item pointed by `w` should still be in the list `self`.
    /// Also, since we modify nearby nodes, no references should be
    /// active for the current or nearby points.
    pub unsafe fn insert_after(&mut self, w: &mut Witness<T>, item: T)
                               -> Witness<T> {
        // We use a mutable reference for `w` to have exclusive access
        // as it disallows all references obtained through `Witness`
        // methods.  This lower the potential for misuse.
        let new = Node {
            item,
            prev: Some(w.node),
            next: w.node.as_ref().next };
        let new = NonNull::new_unchecked(
            Box::into_raw(Box::new(new)));
        match w.node.as_ref().next {
            None => self.tail = Some(new), // last node
            Some(mut next) => next.as_mut().prev = Some(new),
        }
        w.node.as_mut().next = Some(new);
        Witness { node: new }
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter { head: self.head,
               //tail: self.tail,
               marker: PhantomData }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut { head: self.head,
                  //tail: self.tail,
                  marker: PhantomData }
    }

    pub fn into_iter(mut self) -> IntoIter<T> {
        let iter = IntoIter {
            head: self.head,
            marker: PhantomData
        };
        // Prevent the nodes of `self` to be freed at the end of this
        // block — the iterator is now in charge of them.
        self.head = None;
        self.tail = None;
        iter
    }

    /// Iterate on consecutive points (with a peek at the following
    /// segment so that a cost can be computed).
    ///
    /// # Safety
    ///
    /// The references handled by this iterator can only be used
    /// before a call to `next` is made.
    pub unsafe fn iter_segments_mut(&mut self) -> IterSegmentsMut<'_, T> {
        IterSegmentsMut::new(self)
    }
}

/// An iterator over the elements of the list.
pub struct Iter<'a, T: 'a> {
    head: Option<NonNull<Node<T>>>, // None if at end
    //tail: Option<NonNull<Node<T>>>,
    marker: PhantomData<&'a Node<T>>,
}

/// A mutable iterator over the elements of the list.
pub struct IterMut<'a, T: 'a> {
    head: Option<NonNull<Node<T>>>, // None if at end
    //tail: Option<NonNull<Node<T>>>,
    marker: PhantomData<&'a mut Node<T>>,
}

pub struct IterSegmentsMut<'a, T: 'a> {
    node0: Option<NonNull<Node<T>>>, // Start point of the segment
    node1: Option<NonNull<Node<T>>>, // Endpoint of the segment
    _marker: PhantomData<&'a mut T>,
}

/// An iterator over the elements of the list.
pub struct IntoIter<T> {
    head: Option<NonNull<Node<T>>>, // None if at end
    marker: PhantomData<Node<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.head.map(|node| unsafe {
            // Need an unbound lifetime to get 'a
            let node = &*node.as_ptr();
            self.head = node.next;
            &node.item
        })
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        self.head.map(|node| unsafe {
            // Need an unbound lifetime to get 'a
            let node = &mut *node.as_ptr();
            self.head = node.next;
            &mut node.item
        })
    }
}

impl<'a, T> IterSegmentsMut<'a, T> {
    fn new(list: &mut List<T>) -> Self {
        let node0 = list.head;
        let node1 = node0.and_then(|node0| {
            unsafe { node0.as_ref().next }
        });
        Self { node0, node1, _marker: PhantomData }
    }
}

impl<'a, T> Iterator for IterSegmentsMut<'a, T> {
    // SAFETY: Since the mutable value overlap between calls, this
    // allows to create multiple mutable references to a given element
    // by repeatedly using `next`.  Moreover it also allows to create
    // an exclusive mutable reference and a shared reference to the
    // same value which is forbidden.
    type Item = (&'a mut T, Witness<T>, &'a mut T, Option<&'a T>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.node0.and_then(|mut node0| {
            unsafe { node0.as_ref() }.next.map(|mut node1| {
                let item0 = unsafe { &mut node0.as_mut().item };
                let witness0 = Witness { node: node0 };
                let item1 = unsafe { &mut node1.as_mut().item };
                let node2 = unsafe { node1.as_ref().next };
                let item2 = node2.map(|n| unsafe { &n.as_ref().item });
                self.node0 = self.node1;
                self.node1 = node2;
                (item0, witness0, item1, item2)
            })
        })
    }
}

/// If the iterator is not completely consumed, dropping it must free
/// the remaining memory of the list.
impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        let mut node = self.head;
        while let Some(n) = node {
            let n = unsafe { Box::from_raw(n.as_ptr()) };
            node = n.next;
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.head.map(|node| {
            let node = unsafe { Box::from_raw(node.as_ptr()) };
            self.head = node.next;
            node.item
        })
    }
}

impl<T> Witness<T> {
    /// Return a copy of the witness.  We careful that several
    /// witnesses for a given item increase the risk of having several
    /// mutable references to the same item.  This is why the [`Clone`]
    /// trait is *not* implemented (so never automatically derive'ed).
    pub fn clone(&self) -> Self {
        Self { node: self.node }
    }
}

impl<T> std::convert::From<Vec<T>> for List<T> {
    fn from(value: Vec<T>) -> Self {
        let mut l = List::new();
        for x in value {
            l.push_back(x);
        }
        l
    }
}


#[cfg(test)]
mod test {
    use super::List;

    #[test]
    fn basic() {
        let mut l = List::new();
        assert!(l.is_empty());
        l.push_back("a");
        l.push_back("b");
        l.push_back("c");
        assert_eq!(l.first(), Some(&"a"));
        assert_eq!(l.last(), Some(&"c"));
    }

    #[test]
    fn increase_priority() {
        let mut l = List::new();
        l.push_back("a");
        let w = l.push_back("b");
        l.push_back("c");
        unsafe { *l.get_mut(&w) = "d" }
        let v: Vec<_> = l.iter().collect();
        assert_eq!(v, [&"a", &"d", &"c"])
    }

    #[test]
    fn iter() {
        let mut l = List::new();
        l.push_back("a");
        l.push_back("b");
        l.push_back("c");
        let v: Vec<_> = l.iter().collect();
        assert_eq!(v, vec![&"a", &"b", &"c"]);
    }

    #[test]
    fn iter_mut() {
        let mut l = List::new();
        l.push_back("a");
        l.push_back("b");
        l.push_back("c");
        let v: Vec<_> = l.iter_mut().collect();
        assert_eq!(v, vec![&"a", &"b", &"c"]);
    }

    #[test]
    fn into_iter() {
        let mut l = List::new();
        l.push_back("a".to_string());
        l.push_back("b".to_string());
        l.push_back("c".to_string());
        let v: Vec<String> = l.into_iter().collect();
        assert_eq!(v, vec!["a", "b", "c"]);
    }

    #[test]
    fn into_iter_not_consumed() {
        let mut l = List::new();
        l.push_back("a".to_string());
        l.push_back("b".to_string());
        l.push_back("c".to_string());
        let mut i = l.into_iter();
        assert_eq!(i.next(), Some("a".to_string()));
        // Do not consume all of `i` so that Miri can check whether
        // the nodes are indeed freed by dropping `i`.
    }

    #[test]
    fn iter_mut_modif() {
        let mut l = List::new();
        l.push_back("a".to_string());
        l.push_back("b".to_string());
        l.push_back("c".to_string());
        for p in l.iter_mut() {
            p.push('d')
        }
        let v: Vec<_> = l.iter_mut().collect();
        assert_eq!(v, vec![&"ad".to_string(), &"bd".to_string(),
                           &"cd".to_string()]);
    }

    #[test]
    fn insert_after() {
        let mut l = List::new();
        let mut w = l.push_back("a");
        l.push_back("b");
        assert_eq!(unsafe { l.get(&w) }, &"a");
        unsafe {
            *l.get_mut(&w) = "c";
            l.insert_after(&mut w, "d"); }
        let v: Vec<_> = l.iter_mut().collect();
        assert_eq!(v, vec![&"c", &"d", &"b"]);
    }

    #[test]
    fn witness_next() {
        let mut l = List::new();
        let w = l.push_back("a");
        l.push_back("b");
        assert!(unsafe { l.prev(&w) }.is_none());
        match unsafe { l.next(&w) } {
            None => panic!("There is an element after 'a'!"),
            Some(w1) => {
                assert_eq!(unsafe { l.get(&w1) }, &"b");
                assert!(unsafe { l.next(&w1) }.is_none());
            }
        }
    }

    #[test]
    fn insert_after_witness() {
        let mut l = List::new();
        let mut w = l.push_back("a");
        l.push_back("b");
        let mut w1 = unsafe { l.insert_after(&mut w, "d") };
        unsafe { l.insert_after(&mut w1, "e"); }
        let v: Vec<_> = l.iter_mut().collect();
        assert_eq!(v, vec![&"a", &"d", &"e", &"b"]);
    }

    #[test]
    fn replace() {
        let mut l = List::new();
        l.push_back("a");
        let mut w = l.push_back("b");
        l.push_back("c");
        unsafe { l.replace(&mut w, "d"); }
        let v: Vec<_> = l.iter().collect();
        assert_eq!(v, vec![&"a", &"d", &"c"])
    }

}
