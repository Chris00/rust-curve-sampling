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

    /// Return the last item in the list, if not empty, otherwise
    /// return `None`.
    pub fn last(&self) -> Option<&T> {
        self.tail.map(|node| unsafe { &node.as_ref().item })
    }

    /// Remove the final point (if any) of the list.
    ///
    /// # Safety
    /// After popping out a node, its witnesses should not be used.
    pub fn pop_back(&mut self) -> Option<T> {
        self.tail.map(|node| unsafe {
            let node = Box::from_raw(node.as_ptr());
            self.tail = node.prev;
            match self.tail {
                None => self.head = None,
                // Not creating new mutable (unique!) references
                // overlapping `element`.
                Some(tail) => (*tail.as_ptr()).next = None,
            }
            node.item
        })
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

    pub fn iter_witness_mut(&mut self) -> IterWitnessMut<'_, T> {
        IterWitnessMut { head: self.head,
                         //tail: self.tail,
                         marker: PhantomData }
    }

}

/// An iterator over the elements of the list.
#[derive(Clone)]
pub struct Iter<'a, T: 'a> {
    head: Option<NonNull<Node<T>>>, // None if at end
    //tail: Option<NonNull<Node<T>>>,
    marker: PhantomData<&'a Node<T>>,
}

/// A mutable iterator over the elements of the list.
#[derive(Clone)]
pub struct IterMut<'a, T: 'a> {
    head: Option<NonNull<Node<T>>>, // None if at end
    //tail: Option<NonNull<Node<T>>>,
    marker: PhantomData<&'a mut Node<T>>,
}

/// A mutable iterator over witnesses to elements of the list.
#[derive(Clone)]
pub struct IterWitnessMut<'a, T: 'a> {
    head: Option<NonNull<Node<T>>>, // None if at end
    //tail: Option<NonNull<Node<T>>>,
    marker: PhantomData<&'a mut Node<T>>,
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

impl<'a, T> Iterator for IterWitnessMut<'a, T> {
    type Item = Witness<T>;

    #[inline]
    fn next(&mut self) -> Option<Witness<T>> {
        self.head.map(|node| unsafe {
            self.head = node.as_ref().next;
            Witness { node }
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

    /// Return a reference to the value.
    ///
    /// # Safety
    /// The witness must point to an element still in the list.  No
    /// mutable references (created through other witnesses) to the
    /// same item can exist.
    pub unsafe fn as_ref(&self) -> &T {
        &self.node.as_ref().item
    }

    /// Return a mutable reference to the value.
    ///
    /// # Safety
    /// The witness must point to an element still in the list.  No
    /// mutable references (created through other witnesses) to the
    /// same item can exist.
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut self.node.as_mut().item
    }

    /// Return a witness to the item right after `self`, if any.
    ///
    /// # Safety
    /// The witness must point to an element still in the list.
    pub unsafe fn next(&self) -> Option<Witness<T>> {
        self.node.as_ref().next.map(|node| Witness { node  })
    }

    /// Return a witness to the item right before `self`, if any.
    ///
    /// # Safety
    /// The witness must point to an element still in the list.
    pub unsafe fn prev(&self) -> Option<Witness<T>> {
        self.node.as_ref().prev.map(|node| Witness { node })
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
        assert_eq!(l.pop_back(), Some("c"));
        assert_eq!(l.pop_back(), Some("b"));
        assert_eq!(l.pop_back(), Some("a"));
        assert_eq!(l.pop_back(), None);
    }

    #[test]
    fn increase_priority() {
        let mut l = List::new();
        l.push_back("a");
        let mut w = l.push_back("b");
        l.push_back("c");
        unsafe { *w.as_mut() = "d" }
        assert_eq!(l.pop_back(), Some("c"));
        assert_eq!(l.pop_back(), Some("d"));
        assert_eq!(l.pop_back(), Some("a"));
        assert_eq!(l.pop_back(), None);
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
    fn iter_with_pop() {
        let mut l = List::new();
        l.push_back("a");
        l.push_back("b");
        l.push_back("c");
        assert_eq!(l.pop_back(), Some("c"));
        let v: Vec<_> = l.iter().collect();
        assert_eq!(v, vec![&"a", &"b"]);
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
        assert_eq!(unsafe { w.as_ref() }, &"a");
        unsafe {
            *w.as_mut() = "c";
            l.insert_after(&mut w, "d"); }
        let v: Vec<_> = l.iter_mut().collect();
        assert_eq!(v, vec![&"c", &"d", &"b"]);
    }

    #[test]
    fn witness_next() {
        let mut l = List::new();
        let w = l.push_back("a");
        l.push_back("b");
        assert!(unsafe { w.prev() }.is_none());
        match unsafe { w.next() } {
            None => panic!("There is an element after 'a'!"),
            Some(w1) => {
                assert_eq!(unsafe { w1.as_ref() }, &"b");
                assert!(unsafe { w1.next() }.is_none());
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
