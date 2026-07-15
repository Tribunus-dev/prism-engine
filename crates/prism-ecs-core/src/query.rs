use crate::column::Column;
use crate::component::Component;
use crate::entity::Entity;

/// Dense iterator over entities that have component type A.
/// Backed by the Column<T> SparseSet storage — iterates the dense array
/// directly without HashMap overhead.
#[derive(Debug)]
pub struct Query<'w, A: Component> {
    pub(crate) col: Option<&'w Column<A>>,
    pub(crate) cursor: usize,
}

impl<'w, A: Component> Iterator for Query<'w, A> {
    type Item = (Entity, &'w A);
    fn next(&mut self) -> Option<Self::Item> {
        let col = self.col?;
        if self.cursor >= col.len() {
            return None;
        }
        let idx = self.cursor;
        self.cursor += 1;
        let e = col.entities()[idx];
        Some((e, &col.dense()[idx]))
    }
}

/// Dense mutable iterator over entities with component type A.
#[derive(Debug)]
pub struct QueryMut<'w, A: Component> {
    pub(crate) col: Option<&'w mut Column<A>>,
    pub(crate) cursor: usize,
}

impl<'w, A: Component> Iterator for QueryMut<'w, A> {
    type Item = (Entity, &'w mut A);
    fn next(&mut self) -> Option<Self::Item> {
        let col = self.col.as_mut()?;
        if self.cursor >= col.len() {
            return None;
        }
        let idx = self.cursor;
        self.cursor += 1;
        let e = col.entities()[idx];
        // SAFETY: each element yielded at most once due to cursor advancement
        let ptr = col.dense_mut().as_mut_ptr();
        Some((e, unsafe { &mut *ptr.add(idx) }))
    }
}

/// Multi-component query: iterates entities that have BOTH component A and B.
#[derive(Debug)]
pub struct Query2<'w, A: Component, B: Component> {
    pub(crate) col_a: Option<&'w Column<A>>,
    pub(crate) col_b: Option<&'w Column<B>>,
    pub(crate) cursor: usize,
}

impl<'w, A: Component, B: Component> Iterator for Query2<'w, A, B> {
    type Item = (Entity, &'w A, &'w B);
    fn next(&mut self) -> Option<Self::Item> {
        let col_a = self.col_a?;
        let col_b = self.col_b?;
        while self.cursor < col_a.len() {
            let idx = self.cursor;
            self.cursor += 1;
            let e = col_a.entities()[idx];
            if col_b.has(e) {
                return Some((e, &col_a.dense()[idx], col_b.get(e).unwrap()));
            }
        }
        None
    }
}

/// Multi-component query: iterates entities that have ALL of components A, B, C.
#[derive(Debug)]
pub struct Query3<'w, A: Component, B: Component, C: Component> {
    pub(crate) col_a: Option<&'w Column<A>>,
    pub(crate) col_b: Option<&'w Column<B>>,
    pub(crate) col_c: Option<&'w Column<C>>,
    pub(crate) cursor: usize,
}

impl<'w, A: Component, B: Component, C: Component> Iterator for Query3<'w, A, B, C> {
    type Item = (Entity, &'w A, &'w B, &'w C);
    fn next(&mut self) -> Option<Self::Item> {
        let col_a = self.col_a?;
        let col_b = self.col_b?;
        let col_c = self.col_c?;
        while self.cursor < col_a.len() {
            let idx = self.cursor;
            self.cursor += 1;
            let e = col_a.entities()[idx];
            if col_b.has(e) && col_c.has(e) {
                return Some((
                    e,
                    &col_a.dense()[idx],
                    col_b.get(e).unwrap(),
                    col_c.get(e).unwrap(),
                ));
            }
        }
        None
    }
}
