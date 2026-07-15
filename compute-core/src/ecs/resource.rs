//! Typed resource guards for safe World resource access.
//!
//! `Resource<T>` wraps a singleton value with typed access.
//! `ResourceRef` and `ResourceMut` provide guarded references.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// A typed resource wrapper for World-scoped singleton data.
///
/// Provides safe, type-erased storage with Deref/DerefMut access.
#[derive(Debug)]
pub struct Resource<T: 'static + Send + Sync> {
    value: T,
}

impl<T: 'static + Send + Sync> Resource<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }

    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T: 'static + Send + Sync> Deref for Resource<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T: 'static + Send + Sync> DerefMut for Resource<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

/// Immutable borrow guard for a World resource.
#[derive(Debug)]
pub struct ResourceRef<'w, T: 'static + Send + Sync> {
    value: &'w T,
    _phantom: PhantomData<&'w T>,
}

impl<'w, T: 'static + Send + Sync> ResourceRef<'w, T> {
    pub fn new(value: &'w T) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<'w, T: 'static + Send + Sync> Deref for ResourceRef<'w, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

/// Mutable borrow guard for a World resource.
#[derive(Debug)]
pub struct ResourceMut<'w, T: 'static + Send + Sync> {
    value: &'w mut T,
    _phantom: PhantomData<&'w mut T>,
}

impl<'w, T: 'static + Send + Sync> ResourceMut<'w, T> {
    pub fn new(value: &'w mut T) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<'w, T: 'static + Send + Sync> Deref for ResourceMut<'w, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'w, T: 'static + Send + Sync> DerefMut for ResourceMut<'w, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}
