use serde::{de::DeserializeOwned, Serialize};
use std::{collections::BTreeMap, marker::PhantomData};
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("missing key: {0}")]
    Missing(String),
    #[error("serialization: {0}")]
    Serialization(String),
}
#[derive(Debug, Default)]
pub struct DataStore<T> {
    values: BTreeMap<String, Vec<u8>>,
    marker: PhantomData<T>,
}
impl<T> DataStore<T>
where
    T: Serialize + DeserializeOwned,
{
    pub fn put(&mut self, key: impl Into<String>, value: &T) -> Result<(), DataError> {
        self.values.insert(
            key.into(),
            serde_json::to_vec(value).map_err(|e| DataError::Serialization(e.to_string()))?,
        );
        Ok(())
    }
    pub fn get(&self, key: &str) -> Result<T, DataError> {
        let bytes = self
            .values
            .get(key)
            .ok_or_else(|| DataError::Missing(key.into()))?;
        serde_json::from_slice(bytes).map_err(|e| DataError::Serialization(e.to_string()))
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
}
