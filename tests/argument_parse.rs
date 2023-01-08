use assert_cmd::Command;
use tempfile::{Builder, TempDir};
use once_cell::sync::Lazy;
use anyhow::Result;

static TEST_DIRECTORY: Lazy<TempDir> = Lazy::new(|| {
   let directory = TempDir::new();

   directory.unwrap()
});

#[test]
fn test_basic() {
}