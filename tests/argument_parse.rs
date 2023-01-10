use anyhow::Result;
use assert_cmd::Command;
use once_cell::sync::Lazy;
use tempfile::{Builder, TempDir};

static TEST_DIRECTORY: Lazy<TempDir> = Lazy::new(|| {
    let directory = TempDir::new();

    directory.unwrap()
});

#[test]
fn test_basic() {}
