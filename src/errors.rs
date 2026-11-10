use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt: {0}")]
    Corrupt(String),
    #[error("invalid: {0}")]
    Invalid(String),
}
pub type Result<T> = std::result::Result<T, Error>;
