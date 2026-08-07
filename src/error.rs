use thiserror::Error;

#[derive(Error, Debug)]
pub enum SaxsError {
    #[error("failed to parse structure file: {0}")]
    StructureParse(String),

    #[error("failed to read experimental data file: {0}")]
    ExperimentalDataParse(String),

    #[error("unsupported element: {0}")]
    UnsupportedElement(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("fitting did not converge: {0}")]
    FitDidNotConverge(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, SaxsError>;
