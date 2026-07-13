use pyo3::prelude::*;

pub mod components;
pub mod sim_core;

use components::{Task, UAV, Threat, Obstacle};
use sim_core::SimCore;

/// A Python module implemented in Rust.
#[pymodule]
fn core_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Task>()?;
    m.add_class::<UAV>()?;
    m.add_class::<Threat>()?;
    m.add_class::<Obstacle>()?;
    m.add_class::<SimCore>()?;
    Ok(())
}
