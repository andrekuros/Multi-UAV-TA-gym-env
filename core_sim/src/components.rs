use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct Task {
    #[pyo3(get, set)]
    pub id: i32,
    #[pyo3(get, set)]
    pub position: [f64; 2],
    #[pyo3(get, set)]
    pub task_type: String,
    
    // Status: 0 - waiting Allocation / 1 - Allocated / 2 - Concluded / (-1) - Inactive
    #[pyo3(get, set)]
    pub status: i32,
    #[pyo3(get, set)]
    pub task_window: f64,
    #[pyo3(get, set)]
    pub task_duration: f64,
    
    #[pyo3(get, set)]
    pub init_time: f64,
    #[pyo3(get, set)]
    pub done_time: f64,
}

#[pymethods]
impl Task {
    #[new]
    pub fn new(id: i32, position: [f64; 2], task_type: String, task_window: f64, task_duration: f64) -> Self {
        Task {
            id,
            position,
            task_type,
            status: 0,
            task_window,
            task_duration,
            init_time: -1.0,
            done_time: -1.0,
        }
    }
}

#[pyclass]
pub struct UAV {
    #[pyo3(get, set)]
    pub id: i32,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub position: [f64; 2],
    #[pyo3(get, set)]
    pub altitude: f64,
    
    // 0 - Idle / 1 - Navigating / 2 - In Task / 3 - Retuning to Base / (4) - Out of Service
    #[pyo3(get, set)]
    pub state: i32,
    #[pyo3(get, set)]
    pub uav_type: String,
    #[pyo3(get, set)]
    pub max_speed: f64,
    #[pyo3(get, set)]
    pub engage_range: f64,
    
    #[pyo3(get, set)]
    pub next_free_time: f64,
    #[pyo3(get, set)]
    pub next_free_position: [f64; 2],
}

#[pymethods]
impl UAV {
    #[new]
    pub fn new(id: i32, name: String, position: [f64; 2], uav_type: String, max_speed: f64, engage_range: f64) -> Self {
        UAV {
            id,
            name,
            position,
            altitude: 1000.0,
            state: 0,
            uav_type,
            max_speed,
            engage_range,
            next_free_time: 0.0,
            next_free_position: position,
        }
    }
}

#[pyclass]
pub struct Threat {
    #[pyo3(get, set)]
    pub id: i32,
    #[pyo3(get, set)]
    pub position: [f64; 2],
    #[pyo3(get, set)]
    pub max_speed: f64,
    #[pyo3(get, set)]
    pub engage_range: f64,
    #[pyo3(get, set)]
    pub attack: f64,
    #[pyo3(get, set)]
    pub defence: f64,
    #[pyo3(get, set)]
    pub status: i32,
}

#[pymethods]
impl Threat {
    #[new]
    pub fn new(id: i32, position: [f64; 2], max_speed: f64, engage_range: f64, attack: f64, defence: f64) -> Self {
        Threat {
            id,
            position,
            max_speed,
            engage_range,
            attack,
            defence,
            status: 1,
        }
    }
}

#[pyclass]
pub struct Obstacle {
    #[pyo3(get, set)]
    pub position: [f64; 2],
    #[pyo3(get, set)]
    pub size: f64,
}

#[pymethods]
impl Obstacle {
    #[new]
    pub fn new(position: [f64; 2], size: f64) -> Self {
        Obstacle { position, size }
    }
}
