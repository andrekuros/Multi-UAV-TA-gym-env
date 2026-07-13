use pyo3::prelude::*;
use std::f64::consts::PI;

use crate::components::{UAV, Task, Obstacle, Threat};

#[pyclass]
pub struct SimCore {
    #[pyo3(get, set)]
    pub time_steps: i32,
    #[pyo3(get, set)]
    pub max_time_steps: i32,
}

#[pymethods]
impl SimCore {
    #[new]
    pub fn new(max_time_steps: i32) -> Self {
        SimCore {
            time_steps: 0,
            max_time_steps,
        }
    }

    /// Computes the avoid_vector for a single agent against a list of obstacles.
    #[staticmethod]
    pub fn avoid_obstacles(agent_pos: [f64; 2], obstacles: Vec<[f64; 3]>, movement: [f64; 2]) -> [f64; 2] {
        let mut avoid_vector = [0.0, 0.0];

        for obs in obstacles.iter() {
            let dx = obs[0] - agent_pos[0];
            let dy = obs[1] - agent_pos[1];
            let distance_to_obstacle = (dx * dx + dy * dy).sqrt();
            let distance_to_zone = distance_to_obstacle - obs[2];

            if distance_to_zone < 40.0 {
                let dir_norm = [dx / distance_to_zone, dy / distance_to_zone];
                
                let mut avoid_force = 1.05_f64.max(distance_to_zone).ln();
                avoid_force = 0.5 / (1.0 - avoid_force);
                
                let angle_mov = movement[1].atan2(movement[0]);
                let angle_obs = dy.atan2(dx);
                
                let mut angle_between = angle_mov - angle_obs;
                angle_between = (angle_between + PI) % (2.0 * PI) - PI;

                let rotated = if angle_between > 0.0 {
                    [dir_norm[1], -dir_norm[0]] // Clockwise
                } else {
                    [-dir_norm[1], dir_norm[0]] // Counter-clockwise
                };

                avoid_vector[0] += rotated[0] * avoid_force;
                avoid_vector[1] += rotated[1] * avoid_force;
            }
        }

        avoid_vector
    }
}
