pub mod config;
pub mod cispo;
pub mod grpo;
pub mod rl_trainer;
pub mod data;
pub mod distillation;
pub mod champion_challenger;

pub use config::TrainingConfig;
pub use cispo::CispoTrainer;
pub use grpo::GrpoTrainer;
pub use rl_trainer::RLTrainer;
pub use distillation::{DistillationSample, DistillationCycle, DistillationScheduler};
pub use champion_challenger::{ArenaModel, ArenaMatch, ArenaStatus, ChampionChallenger};
