use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;

use crate::adaptive::DynamicSOTAScheduler;
use crate::audit::ComplianceReporter;
use crate::configs::loader::{load_config, ZilliConfig};
use crate::data::{TrajectoryCleaner, TrajectoryStore};
use crate::envs::cost_controller::{CostController, PlannerState};
use crate::envs::mock_env::HermesSandbox;
use crate::evolution::SkillEvolutionEngine;
use crate::hybrid::executor::HybridExecutor;
use crate::hybrid::gatekeeper::PrivacyGatekeeper;
use crate::industry::WorkflowRegistry;
use crate::infra::{AsyncRolloutScheduler, LengthElasticController, LayoutAwareDispatcher};
use crate::infra::logging::setup_logging;
use crate::learner::ContinuousLearner;
use crate::loops::runner::LoopRunner;
use crate::loops::verification::PredicateVerifier;
use crate::models::config::ModelRole;
use crate::models::registry::ModelRegistry;
use crate::privacy::classifier::DataClassifier;
use crate::privacy::consent::{ConsentManager, DataUse};
use crate::privacy::engine::{PrivacyEngine, SanitizationMode};
use crate::privacy::policy::PolicyStore;
use crate::rewards::VerifiableReward;
use crate::schema::actions::{ActionType, BaseAction};
use crate::security::pii::PIIDetector;
use crate::security::Sanitizer;
use crate::task::{TaskRunner, runner::load_tasks};
use crate::training::{ArenaStatus, ChampionChallenger, DistillationScheduler, RLTrainer, TrainingConfig};

#[derive(Parser)]
#[command(name = "zilli", version, about = "Zilli agent framework")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the Zilli server
    Serve {
        #[arg(short, long, default_value = "127.0.0.1")]
        host: String,
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
    /// Route a request through the privacy-aware hybrid pipeline
    Route {
        request: String,
        #[arg(short, long, default_value = "local_inference")]
        data_use: String,
        #[arg(long, default_value = "default_tenant")]
        tenant: String,
    },
    /// Generate a compliance report
    Report {
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(short, long)]
        format: Option<String>,
    },
    /// Check health of the system
    Health,
    /// List all available models
    Models,
    /// Run RL training (cispo / grpo)
    Train {
        #[arg(short, long, default_value = "cispo")]
        algorithm: String,
        #[arg(short, long, default_value_t = 100)]
        steps: u32,
    },
    /// Run skill evolution on trajectories
    Evolve {
        #[arg(short, long, default_value = "./skills")]
        skills_dir: String,
        #[arg(short, long, default_value_t = 3)]
        iterations: i32,
        #[arg(short, long, default_value_t = 5)]
        trajectories: usize,
    },
    /// Run a continuous learning cycle
    Learn {
        #[arg(long, default_value = "./data")]
        data_dir: String,
    },
    /// Run a verification loop
    Loop {
        input: String,
        #[arg(short, long, default_value_t = 3)]
        max_retries: i32,
        #[arg(short, long, default_value_t = 5)]
        min_length: usize,
    },
    /// Test adaptive planner vs executor decisions
    Schedule {
        task_type: String,
        #[arg(short, long, default_value_t = 0.7)]
        confidence: f64,
        #[arg(short, long, default_value_t = 100.0)]
        budget: f64,
    },
    /// Run a task evaluation
    Task {
        #[arg(default_value = "basic")]
        category: String,
    },
    /// Evaluate a trajectory reward
    Evaluate {
        #[arg(short, long, default_value_t = 5)]
        steps: u32,
    },
    /// Inspect internal subsystem state
    Inspect {
        #[arg(value_enum)]
        target: InspectTarget,
    },
    /// Run the sandbox environment
    Sandbox {
        #[arg(short, long, default_value_t = 10)]
        max_steps: i32,
        command: String,
    },
    /// Test dynamic length adaptation
    Adapt {
        #[arg(short, long, default_value_t = 4096)]
        initial: usize,
        #[arg(short, long, default_value_t = 512)]
        min: usize,
        #[arg(short, long, default_value_t = 8192)]
        max: usize,
        #[arg(short, long, default_value_t = 10)]
        iterations: usize,
    },
    /// List available industries and their compliance rules
    Industries,
    /// Run a champion-challenger arena match
    Arena {
        #[arg(long, default_value = "champion-v1")]
        champion: String,
        #[arg(long, default_value = "challenger-v2")]
        challenger: String,
        #[arg(short, long, default_value_t = 0.05)]
        win_gap: f64,
    },
    /// Run a distillation cycle
    Distill {
        #[arg(short, long, default_value_t = 100)]
        samples: usize,
    },
    /// Test async rollout scheduler
    Rollout {
        #[arg(short, long, default_value_t = 4)]
        concurrent: usize,
        #[arg(short, long, default_value_t = 5)]
        tasks: usize,
    },
}

#[derive(Clone, ValueEnum)]
enum InspectTarget {
    Trajectory,
    Models,
    Budget,
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let config_path = cli.config.to_str();
    let _cfg = load_config(config_path).unwrap_or_else(|e| {
        tracing::warn!("Config load failed: {}, using defaults", e);
        ZilliConfig::default()
    });

    match cli.command {
        Some(Commands::Serve { host, port }) => {
            tracing::info!("Starting Zilli server...");
            if let Err(e) = crate::server::run_server(&host, port).await {
                tracing::error!("Server failed: {}", e);
                eprintln!("Server error: {}", e);
            }
        }

        Some(Commands::Route { request, data_use, tenant }) => {
            setup_logging();
            let classifier = DataClassifier::new();
            let detector = PIIDetector::new();
            let sanitizer = Sanitizer::new();
            let consent_manager = ConsentManager::new();
            let gatekeeper_consent = consent_manager.clone();
            let policy = PolicyStore::new();
            let engine = PrivacyEngine::new(classifier, detector, sanitizer, consent_manager, policy);

            let data_use_enum: DataUse = data_use.parse().unwrap_or(DataUse::LocalInference);

            let verdict = engine.evaluate(
                &request,
                &data_use_enum,
                &tenant,
                SanitizationMode::Auto,
            );
            println!("=== Privacy Evaluation ===");
            println!("Verdict:      {}", serde_json::to_string_pretty(&serde_json::json!({
                "passed": verdict.passed,
                "data_class": format!("{:?}", verdict.data_class),
                "risk_score": verdict.risk_score,
                "sanitized": verdict.sanitized_text.is_some(),
            }))?);

            let gatekeeper = PrivacyGatekeeper::new(
                engine,
                PolicyStore::new(),
                gatekeeper_consent,
            );
            let registry = ModelRegistry::new();
            let executor = HybridExecutor::new(gatekeeper, registry);
            let result = executor.execute(&request, ModelRole::Executor, &data_use_enum, &tenant).await;
            println!("\n=== Hybrid Execution ===");
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "target": format!("{:?}", result.target),
                "verdict": result.verdict,
                "model": result.model_name,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "error": result.error,
                "warnings": result.warnings,
            }))?);
        }

        Some(Commands::Report { output, format }) => {
            let reporter = ComplianceReporter::new();
            let report = reporter.generate_report(
                &crate::audit::ComplianceFramework::GDPR,
                "default",
                &[],
                chrono::Utc::now(),
                chrono::Utc::now(),
            );
            match (output, format) {
                (Some(path), _) => {
                    let content = serde_json::to_string_pretty(&report)?;
                    tokio::fs::write(&path, content).await?;
                    println!("Report written to {}", path.display());
                }
                (None, Some(fmt)) if fmt == "json" => {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                }
                _ => {
                    println!("=== Compliance Report ===");
                    println!("Passed: {}", report.passed);
                    println!("Total requests: {}", report.total_requests);
                }
            }
        }

        Some(Commands::Train { algorithm, steps }) => {
            let config = TrainingConfig::new(&algorithm);
            let trainer = RLTrainer::new(Some(config));
            let batch: Vec<f64> = (0..steps).map(|i| i as f64 / steps as f64).collect();
            let loss = trainer.update(&batch);
            println!("=== Training Result ===");
            println!("{}", serde_json::to_string_pretty(&loss)?);
        }

        Some(Commands::Evolve { skills_dir, iterations, trajectories }) => {
            let engine = SkillEvolutionEngine::new(None);
            let trajs: Vec<serde_json::Value> = (0..trajectories)
                .map(|i| serde_json::json!({"error_summary": format!("error_{}", i)}))
                .collect();
            let results = engine.evolve(&trajs, &skills_dir, iterations);
            println!("=== Evolution Results ===");
            for r in &results {
                println!("  {}: {:.3} → {:.3} (improved: {})", r.skill_name, r.original_score, r.evolved_score, r.improved);
            }
        }

        Some(Commands::Learn { data_dir }) => {
            let store = TrajectoryStore::new();
            let mut learner = ContinuousLearner::new(store, 24, &data_dir);
            let cycle = learner.run_cycle().await;
            println!("=== Learning Cycle ===");
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "cycle_id": cycle.cycle_id,
                "start_time": cycle.start_time.to_rfc3339(),
                "new_trajectories": cycle.new_trajectories,
                "sft_triggered": cycle.sft_triggered,
                "sft_metrics": cycle.sft_metrics,
            }))?);
        }

        Some(Commands::Loop { input, max_retries, min_length }) => {
            let verifier = PredicateVerifier::new("min_length", Arc::new(move |_input, output| {
                output.len() > min_length
            }));
            let runner = LoopRunner::new(
                Box::new(|s| format!("processed: {}", s)),
                Box::new(verifier),
                max_retries,
            );
            let result = runner.run(&input).await;
            println!("=== Loop Result ===");
            println!("Passed: {}", result.passed);
            println!("Cycles: {}", result.cycles.len());
            println!("Duration: {}ms", result.total_duration_ms);
            if let Some(output) = result.final_output {
                println!("Output: {}", output);
            }
        }

        Some(Commands::Schedule { task_type, confidence, budget }) => {
            let scheduler = DynamicSOTAScheduler::new(budget, 0.05);
            let decision = scheduler.should_call_planner(&task_type, confidence);
            println!("=== Scheduler Decision ===");
            println!("Task type: {}", task_type);
            println!("Executor confidence: {:.2}", confidence);
            println!("Use planner: {}", decision);
            scheduler.record_call("executor-model", &task_type, true);
            println!("\nStats after call:");
            let stats = scheduler.stats();
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "total_calls": stats.total_calls,
                "planner_calls": stats.planner_calls,
                "executor_calls": stats.executor_calls,
                "total_cost": stats.total_cost,
                "remaining_budget": stats.remaining_budget,
                "emergency_mode": stats.emergency_mode,
            }))?);

            let cost = CostController::new(budget);
            let state = PlannerState::new(confidence, 0.6);
            let use_planner = cost.should_use_planner(&task_type, &state);
            println!("\nCost controller decision: use_planner={}", use_planner);
        }

        Some(Commands::Task { category }) => {
            let tasks = load_tasks(Some(&category));
            if tasks.is_empty() {
                println!("No tasks found for category '{}'", category);
                return Ok(());
            }
            for task in &tasks {
                let runner = TaskRunner::new(task.clone());
                let score = runner.evaluate(None);
                println!("=== Task: {} ({}) ===", task.name, task.id);
                println!("Score: {:.4}", score);
                println!("Max steps: {}", task.max_steps);
                if let Some(ref prompt) = task.prompt {
                    println!("Prompt: {}", prompt);
                }
            }
        }

        Some(Commands::Evaluate { steps }) => {
            let reward = VerifiableReward::new();
            let trajectory: Vec<serde_json::Value> = (0..steps)
                .map(|i| serde_json::json!({"action": format!("step_{}", i), "observation": "ok"}))
                .collect();
            let score = reward.compute(&trajectory, None);
            println!("=== Reward Evaluation ===");
            println!("Steps: {}", steps);
            println!("Reward score: {:.4}", score);
            println!("Schema weight: {:.2}", reward.schema_validity_weight);
            println!("Test weight: {:.2}", reward.test_passing_weight);
            println!("Efficiency weight: {:.2}", reward.efficiency_weight);
            println!("Safety weight: {:.2}", reward.safety_weight);
        }

        Some(Commands::Inspect { target }) => {
            match target {
                InspectTarget::Trajectory => {
                    let store = TrajectoryStore::new();
                    let cleaner = TrajectoryCleaner::new();
                    let stats = store.stats();
                    println!("=== Trajectory Store ===");
                    println!("{}", serde_json::to_string_pretty(&stats)?);
                    let sample = serde_json::json!({"action": "test", "observation": "data"});
                    let clean = cleaner.clean(&[sample]);
                    println!("Cleaner warnings: {:?}", clean.warnings);
                }
                InspectTarget::Models => {
                    let registry = ModelRegistry::new();
                    let models = registry.list_models();
                    println!("=== Model Registry ===");
                    println!("{}", serde_json::to_string_pretty(&models)?);
                }
                InspectTarget::Budget => {
                    let cost = CostController::new(1000.0);
                    cost.record_planner_call("code_gen", true);
                    cost.record_planner_call("code_gen", true);
                    cost.record_executor_call("code_gen", true);
                    let snap = cost.snapshot();
                    println!("=== Budget Snapshot ===");
                    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                        "remaining_budget": snap.remaining_budget,
                        "total_calls": snap.total_calls,
                        "calls_this_hour": snap.calls_this_hour,
                        "emergency_mode": snap.emergency_mode,
                    }))?);
                }
            }
        }

        Some(Commands::Sandbox { max_steps, command }) => {
            let sandbox = HermesSandbox::new(max_steps);
            sandbox.register_tool("bash", |s| Ok(format!("executed: {}", s)));
            let action = BaseAction {
                action_id: "1".into(),
                reasoning: Some(command),
                tool_name: ActionType::BashRun,
            };
            let result = sandbox.step(action).await;
            println!("=== Sandbox Result ===");
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "success": result.success,
                "observation": result.observation,
                "done": result.done,
                "reward": result.reward,
            }))?);
        }

        Some(Commands::Adapt { initial, min, max, iterations }) => {
            let controller = LengthElasticController::new(initial, min, max);
            let dispatcher = LayoutAwareDispatcher;
            for i in 0..iterations {
                let lens: Vec<usize> = (0..50).map(|_| {
                    let base = min + (max - min) / 2;
                    base + (i * 100) % (max / 2)
                }).collect();
                controller.adapt(&lens);
                let stats = controller.get_stats();
                println!("Iteration {}: cap={}, p95={}", i, stats["current_cap"], stats["p95"]);
            }
            let data: Vec<String> = (0..20).map(|i| format!("item_{}", i)).collect();
            let batches = dispatcher.dispatch(&data, 4);
            println!("Dispatched into {} batches", batches.len());

            let scheduler = AsyncRolloutScheduler::new(4);
            let rollout_fn: Arc<dyn Fn(String) -> Result<String, String> + Send + Sync> =
                Arc::new(|s: String| Ok(format!("processed: {}", s)));
            let tasks: Vec<String> = (0..5).map(|i| format!("task_{}", i)).collect();
            let results = scheduler.schedule(rollout_fn, tasks, 30).await;
            println!("\nRollout results: {} completed", results.iter().filter(|r| r.status == crate::infra::RolloutStatus::Completed).count());
        }

        Some(Commands::Industries) => {
            let registry = WorkflowRegistry::new();
            let industries = registry.list_industries();
            println!("=== Registered Industries ===");
            println!("{}", serde_json::to_string_pretty(&industries)?);
        }

        Some(Commands::Arena { champion, challenger, win_gap }) => {
            let arena = ChampionChallenger::new(win_gap);
            arena.register_model(&champion, "1.0", ArenaStatus::Champion);
            arena.register_model(&challenger, "2.0", ArenaStatus::Challenger);
            let result = arena.run_match(&champion, 0.85, 0.92);
            if let Some(m) = result {
                println!("=== Arena Match ===");
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "champion": m.champion,
                    "challenger": m.challenger,
                    "champion_score": m.champion_score,
                    "challenger_score": m.challenger_score,
                    "winner": m.winner,
                    "significant": m.significant,
                }))?);
            }
            println!("\nLeaderboard:");
            for m in arena.leaderboard() {
                println!("  {} v{} [{}]", m.name, m.version, m.status);
            }
        }

        Some(Commands::Distill { samples }) => {
            let scheduler = DistillationScheduler::new(0.5, 0.3, 0.2, 10, 1000);
            let batch = crate::training::data::make_dummy_distillation_samples(samples as i32);
            scheduler.add_batch(batch);
            println!("=== Distillation ===");
            println!("Samples: {}", samples);
            println!("Should distill: {}", scheduler.should_distill());
            if let Some(cycle) = scheduler.run_cycle() {
                println!("Cycle: {}", serde_json::to_string_pretty(&serde_json::json!({
                    "loss_bc": cycle.loss_bc,
                    "loss_rl": cycle.loss_rl,
                    "kl_divergence": cycle.kl_divergence,
                    "lora_triggered": cycle.lora_triggered,
                }))?);
            }
            println!("\nStats: {}", serde_json::to_string_pretty(&scheduler.stats())?);
        }

        Some(Commands::Rollout { concurrent, tasks }) => {
            let scheduler = AsyncRolloutScheduler::new(concurrent);
            let rollout_fn: Arc<dyn Fn(String) -> Result<String, String> + Send + Sync> =
                Arc::new(|s: String| Ok(format!("result: {}", s)));
            let task_list: Vec<String> = (0..tasks).map(|i| format!("rollout_{}", i)).collect();
            let results = scheduler.schedule(rollout_fn, task_list, 30).await;
            println!("=== Rollout Results ===");
            for r in &results {
                let status = format!("{:?}", r.status);
                println!("  {}: status={}, reward={:?}", r.task_id, status, r.reward);
            }
            println!("\nStats: {}", serde_json::to_string_pretty(&scheduler.get_stats())?);
        }

        Some(Commands::Health) | None => {
            let registry = ModelRegistry::new();
            let models = registry.health_check().await;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "status": "ok",
                "version": "0.3.0",
                "models": models,
            }))?);
        }

        Some(Commands::Models) => {
            let registry = ModelRegistry::new();
            let models = registry.list_models();
            println!("{}", serde_json::to_string_pretty(&models)?);
        }
    }

    Ok(())
}
