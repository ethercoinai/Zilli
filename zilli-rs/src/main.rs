#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    zilli::infra::logging::setup_logging();

    zilli::cli::run().await
}
