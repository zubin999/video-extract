// Video Processing Service with Whisper Transcription
//
// Environment Variables:
// - ROLE: "api" or "worker" (default: "api")
// - BIND_ADDR: API server bind address (default: "0.0.0.0:8001")
// - REDIS_URL: Redis connection URL (default: "redis://127.0.0.1:6379")
// - DATA_DIR: Data storage directory (default: "./data")
// - QUEUE_NAME: Redis queue name (default: "video_tasks")
// - FRAME_INTERVAL_SECONDS: Frame extraction interval (default: 2)
// - MAX_CONCURRENT_TASKS: Maximum concurrent tasks per worker (default: min(CPU_count, 4))
// - WHISPER_MODEL: Model name like "small", "medium", "large" (default: "small")
// - WHISPER_MODEL_PATH: Full path to model file (overrides WHISPER_MODEL)
// - WHISPER_POOL_SIZE: Number of Whisper context instances (default: min(MAX_CONCURRENT_TASKS, 2))
// - WHISPER_THREADS: Number of threads per Whisper instance (default: auto-calculated)
// - WHISPER_USE_GPU: Enable GPU acceleration if available (default: false)
// - FFMPEG_THREADS: Number of threads for FFmpeg operations (default: CPU count)

use std::{
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use redis::{aio::MultiplexedConnection, AsyncCommands};
use serde::{Deserialize, Serialize};
use tokio::{fs as tokio_fs, net::TcpListener, process::Command};
use tower_http::services::ServeDir;
use tracing::{error, info, warn};

// -------------------- Shared types --------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubmitTaskReq {
    video_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SubmitTaskResp {
    task_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum TaskStatus {
    Queued,
    Processing,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskResult {
    frames: Vec<String>,
    transcript_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskRecord {
    task_id: String,
    video_url: String,
    status: TaskStatus,
    created_at: u64,
    updated_at: u64,
    error: Option<String>,
    result: Option<TaskResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueueMessage {
    task_id: String,
    video_url: String,
    data_dir: String,
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn task_key(task_id: &str) -> String {
    format!("task:{}", task_id)
}

// -------------------- Task Statistics --------------------
#[derive(Debug, Default)]
struct TaskStats {
    total_processed: std::sync::atomic::AtomicU64,
    total_failed: std::sync::atomic::AtomicU64,
    currently_processing: std::sync::atomic::AtomicU64,
}

impl TaskStats {
    fn start_task(&self) {
        self.currently_processing
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn complete_task(&self, success: bool) {
        self.currently_processing
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        if success {
            self.total_processed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.total_failed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.total_processed
                .load(std::sync::atomic::Ordering::Relaxed),
            self.total_failed.load(std::sync::atomic::Ordering::Relaxed),
            self.currently_processing
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }
}

// -------------------- Whisper Context Pool --------------------
struct WhisperContextPool {
    contexts: Vec<whisper_rs::WhisperContext>,
    model_path: String,
    current_index: std::sync::atomic::AtomicUsize,
}

impl WhisperContextPool {
    fn new(model_path: &str, pool_size: usize) -> Result<Self> {
        info!("Loading {} Whisper model instances from: {}", pool_size, model_path);

        // Check if model file exists
        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow!("Whisper model file not found at: {}", model_path));
        }

        // Enable GPU if available (optional optimization)
        let _use_gpu = if let Ok(use_gpu) = env::var("WHISPER_USE_GPU") {
            if use_gpu.to_lowercase() == "true" {
                info!("GPU acceleration enabled for Whisper");
                true
                // Note: This depends on whisper-rs GPU support
            } else {
                false
            }
        } else {
            false
        };

        let mut contexts = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let params = whisper_rs::WhisperContextParameters::default();
            let context = whisper_rs::WhisperContext::new_with_params(model_path, params)
                .with_context(|| format!("failed to load model instance {} at {}", i, model_path))?;
            contexts.push(context);
        }

        info!("Whisper model pool loaded successfully: {} instances from {}", pool_size, model_path);
        Ok(Self {
            contexts,
            model_path: model_path.to_string(),
            current_index: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn get_context(&self) -> &whisper_rs::WhisperContext {
        let index = self.current_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.contexts.len();
        &self.contexts[index]
    }

    fn create_state(&self) -> Result<whisper_rs::WhisperState> {
        self.get_context().create_state().map_err(Into::into)
    }

    fn model_path(&self) -> &str {
        &self.model_path
    }
    
    fn pool_size(&self) -> usize {
        self.contexts.len()
    }
}

// -------------------- API server --------------------
#[derive(Clone)]
struct ApiState {
    redis: MultiplexedConnection,
    data_dir: PathBuf,
    queue_name: String,
}

async fn handler_submit(
    State(state): State<ApiState>,
    Json(payload): Json<SubmitTaskReq>,
) -> Result<(StatusCode, Json<SubmitTaskResp>), (StatusCode, String)> {
    let SubmitTaskReq { video_url } = payload;
    if video_url.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "video_url is required".into()));
    }

    let task_id = uuid::Uuid::new_v4().to_string();
    let task_dir = state.data_dir.join(&task_id);
    let frames_dir = task_dir.join("frames");

    tokio_fs::create_dir_all(&frames_dir)
        .await
        .map_err(internal_err)?;

    // Persist initial record
    let record = TaskRecord {
        task_id: task_id.clone(),
        video_url: video_url.clone(),
        status: TaskStatus::Queued,
        created_at: now_ts(),
        updated_at: now_ts(),
        error: None,
        result: None,
    };

    let key = task_key(&task_id);
    let record_json = serde_json::to_string(&record).map_err(internal_err)?;
    let mut conn = state.redis.clone();
    conn.set::<_, _, ()>(&key, record_json)
        .await
        .map_err(internal_err)?;

    // Push queue message
    let msg = QueueMessage {
        task_id: task_id.clone(),
        video_url,
        data_dir: state.data_dir.to_str().unwrap_or("./data").to_string(),
    };
    let msg_json = serde_json::to_string(&msg).map_err(internal_err)?;
    conn.lpush::<_, _, ()>(&state.queue_name, msg_json)
        .await
        .map_err(internal_err)?;

    Ok((StatusCode::OK, Json(SubmitTaskResp { task_id })))
}

#[derive(Debug, Deserialize)]
struct QueryReq {
    task_id: String,
}

async fn handler_query(
    State(state): State<ApiState>,
    Query(q): Query<QueryReq>,
) -> Result<Json<TaskRecord>, (StatusCode, String)> {
    let mut conn = state.redis.clone();
    let key = task_key(&q.task_id);
    let val: Option<String> = conn.get(&key).await.map_err(internal_err)?;
    match val {
        Some(v) => {
            let record: TaskRecord = serde_json::from_str(&v).map_err(internal_err)?;
            Ok(Json(record))
        }
        None => Err((StatusCode::NOT_FOUND, "task not found".into())),
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    timestamp: u64,
    redis_connected: bool,
}

async fn handler_health(
    State(state): State<ApiState>,
) -> Result<Json<HealthResponse>, (StatusCode, String)> {
    let mut conn = state.redis.clone();
    // Use a simple Redis command to check connectivity
    let redis_connected = conn
        .get::<&str, Option<String>>("__health_check__")
        .await
        .is_ok();

    Ok(Json(HealthResponse {
        status: if redis_connected {
            "healthy".to_string()
        } else {
            "unhealthy".to_string()
        },
        timestamp: now_ts(),
        redis_connected,
    }))
}

async fn run_api() -> Result<()> {
    let bind = env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8001".to_string());
    let redis_url = env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let data_dir = env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let queue_name = env::var("QUEUE_NAME").unwrap_or_else(|_| "video_tasks".to_string());

    info!(bind = %bind, redis = %redis_url, data_dir = %data_dir, queue = %queue_name, "API config loaded");

    tokio_fs::create_dir_all(&data_dir).await?;

    let client = redis::Client::open(redis_url.clone())?;
    let conn = client.get_multiplexed_tokio_connection().await?;
    info!("Connected to Redis");

    let state = ApiState {
        redis: conn,
        data_dir: PathBuf::from(&data_dir),
        queue_name,
    };

    let static_service = ServeDir::new(&data_dir);

    let app = Router::new()
        .route("/video/process", post(handler_submit))
        .route("/task/query", get(handler_query))
        .route("/health", get(handler_health))
        .nest_service("/static", static_service)
        .with_state(state);

    info!(%bind, "API listening");
    let listener = TcpListener::bind(&bind).await?;
    axum::serve(listener, app.into_make_service()).await?;

    Ok(())
}

// -------------------- Worker --------------------
async fn run_worker() -> Result<()> {
    let redis_url = env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let queue_name = env::var("QUEUE_NAME").unwrap_or_else(|_| "video_tasks".to_string());

    // Configure concurrent task processing
    let max_concurrent_tasks: usize = env::var("MAX_CONCURRENT_TASKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            // Default to CPU count, but cap at reasonable limit for video processing
            let cpu_count = num_cpus::get();
            std::cmp::min(cpu_count, 4)
        });

    info!(
        redis = %redis_url,
        queue = %queue_name,
        max_concurrent = max_concurrent_tasks,
        "Worker config loaded"
    );

    // Initialize Whisper model pool at startup
    let model_path = if let Ok(p) = env::var("WHISPER_MODEL_PATH") {
        p
    } else {
        let model = env::var("WHISPER_MODEL").unwrap_or_else(|_| "small".to_string());
        let models_dir = "./models";

        // Ensure models directory exists
        if let Err(e) = std::fs::create_dir_all(models_dir) {
            warn!("Failed to create models directory: {}", e);
        }

        format!("{}/ggml-{}.bin", models_dir, model)
    };

    // Create multiple Whisper contexts for better concurrency
    let whisper_pool_size: usize = env::var("WHISPER_POOL_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::cmp::min(max_concurrent_tasks, 2)); // Default to min(concurrent_tasks, 2)

    let whisper_pool = Arc::new(
        WhisperContextPool::new(&model_path, whisper_pool_size)
            .context("Failed to initialize Whisper model pool")?
    );

    let client = redis::Client::open(redis_url.clone())?;
    info!("Connected to Redis");

    // Create a semaphore to limit concurrent tasks
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent_tasks));
    let mut active_tasks = tokio::task::JoinSet::new();
    let stats = Arc::new(TaskStats::default());

    // Start a background task for periodic stats reporting
    let stats_clone = stats.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let (processed, failed, current) = stats_clone.get_stats();
            info!(
                processed = processed,
                failed = failed,
                currently_processing = current,
                "Worker statistics"
            );
        }
    });

    info!(queue = %queue_name, "Worker started with {} concurrent task slots", max_concurrent_tasks);

    // Setup graceful shutdown
    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            // Handle shutdown signal
            _ = &mut shutdown => {
                info!("Shutdown signal received, waiting for active tasks to complete...");

                // Stop accepting new tasks and wait for existing ones to complete
                while let Some(result) = active_tasks.join_next().await {
                    if let Err(e) = result {
                        error!("Task panicked during shutdown: {}", e);
                    }
                }

                let (processed, failed, _) = stats.get_stats();
                info!(
                    processed = processed,
                    failed = failed,
                    "Worker shutdown complete"
                );
                break;
            }

            // Clean up completed tasks or process new ones
            _ = async {
                // First, clean up any completed tasks
                while let Some(result) = active_tasks.try_join_next() {
                    match result {
                        Ok(_) => {}, // Task completed successfully
                        Err(e) => error!("Task panicked: {}", e),
                    }
                }

                // Check if we have available slots for new tasks
                if semaphore.available_permits() == 0 {
                    // Wait for at least one task to complete
                    if let Some(result) = active_tasks.join_next().await {
                        if let Err(e) = result {
                            error!("Task panicked: {}", e);
                        }
                    }
                    return;
                }

                // Try to get a new task from the queue
                let mut conn = match client.get_multiplexed_tokio_connection().await {
                    Ok(conn) => conn,
                    Err(e) => {
                        error!("Redis connection error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        return;
                    }
                };

                let msg: Option<(String, String)> = match conn.brpop(&queue_name, 1f64).await {
                    Ok(msg) => msg,
                    Err(e) => {
                        error!("Redis error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        return;
                    }
                };

                let (_list, payload) = match msg {
                    Some(v) => v,
                    None => {
                        // No tasks available
                        return;
                    }
                };

                let qm: QueueMessage = match serde_json::from_str(&payload) {
                    Ok(v) => v,
                    Err(e) => {
                        error!(error = ?e, "Invalid queue message, skipping");
                        return;
                    }
                };

                // Acquire semaphore permit
                let permit = match semaphore.clone().acquire_owned().await {
                    Ok(permit) => permit,
                    Err(_) => {
                        error!("Failed to acquire semaphore permit");
                        return;
                    }
                };

                // Spawn task for concurrent processing
                let client_clone = client.clone();
                let whisper_pool_clone = whisper_pool.clone();
                let stats_clone = stats.clone();
                let task_id = qm.task_id.clone();

                active_tasks.spawn(async move {
                    let _permit = permit; // Hold permit until task completes

                    stats_clone.start_task();
                    info!(task_id = %task_id, "Starting concurrent task processing");

                    let success = if let Err(e) = process_task(&client_clone, &qm, whisper_pool_clone).await {
                        error!(task_id = %qm.task_id, error = ?e, "Task failed");
                        let _ = set_task_failed(&client_clone, &qm.task_id, &format!("{e:#}"), &qm.video_url).await;
                        false
                    } else {
                        info!(task_id = %task_id, "Task completed successfully");
                        true
                    };

                    stats_clone.complete_task(success);
                });

                info!(
                    active_tasks = active_tasks.len(),
                    available_slots = semaphore.available_permits(),
                    "Task queued for processing"
                );
            } => {}
        }
    }

    Ok(())
}

async fn set_task_status(
    client: &redis::Client,
    task_id: &str,
    status: TaskStatus,
    error: Option<String>,
    result: Option<TaskResult>,
    video_url: &str,
) -> Result<()> {
    let mut conn = client.get_multiplexed_tokio_connection().await?;
    let key = task_key(task_id);
    let val: Option<String> = conn.get(&key).await?;
    let mut record = if let Some(v) = val {
        serde_json::from_str::<TaskRecord>(&v).unwrap_or(TaskRecord {
            task_id: task_id.to_string(),
            video_url: video_url.to_string(),
            status: TaskStatus::Queued,
            created_at: now_ts(),
            updated_at: now_ts(),
            error: None,
            result: None,
        })
    } else {
        TaskRecord {
            task_id: task_id.to_string(),
            video_url: video_url.to_string(),
            status: TaskStatus::Queued,
            created_at: now_ts(),
            updated_at: now_ts(),
            error: None,
            result: None,
        }
    };

    record.status = status;
    record.error = error;
    record.result = result;
    record.updated_at = now_ts();

    conn.set::<_, _, ()>(key, serde_json::to_string(&record)?)
        .await?;
    Ok(())
}

async fn set_task_failed(
    client: &redis::Client,
    task_id: &str,
    err: &str,
    video_url: &str,
) -> Result<()> {
    set_task_status(
        client,
        task_id,
        TaskStatus::Failed,
        Some(err.to_string()),
        None,
        video_url,
    )
    .await
}

async fn process_task(
    client: &redis::Client,
    qm: &QueueMessage,
    whisper_pool: Arc<WhisperContextPool>,
) -> Result<()> {
    info!(task_id = %qm.task_id, url = %qm.video_url, "Processing task");
    set_task_status(
        client,
        &qm.task_id,
        TaskStatus::Processing,
        None,
        None,
        &qm.video_url,
    )
    .await?;

    let task_dir = Path::new(&qm.data_dir).join(&qm.task_id);
    let frames_dir = task_dir.join("frames");

    tokio_fs::create_dir_all(&frames_dir).await?;

    // 1. Download video
    let video_path = task_dir.join("source.mp4");
    download_file(&qm.video_url, &video_path)
        .await
        .context("download video")?;

    // 2 & 3. Extract frames and audio in parallel
    let interval: u64 = env::var("FRAME_INTERVAL_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let audio_wav = task_dir.join("audio.wav");

    // Start frame extraction and audio extraction concurrently
    let frames_task = tokio::spawn({
        let video_path = video_path.clone();
        let frames_dir = frames_dir.clone();
        async move { ffmpeg_extract_frames(&video_path, &frames_dir, interval).await }
    });

    let audio_task = tokio::spawn({
        let video_path = video_path.clone();
        let audio_wav = audio_wav.clone();
        async move { ffmpeg_extract_audio_wav(&video_path, &audio_wav).await }
    });

    // Wait for both tasks to complete
    let (frames_result, audio_result) = tokio::join!(frames_task, audio_task);
    frames_result
        .map_err(|e| anyhow!("frames task panicked: {}", e))?
        .context("extract frames")?;
    audio_result
        .map_err(|e| anyhow!("audio task panicked: {}", e))?
        .context("extract audio")?;

    // 4 & 5. Transcribe and build frame list in parallel
    let transcript_txt = task_dir.join("transcript.txt");

    let transcribe_task = tokio::spawn({
        let whisper_pool = whisper_pool.clone();
        let audio_wav = audio_wav.clone();
        let transcript_txt = transcript_txt.clone();
        async move { transcribe_whisper_with_pool(whisper_pool, &audio_wav, &transcript_txt).await }
    });

    let frames_list_task = tokio::spawn({
        let frames_dir = frames_dir.clone();
        let task_id = qm.task_id.clone();
        async move {
            let mut frames: Vec<String> = vec![];
            let mut entries = tokio_fs::read_dir(&frames_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let name = entry.file_name();
                let name = name.to_string_lossy().to_string();
                if name.to_lowercase().ends_with(".jpg")
                    || name.to_lowercase().ends_with(".jpeg")
                    || name.to_lowercase().ends_with(".png")
                {
                    frames.push(format!("/static/{}/frames/{}", task_id, name));
                }
            }
            frames.sort();
            Ok::<Vec<String>, anyhow::Error>(frames)
        }
    });

    // Wait for both transcription and frame listing to complete
    let (transcribe_result, frames_result) = tokio::join!(transcribe_task, frames_list_task);
    transcribe_result
        .map_err(|e| anyhow!("transcribe task panicked: {}", e))?
        .context("transcribe audio")?;
    let frames = frames_result
        .map_err(|e| anyhow!("frames list task panicked: {}", e))?
        .context("build frames list")?;

    let result = TaskResult {
        frames,
        transcript_url: format!("/static/{}/transcript.txt", qm.task_id),
    };

    set_task_status(
        client,
        &qm.task_id,
        TaskStatus::Done,
        None,
        Some(result),
        &qm.video_url,
    )
    .await?;
    info!(task_id = %qm.task_id, "Task done");
    Ok(())
}

async fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout
        .build()?;
        
    let resp = client.get(url).send().await?.error_for_status()?;
    let total_size = resp.content_length();
    
    let parent = dest.parent().ok_or_else(|| anyhow!("invalid dest path"))?;
    tokio_fs::create_dir_all(parent).await?;
    let mut file = tokio_fs::File::create(dest).await?;

    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let mut stream = resp.bytes_stream();
    let mut downloaded = 0u64;
    let start_time = std::time::Instant::now();
    
    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        file.write_all(&bytes).await?;
        downloaded += bytes.len() as u64;
        
        // Log progress for large files
        if let Some(total) = total_size {
            if total > 10_000_000 && downloaded % 5_000_000 == 0 { // Log every 5MB for files > 10MB
                let progress = (downloaded as f64 / total as f64) * 100.0;
                let speed = downloaded as f64 / start_time.elapsed().as_secs_f64() / 1_000_000.0; // MB/s
                info!("Download progress: {:.1}% ({:.1} MB/s)", progress, speed);
            }
        }
    }
    
    file.flush().await?;
    
    let elapsed = start_time.elapsed();
    let speed = downloaded as f64 / elapsed.as_secs_f64() / 1_000_000.0; // MB/s
    info!("Download completed: {} bytes in {:.2}s ({:.1} MB/s)", downloaded, elapsed.as_secs_f32(), speed);

    Ok(())
}

async fn ffmpeg_extract_frames(
    video_path: &Path,
    frames_dir: &Path,
    interval_seconds: u64,
) -> Result<()> {
    let pattern = frames_dir.join("frame_%05d.jpg");
    
    // Get number of CPU threads for FFmpeg
    let threads = env::var("FFMPEG_THREADS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or_else(|| num_cpus::get() as u32);
    
    let status = Command::new("ffmpeg")
        .arg("-y") // Overwrite output files
        .arg("-threads").arg(threads.to_string()) // Use multiple threads
        .arg("-i").arg(video_path)
        .arg("-vf").arg(format!("fps=1/{}", interval_seconds))
        .arg("-q:v").arg("3") // High quality JPEG (1-31, lower is better)
        .arg("-preset").arg("fast") // Fast encoding preset
        .arg(pattern)
        .status()
        .await?;

    if !status.success() {
        return Err(anyhow!(
            "ffmpeg extract frames failed with status: {:?}",
            status
        ));
    }
    Ok(())
}

async fn ffmpeg_extract_audio_wav(video_path: &Path, wav_path: &Path) -> Result<()> {
    // Get number of CPU threads for FFmpeg
    let threads = env::var("FFMPEG_THREADS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or_else(|| num_cpus::get() as u32);
        
    let status = Command::new("ffmpeg")
        .arg("-y") // Overwrite output files
        .arg("-threads").arg(threads.to_string()) // Use multiple threads
        .arg("-i").arg(video_path)
        .arg("-vn") // No video
        .arg("-ac").arg("1") // Mono audio
        .arg("-ar").arg("16000") // 16kHz sample rate
        .arg("-acodec").arg("pcm_s16le") // PCM 16-bit little-endian
        .arg("-f").arg("wav") // WAV format
        .arg(wav_path)
        .status()
        .await?;

    if !status.success() {
        return Err(anyhow!(
            "ffmpeg extract audio failed with status: {:?}",
            status
        ));
    }
    Ok(())
}

async fn transcribe_whisper_with_pool(
    whisper_pool: Arc<WhisperContextPool>,
    wav_path: &Path,
    out_txt: &Path,
) -> Result<()> {
    let start_time = std::time::Instant::now();
    info!("Starting transcription for: {:?}", wav_path);

    // Load wav samples (mono i16/float -> f32) in a blocking task to avoid blocking the async runtime
    let samples_and_duration = tokio::task::spawn_blocking({
        let wav_path = wav_path.to_path_buf();
        move || -> Result<(Vec<f32>, f32)> {
            let mut reader = hound::WavReader::open(&wav_path)
                .with_context(|| format!("Failed to open audio file: {:?}", wav_path))?;
            let spec = reader.spec();

            if spec.channels != 1 {
                warn!(
                    "Audio is not mono (channels: {}), proceeding but results may be suboptimal",
                    spec.channels
                );
            }

            if spec.sample_rate != 16000 {
                warn!(
                    "Audio sample rate is {} Hz, expected 16000 Hz",
                    spec.sample_rate
                );
            }

            let mut samples_f32: Vec<f32> = Vec::with_capacity(reader.len() as usize);
            match spec.sample_format {
                hound::SampleFormat::Int => {
                    let max_ampl = i16::MAX as f32;
                    for s in reader.samples::<i16>() {
                        let v = s? as f32 / max_ampl;
                        samples_f32.push(v);
                    }
                }
                hound::SampleFormat::Float => {
                    for s in reader.samples::<f32>() {
                        let v = s?;
                        samples_f32.push(v);
                    }
                }
            }

            let audio_duration = samples_f32.len() as f32 / spec.sample_rate as f32;
            Ok((samples_f32, audio_duration))
        }
    }).await??;

    let (samples_f32, audio_duration) = samples_and_duration;
    info!(
        "Loaded audio: {:.2}s duration, {} samples",
        audio_duration,
        samples_f32.len()
    );

    // Use the pre-loaded whisper context pool (no mutex needed!)
    let transcript = tokio::task::spawn_blocking({
        let whisper_pool = whisper_pool.clone();
        let samples = samples_f32.clone();
        move || -> Result<String> {
            info!(
                "Using Whisper context pool: {} instances available, model: {}",
                whisper_pool.pool_size(),
                whisper_pool.model_path()
            );

            let mut state = whisper_pool
                .create_state()
                .context("Failed to create Whisper state")?;

            let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::default());
            let n_threads: i32 = env::var("WHISPER_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| {
                    // Use fewer threads per instance when we have multiple contexts
                    let total_threads = num_cpus::get().max(1);
                    let threads_per_instance = std::cmp::max(1, total_threads / whisper_pool.pool_size());
                    threads_per_instance as i32
                });
            params.set_n_threads(n_threads);

            // Auto language detection
            params.set_language(Some("auto"));

            // Optional: Set other parameters for better performance
            params.set_translate(false); // Don't translate to English
            params.set_no_context(true); // Disable context from previous segments for faster processing
            params.set_single_segment(false); // Allow multiple segments

            info!("Starting Whisper inference with {} threads", n_threads);
            let inference_start = std::time::Instant::now();

            state
                .full(params, &samples)
                .context("Whisper inference failed")?;

            let inference_time = inference_start.elapsed();
            info!(
                "Whisper inference completed in {:.2}s",
                inference_time.as_secs_f32()
            );

            let num_segments = state.full_n_segments()?;
            let mut transcript = String::new();
            for i in 0..num_segments {
                let seg = state.full_get_segment_text(i).unwrap_or_default();
                transcript.push_str(seg.trim());
                transcript.push('\n');
            }

            info!(
                "Transcription generated {} segments, {} characters",
                num_segments,
                transcript.len()
            );
            Ok(transcript)
        }
    }).await??;

    tokio_fs::write(out_txt, &transcript)
        .await
        .with_context(|| format!("Failed to write transcript to: {:?}", out_txt))?;

    let total_time = start_time.elapsed();
    info!(
        "Transcription completed in {:.2}s (realtime factor: {:.2}x)",
        total_time.as_secs_f32(),
        audio_duration / total_time.as_secs_f32()
    );

    Ok(())
}

// // Keep the old function for backward compatibility if needed
// async fn transcribe_whisper(model_path: &str, wav_path: &Path, out_txt: &Path) -> Result<()> {
//     // Load wav samples (mono i16/float -> f32)
//     let mut reader = hound::WavReader::open(wav_path)?;
//     let spec = reader.spec();
//     if spec.channels != 1 {
//         warn!("audio is not mono, proceeding but results may be suboptimal");
//     }

//     let mut samples_f32: Vec<f32> = Vec::with_capacity(reader.len() as usize);
//     match spec.sample_format {
//         hound::SampleFormat::Int => {
//             let max_ampl = i16::MAX as f32;
//             for s in reader.samples::<i16>() {
//                 let v = s? as f32 / max_ampl;
//                 samples_f32.push(v);
//             }
//         }
//         hound::SampleFormat::Float => {
//             for s in reader.samples::<f32>() {
//                 let v = s?;
//                 samples_f32.push(v);
//             }
//         }
//     }

//     // Initialize whisper
//     let ctx = whisper_rs::WhisperContext::new_with_params(
//         model_path,
//         whisper_rs::WhisperContextParameters::default(),
//     )
//     .with_context(|| format!("failed to load model at {}", model_path))?;
//     let mut state = ctx.create_state()?;

//     let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::default());
//     let n_threads: i32 = num_cpus::get().max(1) as i32;
//     params.set_n_threads(n_threads);

//     // Auto language detection
//     params.set_language(Some("auto"));

//     state.full(params, &samples_f32)?;

//     let num_segments = state.full_n_segments()?;
//     let mut transcript = String::new();
//     for i in 0..num_segments {
//         let seg = state.full_get_segment_text(i).unwrap_or_default();
//         transcript.push_str(seg.trim());
//         transcript.push('\n');
//     }

//     tokio_fs::write(out_txt, transcript).await?;
//     Ok(())
// }

fn internal_err<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let role = env::var("ROLE").unwrap_or_else(|_| "api".to_string());
    info!(%role, "Starting service");

    let res = match role.as_str() {
        "worker" => run_worker().await,
        _ => run_api().await,
    };

    if let Err(e) = &res {
        error!(error = %format!("{e:#}"), "Service exited with error");
    } else {
        info!("Service exited normally");
    }

    res
}
