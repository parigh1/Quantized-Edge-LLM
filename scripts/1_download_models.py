from huggingface_hub import snapshot_download

MODELS = {
    "llama-1b-mental": "Parigh1/Llama-3.2-1B-Mental-Health-Friend",
    "gugu-merged":     "Parigh1/gugu-merged",
}

for name, repo in MODELS.items():
    print(f"\nDownloading {name} from {repo}...")
    snapshot_download(
        repo_id=repo,
        local_dir=f"models/{name}",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print(f"Done: models/{name}")
