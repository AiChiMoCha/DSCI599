import kagglehub

# Download latest version
path = kagglehub.dataset_download("beatoa/spamassassin-public-corpus")

print("Path to dataset files:", path)