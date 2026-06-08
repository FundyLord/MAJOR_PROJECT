from datasets import load_dataset

try:
    # 1. Load the full dataset
    ds = load_dataset("AbdomenAtlas/AbdomenAtlas3.0", streaming=True)
    
    # 2. Grab the first actual row to force a download of the first chunk
    print("🛰️ Attempting to fetch first sample...")
    sample = next(iter(ds['train']))
    
    print("\n✅ Success! Available keys in the full dataset:")
    print(sample.keys())
    
    print("\n📝 Sample data preview:")
    for key, value in sample.items():
        # Print the type/size instead of full content for cleanliness
        print(f"Key: {key} | Type: {type(value)}")

except Exception as e:
    print(f"\n❌ Failed to access data: {e}")