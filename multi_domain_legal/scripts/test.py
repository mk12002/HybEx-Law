import json

# Check a few samples from train_split.json
with open('data/train_split.json', 'r') as f:
    samples = json.load(f)

# Print first 5 samples with their extracted entities
for i, sample in enumerate(samples[:5]):
    print(f"\nSample {i+1}: {sample['sample_id']}")
    print(f"Query: {sample['query'][:100]}...")
    print(f"Extracted entities: {sample['extracted_entities']}")
