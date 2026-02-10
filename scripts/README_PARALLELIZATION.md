# Parallelized FBA Training Set Generation

This guide explains how to use the optimized scripts for generating FBA-simulated training datasets using parallelization on HPC clusters.

## Overview

The training set generation has been optimized with two levels of parallelization:

2. **Multiprocessing**: Parallelize FBA simulations within each node using multiple CPU cores


---

## Files

- `scripts/create_reservoir_training_dataset.py` - Main script with parallelization support
- `scripts/merge_reservoir_training_dataset.py` - Merges partial results from array jobs
- `scripts/slurm/create_reservoir_training_dataset.slurm` - SLURM array job script
- `scripts/slurm/merge_reservoir_training_dataset.slurm` - SLURM merge job script
- `configs/reservoir_training_dataset.yaml` - Configuration file

---

## Usage Options

### Option 1: Local Testing (Single Worker)

For testing or small datasets:

```bash
python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml
```

### Option 2: Local with Multiprocessing

Use multiple CPU cores on a single machine:

```bash
# Use 8 workers
python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml -w 8

# Use all available cores
python scripts/create_reservoir_training_dataset.py configs/reservoir_training_dataset.yaml -w $(nproc)
```

### Option 3: SLURM Array Jobs (Recommended for HPC)

Best for large datasets (900+ conditions) on HPC clusters.

#### Step 1: Submit Array Job

```bash
# Submit with 10 array tasks (each processes ~92 conditions)
sbatch --array=0-9 scripts/slurm/create_reservoir_training_dataset.slurm

# Check job status
squeue -u $USER

# View logs
tail -f logs/reservoir_train_*.out
```

#### Step 2: Merge Results

After all array tasks complete:

```bash
# Manual merge
python scripts/merge_reservoir_training_dataset.py aidaying_iml1515_UB

# Or submit as dependent job (automatically runs after array completes)
sbatch --dependency=afterok:<JOB_ID> scripts/slurm/merge_reservoir_training_dataset.slurm
```

**Note**: Replace `<JOB_ID>` with the array job ID from Step 1.

---

## Configuration

### SLURM Array Job Settings

Edit `scripts/slurm/create_reservoir_training_dataset.slurm`:

```bash
# Number of array tasks (split 924 conditions across N tasks)
NUM_ARRAY_TASKS=10      # 10 tasks × ~92 conditions each

# Workers per task (should match --cpus-per-task)
NUM_WORKERS=8           # 8 parallel FBA simulations per task

#SBATCH --cpus-per-task=8    # Match NUM_WORKERS
#SBATCH --mem=16G            # Memory per task (adjust if needed)
#SBATCH --time=06:00:00      # Time limit per task
```

**Scaling Guidelines:**

| Conditions | Array Tasks | CPUs/Task | Total CPUs | Expected Time |
|------------|-------------|-----------|------------|---------------|
| 924        | 10          | 8         | 80         | ~1-2 hours    |
| 924        | 20          | 4         | 80         | ~1-2 hours    |
| 924        | 5           | 16        | 80         | ~1-2 hours    |

### Command-Line Arguments

```bash
python scripts/create_reservoir_training_dataset.py CONFIG [OPTIONS]

Options:
  --array-task-id, -a      SLURM array task ID (use $SLURM_ARRAY_TASK_ID)
  --num-array-tasks, -n    Total number of array tasks
  --num-workers, -w        Number of parallel workers (default: 1)
  --checkpoint-interval, -c Save checkpoint every N conditions (default: 50)
```

### YAML Configuration

Edit `configs/reservoir_training_dataset.yaml`:

```yaml
# Training parameters
size_i: 50              # Samples per condition (50 = 46,200 total samples)

# Model parameters
cobraname: 'iML1515_duplicated'
mediumbound: 'UB'

# Output
output_prefix: 'aidaying_iml1515'  # Output: aidaying_iml1515_UB.npz
```

---

## Performance Optimization

### Determining Optimal Configuration

1. **Total CPUs available**: Check your HPC allocation
2. **Memory per condition**: ~500MB-2GB (depends on model size)
3. **Calculate**:
   - `NUM_ARRAY_TASKS × NUM_WORKERS = Total CPUs`
   - `NUM_WORKERS × 2GB ≈ Memory per task`

### Example Configurations

#### Conservative (Long runtime, low resources)
```bash
NUM_ARRAY_TASKS=5
NUM_WORKERS=4
--cpus-per-task=4
--mem=8G
--time=12:00:00
```

#### Balanced (Recommended)
```bash
NUM_ARRAY_TASKS=10
NUM_WORKERS=8
--cpus-per-task=8
--mem=16G
--time=06:00:00
```

#### Aggressive (Fast, high resources)
```bash
NUM_ARRAY_TASKS=20
NUM_WORKERS=8
--cpus-per-task=8
--mem=16G
--time=03:00:00
```

---

## Monitoring Progress

### Real-time Progress

The script prints progress updates every 10 conditions:

```
Using multiprocessing with 8 workers
Processing 924 conditions...
  Condition 42: SUCCESS (50 samples)
  Condition 38: SUCCESS (50 samples)
  ...
  >> Progress: 100/924 conditions (10.8%), Rate: 15.2 cond/min, ETA: 54.1 min
```

### Check SLURM Jobs

```bash
# List all array tasks
squeue -u $USER --array

# Check specific task
squeue -j <JOB_ID>

# View logs in real-time
tail -f logs/reservoir_train_<JOB_ID>_<TASK_ID>.out
```

### Verify Partial Results

```bash
# Check partial files created
ls -lh data/processed/partial/

# Count partial files
ls data/processed/partial/aidaying_iml1515_UB_part*.npz | wc -l
```

---

## Troubleshooting

### Problem: Job runs out of memory

**Solution**: Increase `--mem` or reduce `NUM_WORKERS`

```bash
#SBATCH --mem=32G          # Increase memory
NUM_WORKERS=4              # Reduce workers
```

### Problem: Job times out

**Solution**: Increase time limit or split into more array tasks

```bash
#SBATCH --time=12:00:00    # Increase time
NUM_ARRAY_TASKS=20         # More tasks = less work per task
```

### Problem: Some conditions fail

**Solution**: Check logs for specific errors. The script continues processing other conditions. Failed conditions are logged but don't stop the job.

```bash
grep "WARNING" logs/reservoir_train_*.out
```

### Problem: Array job partially fails

**Solution**: Rerun only failed tasks

```bash
# If tasks 3, 7, 9 failed
sbatch --array=3,7,9 scripts/slurm/create_reservoir_training_dataset.slurm
```

### Problem: "No partial files found" during merge

**Solution**: Ensure all array tasks completed successfully

```bash
# Check if all tasks finished
ls data/processed/partial/ | grep "part"

# Check job status
sacct -j <JOB_ID> --format=JobID,State,ExitCode
```

---

## Output Files

### During Array Job Execution

```
data/processed/partial/
├── aidaying_iml1515_UB_part0000.npz       # Array task 0
├── aidaying_iml1515_UB_part0000_meta.pkl
├── aidaying_iml1515_UB_part0001.npz       # Array task 1
├── aidaying_iml1515_UB_part0001_meta.pkl
├── ...
└── aidaying_iml1515_UB_part0009.npz       # Array task 9
```

### After Merge

```
data/processed/
└── aidaying_iml1515_UB.npz                # Final combined result
```

---

## Cleanup

Remove partial files after successful merge:

```bash
# Manual cleanup
rm -rf data/processed/partial/

# Or use automatic cleanup during merge
python scripts/merge_reservoir_training_dataset.py aidaying_iml1515_UB --cleanup
```

---

## Quick Start Checklist

- [ ] Edit `configs/reservoir_training_dataset.yaml` with your parameters
- [ ] Update `NUM_ARRAY_TASKS` and `NUM_WORKERS` in SLURM script
- [ ] Update email address in SLURM script
- [ ] Create logs directory: `mkdir -p logs`
- [ ] Submit array job: `sbatch --array=0-N scripts/slurm/create_reservoir_training_dataset.slurm`
- [ ] Monitor progress: `squeue -u $USER`
- [ ] After completion, merge results
- [ ] Verify final output exists: `ls -lh data/processed/aidaying_iml1515_UB.npz`

---

## Advanced: Custom Parallelization

For fine-grained control, call the script directly with custom parameters:

```bash
# Process conditions 0-99 with 16 workers
python scripts/create_reservoir_training_dataset.py \
    configs/reservoir_training_dataset.yaml \
    --array-task-id 0 \
    --num-array-tasks 10 \
    --num-workers 16 \
    --checkpoint-interval 20
```

This is useful for:
- Testing different parallelization strategies
- Rerunning specific condition ranges
- Non-SLURM HPC environments (SGE, PBS, etc.)

---

## Questions?

If you encounter issues not covered here, check:
1. SLURM output logs in `logs/`
2. Python traceback in error logs
3. Partial result metadata in `.pkl` files
